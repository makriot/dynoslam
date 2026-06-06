"""
eval_gat_slam_mpc.py
====================
ПОЛНЫЙ DynoSLAM-пайплайн в замкнутой петле: GAT + DynamicSLAM + MPC.

В отличие от eval_gat_mpc.py (который управляет по ИДЕАЛЬНЫМ позициям пешеходов
прямо из симулятора), здесь робот НЕ видит истинных координат. Вместо этого:

  [симулятор] --(шумная одометрия + range-bearing наблюдения)--> [DynamicSLAM]
       SLAM на скользящем окне (факторный граф с GAT-кинематик-prior, Adam)
       --> денойзенная поза робота m*_robot и позиции пешеходов m*_k
       --> стохастический GAT по m*-истории --> (mu, Sigma) будущего
       --> GAT-MPC (мягкий Mahalanobis-барьер) --> u --> симулятор --> ...

Идея: SLAM фильтрует шум восприятия, поэтому управление должно остаться
робастным даже без идеальной перцепции. Сравнивается с теми же контроллерами
поверх CVM/MLP/идеального GAT.

ВАЖНО:
  * Робот ПЛАНИРУЕТ по SLAM-оценкам (m*_robot, mu, Sigma).
  * Все МЕТРИКИ (collision/near-miss/дистанция/цель) считаются по
    ground-truth позициям из симулятора — это честная оценка реальной
    безопасности при несовершенной локализации.
  * Модель шума одометрии/наблюдений идентична data_processor.py
    (на которой обучался/оценивался SLAM), для консистентности.

Это вычислительно тяжёлый эксперимент (SLAM = Adam-оптимизация на каждом окне),
поэтому есть --slam_every (как часто пересчитывать SLAM) и --slam_epochs.
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import argparse
from collections import deque

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import (
    HeadedSocialForceModelPolicy,
    RandomWaypointTracker,
    HSFMParams,
)

# переиспользуем оттестированные компоненты GAT-MPC
import eval_gat_mpc as egm
from eval_gat_mpc import StochasticGATPredictor, GATMPCController
from slam.nn_dynamic_slam import NeuralDynamicSLAM

# ── гиперпараметры ────────────────────────────────────────────────────────────
SIM_DT      = 0.01
MPC_DT      = 0.1
HORIZON     = 20
MAX_TIME    = 30.0
HISTORY_LEN = 5

R_GOAL      = 0.35
R_COLLISION = 0.35
R_NEAR_MISS = 0.8

MAXPEDS_MPC = egm.MAXPEDS_MPC   # 10
SIGMA_MIN   = egm.SIGMA_MIN

# SLAM / окно
WINDOW_SIZE = 10        # размер скользящего окна SLAM
SLAM_EPOCHS = 40        # эпох Adam на каждое окно
SLAM_EVERY  = 1         # пересчитывать SLAM раз в N вызовов MPC
N_ROLLOUTS  = 20
SIGMA_PERT  = egm.SIGMA_PERTURB  # 0.03 (tuned)

# модель шума сенсоров (как в data_processor.py)
MAX_RANGE     = 10.0
NOISE_ODOM_V  = 0.05
NOISE_ODOM_W  = 0.05
NOISE_OBS_R   = 0.05
NOISE_OBS_PHI = 0.02

# SLAM-сигмы (как в NeuralDynamicSLAM по умолчанию)
SIGMA_ODOM_V = 0.1
SIGMA_ODOM_W = 0.1
SIGMA_OBS_R  = 0.1
SIGMA_OBS_PHI = 0.05
SIGMA_KIN    = 0.2

# distribution (in-distribution по умолчанию)
NUM_EPISODES = 100
PED_COUNT    = (1, 15)
SPEED_RANGE  = (0.8, 1.5)
TAU_RANGE    = (0.2, 0.6)
WORLD_SIZE   = 10.0
BASE_SEED    = 42

WEIGHTS_PATH = "weights/gat_best.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV   = "results_gat_slam_mpc.csv"


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── генерация сценария (как в остальных скриптах) ─────────────────────────────
def generate_scenario():
    n_peds      = np.random.randint(PED_COUNT[0], PED_COUNT[1] + 1)
    robot_start = np.array([np.random.uniform(-4, -2), np.random.uniform(-4, 4), 0.0])
    robot_goal  = np.array([np.random.uniform(2, 4),  np.random.uniform(-4, 4), 0.0])
    ped_starts       = np.random.uniform(-3, 3, size=(n_peds, 3))
    ped_starts[:, 2] = np.random.uniform(-np.pi, np.pi, size=n_peds)
    ped_speeds       = np.random.uniform(SPEED_RANGE[0], SPEED_RANGE[1], size=n_peds)
    tau              = np.random.uniform(TAU_RANGE[0], TAU_RANGE[1])
    params           = HSFMParams.create_default()
    params.tau       = tau
    return dict(n_peds=n_peds, robot_start=robot_start, robot_goal=robot_goal,
                ped_starts=ped_starts, ped_speeds=ped_speeds, hsfm_params=params)


# ── модель сенсоров (реплика data_processor.py) ───────────────────────────────
def make_observations(robot_pose, ped_poses_dict):
    """
    Генерирует зашумлённые range-bearing наблюдения пешеходов в радиусе MAX_RANGE.
    Возвращает список dict {lm_id, range(tensor), bearing(tensor)} (формат SLAM).
    """
    rx, ry, rth = robot_pose
    obs = []
    for lm_id, pose in ped_poses_dict.items():
        dx, dy = pose[0] - rx, pose[1] - ry
        r = float(np.sqrt(dx * dx + dy * dy))
        if r <= MAX_RANGE:
            phi = wrap_angle(np.arctan2(dy, dx) - rth)
            obs.append({
                'lm_id': int(lm_id),
                'range':   torch.tensor(r + np.random.normal(0, NOISE_OBS_R),   dtype=torch.float32),
                'bearing': torch.tensor(phi + np.random.normal(0, NOISE_OBS_PHI), dtype=torch.float32),
            })
    return obs


def noisy_odometry(u_v, u_w):
    return [float(u_v + np.random.normal(0, NOISE_ODOM_V)),
            float(u_w + np.random.normal(0, NOISE_ODOM_W))]


# ── один эпизод closed-loop ───────────────────────────────────────────────────
def run_episode(episode_id, stoch_gat, slam, device, seed=None,
                slam_every=SLAM_EVERY, slam_epochs=SLAM_EPOCHS):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    cfg        = generate_scenario()
    robot_goal = cfg['robot_goal']

    robot_model = UnicycleRobotModel(initial_pose=cfg['robot_start'],
                                     initial_control=np.array([0.0, 0.0]))
    tracker   = RandomWaypointTracker(world_size=(WORLD_SIZE, WORLD_SIZE))
    ped_model = HeadedSocialForceModelPolicy(
        n_pedestrians=cfg['n_peds'], waypoint_tracker=tracker,
        initial_poses=cfg['ped_starts'],
        pedestrian_linear_velocity_magnitude=cfg['ped_speeds'],
        hsfm_params=cfg['hsfm_params'])
    sim = Simulation(sim_dt=SIM_DT, world_map=EmptyWorld(), robot_model=robot_model,
                     pedestrians_model=ped_model, sensors=[], rt_factor=None)

    controller = GATMPCController(dt=MPC_DT, goal=robot_goal, horizon=HORIZON)

    # буферы окна SLAM
    W = WINDOW_SIZE
    frames = deque(maxlen=W + 1)            # каждый: dict(obs, odom_in, pose_est)
    lm_est_hist = {}                        # {lmid: deque(maxlen=HISTORY_LEN)} оценок позиций
    est_pose = cfg['robot_start'].astype(np.float64).copy()  # текущая SLAM-оценка позы

    sim.step()  # прогрев Numba

    hold_time    = MPC_DT
    time_elapsed = 0.0
    u_pred       = np.array([0.0, 0.0])
    last_odom    = [0.0, 0.0]
    slam_call    = 0

    reached_goal    = False
    path_length     = 0.0
    prev_pos        = cfg['robot_start'][:2].copy()
    prev_omega      = 0.0
    smoothness_list = []
    min_dist_list   = []
    loc_err_list    = []   # ||SLAM-оценка позы - GT|| (валидация локализации)
    collision_steps = 0
    near_miss_steps = 0
    total_steps     = 0

    while time_elapsed < MAX_TIME:
        world = sim.current_state.world
        if world.robot is None or world.pedestrians is None:
            break
        if not np.isfinite(world.robot.pose).all():
            break

        robot_pose_gt  = world.robot.pose.copy()           # GT (только для метрик/сенсоров)
        ped_poses_dict = world.pedestrians.poses
        ped_poses_arr  = np.array([p[:2] for p in ped_poses_dict.values()])

        # ── метрики (по ground-truth) ───────────────────────────────────────
        if len(ped_poses_arr) > 0:
            dists = np.linalg.norm(ped_poses_arr - robot_pose_gt[:2], axis=1)
            min_d = dists.min()
            min_dist_list.append(min_d)
            if min_d < R_COLLISION:
                collision_steps += 1
            elif min_d < R_NEAR_MISS:
                near_miss_steps += 1
        total_steps += 1

        cur_pos      = robot_pose_gt[:2]
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos     = cur_pos.copy()
        omega = u_pred[1]
        smoothness_list.append((omega - prev_omega) / MPC_DT)
        prev_omega = omega

        # ── цель: проверяем по ИСТИННОЙ позе (реально ли доехал) ────────────
        if np.linalg.norm(robot_pose_gt[:2] - robot_goal[:2]) < R_GOAL:
            reached_goal = True
            break

        # ── вызов MPC с частотой MPC_DT ─────────────────────────────────────
        if hold_time >= MPC_DT:
            # 1) собираем зашумлённое наблюдение в текущем (истинном) состоянии
            obs = make_observations(robot_pose_gt, ped_poses_dict)
            frames.append({'obs': obs, 'odom_in': last_odom,
                           'pose_est': est_pose.copy()})

            mpc_pose = est_pose.copy()    # поза, по которой планирует MPC

            # 2) SLAM, когда окно заполнено
            run_slam = (len(frames) == W + 1) and (slam_call % slam_every == 0)
            if len(frames) == W + 1:
                slam_call += 1
            if run_slam:
                try:
                    init_pose = torch.tensor(frames[0]['pose_est'], dtype=torch.float32, device=device)
                    odom = torch.tensor([frames[j]['odom_in'] for j in range(1, W + 1)],
                                        dtype=torch.float32, device=device)          # [W,2]
                    observations = [frames[j]['obs'] for j in range(1, W + 1)]        # W шагов

                    # история перед окном — из накопленных SLAM-оценок
                    lm_history = None
                    if lm_est_hist:
                        lm_history = {lmid: torch.tensor(np.array(list(dq)), dtype=torch.float32)
                                      for lmid, dq in lm_est_hist.items() if len(dq) > 0}

                    robot_poses, landmarks, _ = slam(
                        init_pose, odom, observations, lm_history=lm_history,
                        num_epochs=slam_epochs, prediction_horizon=HORIZON,
                        stochastic=True)

                    # текущая SLAM-оценка позы робота = последняя в окне
                    est_pose = robot_poses[-1].cpu().numpy().astype(np.float64)
                    mpc_pose = est_pose.copy()
                    # сохраняем СКОРРЕКТИРОВАННУЮ позу как будущий якорь окна
                    # (иначе якорь = до-SLAM dead-reckoning → дрейф одометрии)
                    frames[-1]['pose_est'] = est_pose.copy()

                    # текущие оценки позиций пешеходов (последний кадр окна)
                    lm_ids = sorted({o['lm_id'] for st in observations for o in st})
                    cur_est = {}
                    for idx, lmid in enumerate(lm_ids):
                        pos = landmarks[-1, idx].cpu().numpy()
                        cur_est[lmid] = pos
                        if lmid not in lm_est_hist:
                            lm_est_hist[lmid] = deque(maxlen=HISTORY_LEN)
                        lm_est_hist[lmid].append(pos.astype(np.float32))

                    # 3) стохастический GAT по m*-истории -> mu, Sigma
                    track = [lmid for lmid in lm_ids if len(lm_est_hist[lmid]) > 0]
                    if track:
                        hist_list = []
                        for lmid in track:
                            dq = list(lm_est_hist[lmid])
                            if len(dq) < HISTORY_LEN:                  # паддинг повтором
                                dq = [dq[0]] * (HISTORY_LEN - len(dq)) + dq
                            hist_list.append(np.array(dq, dtype=np.float32))
                        hist_arr = np.stack(hist_list, axis=0)         # (K,H,2)
                        mu, cov = stoch_gat.predict(hist_arr, n_steps=HORIZON)

                        # nearest-10 по SLAM-оценке робота
                        cur_peds = hist_arr[:, -1, :]
                        d = np.linalg.norm(cur_peds - mpc_pose[:2], axis=1)
                        sel = np.argsort(d)[:MAXPEDS_MPC]
                        controller.update_gat_predictions(mu[sel], cov[sel])
                except Exception:
                    pass  # при сбое SLAM держим предыдущие TVP/позу

            # валидация: ошибка локализации (SLAM-оценка vs GT)
            loc_err_list.append(float(np.linalg.norm(mpc_pose[:2] - robot_pose_gt[:2])))

            # 4) MPC по SLAM-оценке позы
            try:
                u_pred = controller.predict(mpc_pose)
            except Exception:
                u_pred = np.array([0.0, 0.0])

            # одометрия для следующего перехода + дед-реконинг est_pose (если SLAM не звали)
            last_odom = noisy_odometry(u_pred[0], u_pred[1])
            v, w = last_odom
            th = est_pose[2]
            est_pose = np.array([est_pose[0] + v * np.cos(th) * MPC_DT,
                                 est_pose[1] + v * np.sin(th) * MPC_DT,
                                 wrap_angle(th + w * MPC_DT)])
            hold_time = 0.0

        sim.step(u_pred)
        hold_time    += SIM_DT
        time_elapsed += SIM_DT

    return {
        'episode_id':     episode_id,
        'goal_reached':   int(reached_goal),
        'time_to_goal':   time_elapsed if reached_goal else float('nan'),
        'path_length':    path_length,
        'avg_min_dist':   float(np.mean(min_dist_list)) if min_dist_list else float('nan'),
        'loc_err':        float(np.mean(loc_err_list)) if loc_err_list else float('nan'),
        'collision_rate': collision_steps / max(total_steps, 1),
        'near_miss_rate': near_miss_steps / max(total_steps, 1),
        'avg_speed':      path_length / max(time_elapsed, 1e-6),
        'smoothness_rms': float(np.sqrt(np.mean(np.array(smoothness_list)**2)))
                          if smoothness_list else float('nan'),
    }


def main():
    global PED_COUNT, SPEED_RANGE, TAU_RANGE, WORLD_SIZE, MAX_TIME, HORIZON
    parser = argparse.ArgumentParser(description='Closed-loop GAT + DynamicSLAM + MPC')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES)
    parser.add_argument('--output',   type=str, default=OUTPUT_CSV)
    parser.add_argument('--weights',  type=str, default=WEIGHTS_PATH)
    parser.add_argument('--device',   type=str, default=DEVICE)
    parser.add_argument('--rollouts', type=int, default=N_ROLLOUTS)
    parser.add_argument('--sigma_perturb', type=float, default=SIGMA_PERT)
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--window', type=int, default=WINDOW_SIZE, help='окно SLAM')
    parser.add_argument('--slam_epochs', type=int, default=SLAM_EPOCHS,
                        help='эпох Adam на окно SLAM')
    parser.add_argument('--slam_every', type=int, default=SLAM_EVERY,
                        help='пересчитывать SLAM раз в N вызовов MPC')
    parser.add_argument('--ped_count', type=int, nargs=2, default=list(PED_COUNT), metavar=('MIN', 'MAX'))
    parser.add_argument('--speed_range', type=float, nargs=2, default=list(SPEED_RANGE), metavar=('MIN', 'MAX'))
    parser.add_argument('--tau_range', type=float, nargs=2, default=list(TAU_RANGE), metavar=('MIN', 'MAX'))
    parser.add_argument('--world_size', type=float, default=WORLD_SIZE)
    parser.add_argument('--max_time', type=float, default=MAX_TIME)
    parser.add_argument('--horizon', type=int, default=HORIZON)
    args = parser.parse_args()

    PED_COUNT   = tuple(args.ped_count)
    SPEED_RANGE = tuple(args.speed_range)
    TAU_RANGE   = tuple(args.tau_range)
    WORLD_SIZE  = args.world_size
    MAX_TIME    = args.max_time
    HORIZON     = args.horizon

    print(f"[GAT+SLAM+MPC] device={args.device} rollouts={args.rollouts} "
          f"window={args.window} slam_epochs={args.slam_epochs} slam_every={args.slam_every}")
    print(f"[GAT+SLAM+MPC] episodes={args.episodes} ped_count={PED_COUNT} "
          f"speed={SPEED_RANGE} tau={TAU_RANGE} world={WORLD_SIZE} horizon={HORIZON}")

    # один GAT-backbone и для SLAM-prior, и для стохастического предсказания
    stoch_gat = StochasticGATPredictor(
        weights_path=args.weights, history_len=HISTORY_LEN, dt=MPC_DT,
        device=args.device, n_rollouts=args.rollouts, sigma_perturb=args.sigma_perturb)

    slam = NeuralDynamicSLAM(
        stoch_gat.model, window_size=args.window, dt=MPC_DT,
        sigma_odom_v=SIGMA_ODOM_V, sigma_odom_w=SIGMA_ODOM_W,
        sigma_obs_r=SIGMA_OBS_R, sigma_obs_phi=SIGMA_OBS_PHI,
        sigma_kin=SIGMA_KIN).to(args.device)

    print(f"[GAT+SLAM+MPC] Запуск {args.episodes} эпизодов...")
    records = []
    for ep in tqdm(range(args.episodes)):
        try:
            rec = run_episode(ep, stoch_gat, slam, args.device, seed=ep + args.seed,
                              slam_every=args.slam_every, slam_epochs=args.slam_epochs)
            records.append(rec)
        except Exception as e:
            print(f"  Эпизод {ep} упал: {e}")

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*55}")
    print(f"  GAT + DynamicSLAM + MPC — итоги ({len(df)} эпизодов)")
    print(f"{'='*55}")
    print(f"  Goal Reach Rate    : {df['goal_reached'].mean():.3f}")
    print(f"  Time to Goal       : {df['time_to_goal'].mean():.2f} с")
    print(f"  Path Length        : {df['path_length'].mean():.2f} м")
    print(f"  Avg Min Dist       : {df['avg_min_dist'].mean():.3f} м")
    print(f"  Loc Error (SLAM)   : {df['loc_err'].mean():.3f} м")
    print(f"  Collision Rate     : {df['collision_rate'].mean():.4f}")
    print(f"  Near-Miss Rate     : {df['near_miss_rate'].mean():.4f}")
    print(f"  Avg Speed          : {df['avg_speed'].mean():.3f} м/с")
    print(f"  Smoothness RMS     : {df['smoothness_rms'].mean():.4f} rad/s²")
    print(f"{'='*55}")
    print(f"  Сохранено → {args.output}")


if __name__ == '__main__':
    main()

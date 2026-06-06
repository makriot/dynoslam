"""
eval_slam_mpc.py
================
Closed-loop DynoSLAM + MPC с выбираемым кинематик-prior: CVM / MLP / GAT.

Робот видит только ШУМНЫЕ range-bearing измерения пешеходов (+ шумная
одометрия). DynamicSLAM на скользящем окне ДЕНОЙЗИТ позиции пешеходов
(факторный граф, Adam), после чего предиктор строит будущее, а MPC управляет.

Поза робота инициализируется каждым окном из static-локализации
(GT + опц. малый шум) с сильным prior (как в офлайн-eval, вес 1e6) —
SLAM её уточняет, но не локализует с нуля. Новизна/выигрыш меряется в
ДЕНОЙЗЕ ПЕШЕХОДОВ -> прогнозе -> управлении.

Конфиги (--predictor):
  cvm : CVDynamicSLAM            -> CVM-прогноз      -> фикс. радиус (MLP-контроллер)
  mlp : NeuralDynamicSLAM(MLP)   -> MLP-прогноз      -> фикс. радиус
  gat : NeuralDynamicSLAM(GAT)   -> стох. GAT (mu,Σ) -> Mahalanobis-барьер

Все метрики (collision/near-miss/дистанция/цель) — по ground-truth.
Доп.: ped_est_err (денойз) и ADE/FDE (прогноз).
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
    HeadedSocialForceModelPolicy, RandomWaypointTracker, HSFMParams,
)

import perception_utils as P
import eval_gat_mpc as egm
from eval_gat_mpc import StochasticGATPredictor, GATMPCController
from eval_mlp_mpc import MLPMPCController
from models.single_agent_model import MLPVelocityPredictor
from models.multi_agent_model import SimpleGATPredictor
from slam.nn_dynamic_slam import NeuralDynamicSLAM
from slam.cv_dynamic_slam import CVDynamicSLAM

# ── гиперпараметры ────────────────────────────────────────────────────────────
SIM_DT      = 0.01
MPC_DT      = 0.1
HORIZON     = 20
MAX_TIME    = 30.0
HISTORY_LEN = 5

R_GOAL      = 0.35
R_COLLISION = 0.35
R_NEAR_MISS = 0.8

MAXPEDS_MPC = egm.MAXPEDS_MPC

# наследуем сетап из main_eval_gat.py (на нём считались метрики статьи)
WINDOW_SIZE = 20        # окно SLAM (как в офлайн-eval)
SLAM_EPOCHS = 100       # эпох Adam на окно (как в офлайн-eval)
SLAM_EVERY  = 1
N_ROLLOUTS  = 20
SIGMA_PERT  = egm.SIGMA_PERTURB

# SLAM-сигмы (как в main_eval_gat.py)
SIGMA_ODOM_V = 0.1
SIGMA_ODOM_W = 0.1
SIGMA_OBS_R  = 0.1
SIGMA_OBS_PHI = 0.05
SIGMA_KIN_GAT = 0.5     # доверие к GAT-prior (мягкое — как в офлайне)
SIGMA_KIN_MLP = 0.1     # доверие к MLP-prior
SIGMA_ACC_CVM = 0.5     # CVM acceleration prior

POSE_INIT_NOISE = 0.0   # σ шума на init-позе робота (суррогат static-локализации)

NUM_EPISODES = 100
PED_COUNT    = (1, 15)
SPEED_RANGE  = (0.8, 1.5)
TAU_RANGE    = (0.2, 0.6)
WORLD_SIZE   = 10.0
BASE_SEED    = 42

GAT_WEIGHTS  = "weights/gat_best.pth"
MLP_WEIGHTS  = "weights/mlp_best.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV   = "results_slam_mpc.csv"


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


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


def run_episode(episode_id, predictor_kind, slam, stoch_gat, controller_factory,
                device, seed=None, slam_every=SLAM_EVERY, slam_epochs=SLAM_EPOCHS,
                pose_init_noise=POSE_INIT_NOISE, noise_scale=1.0):
    sr   = P.NOISE_OBS_R   * noise_scale
    sphi = P.NOISE_OBS_PHI * noise_scale
    sv   = P.NOISE_ODOM_V  * noise_scale
    sw   = P.NOISE_ODOM_W  * noise_scale
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    metrics  = P.PerceptionMetrics()
    mpc_step = 0

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

    controller = controller_factory(robot_goal)
    is_gat     = (predictor_kind == 'gat')

    W = WINDOW_SIZE
    frames      = deque(maxlen=W + 1)        # dict(obs, odom_in, pose_gt)
    prev_smooth = {}                         # {lmid: (H,2)} гладкий внутри-оконный
                                             # m*-сегмент из ПРЕДЫДУЩЕГО окна (для GAT-prior)
    est_pose    = cfg['robot_start'].astype(np.float64).copy()

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
    loc_err_list    = []
    collision_steps = 0
    near_miss_steps = 0
    total_steps     = 0

    while time_elapsed < MAX_TIME:
        world = sim.current_state.world
        if world.robot is None or world.pedestrians is None:
            break
        if not np.isfinite(world.robot.pose).all():
            break

        robot_pose_gt  = world.robot.pose.copy()
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

        if np.linalg.norm(robot_pose_gt[:2] - robot_goal[:2]) < R_GOAL:
            reached_goal = True
            break

        if hold_time >= MPC_DT:
            # измерение в текущем состоянии (noise_scale=0 -> чистое)
            obs = P.measure_peds(robot_pose_gt, ped_poses_dict, sigma_r=sr, sigma_phi=sphi)
            frames.append({'obs': obs, 'odom_in': last_odom,
                           'pose_gt': robot_pose_gt.copy()})

            mpc_pose = est_pose.copy()

            run_slam = (len(frames) == W + 1) and (slam_call % slam_every == 0)
            if len(frames) == W + 1:
                slam_call += 1
            if run_slam:
                try:
                    # init-поза = static-локализация (GT + опц. шум), prior 1e6
                    init_np = frames[0]['pose_gt'].astype(np.float64).copy()
                    if pose_init_noise > 0:
                        init_np[:2] += np.random.normal(0, pose_init_noise, 2)
                    init_pose = torch.tensor(init_np, dtype=torch.float32, device=device)
                    odom = torch.tensor([frames[j]['odom_in'] for j in range(1, W + 1)],
                                        dtype=torch.float32, device=device)
                    observations = []
                    for j in range(1, W + 1):
                        observations.append([
                            {'lm_id': o['lm_id'],
                             'range':   torch.tensor(o['range'],   dtype=torch.float32),
                             'bearing': torch.tensor(o['bearing'], dtype=torch.float32)}
                            for o in frames[j]['obs']])

                    # lm_history = ГЛАДКИЙ внутри-оконный m*-сегмент из предыдущего
                    # окна (совместно оптимизирован → без скачков; это и есть
                    # {m*_{i-H..i-1}} из статьи, поданный консистентно)
                    lm_history = None
                    if is_gat and prev_smooth:
                        lm_history = {lmid: torch.tensor(seg, dtype=torch.float32)
                                      for lmid, seg in prev_smooth.items()}

                    if predictor_kind == 'cvm':
                        robot_poses, landmarks, preds = slam(
                            init_pose, odom, observations, lm_history=None,
                            num_epochs=slam_epochs, prediction_horizon=HORIZON)
                    elif predictor_kind == 'mlp':
                        robot_poses, landmarks, preds = slam(
                            init_pose, odom, observations, lm_history=lm_history,
                            num_epochs=slam_epochs, prediction_horizon=HORIZON,
                            stochastic=False)
                    else:  # gat
                        robot_poses, landmarks, preds = slam(
                            init_pose, odom, observations, lm_history=lm_history,
                            num_epochs=slam_epochs, prediction_horizon=HORIZON,
                            stochastic=True)

                    est_pose = robot_poses[-1].cpu().numpy().astype(np.float64)
                    mpc_pose = est_pose.copy()

                    lm_ids = sorted({o['lm_id'] for st in observations for o in st})
                    win = landmarks.detach().cpu().numpy()          # (W+1, M, 2)
                    cur_est = {}
                    prev_smooth = {}
                    for idx, lmid in enumerate(lm_ids):
                        cur_est[lmid] = win[-1, idx]
                        # гладкий внутри-оконный сегмент m* (последние H позиций)
                        prev_smooth[lmid] = win[-HISTORY_LEN:, idx].astype(np.float32)

                    # ── метрики денойза ───────────────────────────────────────
                    metrics.log_estimate(mpc_step, cur_est, ped_poses_dict)

                    if is_gat and len(lm_ids) > 0:
                        # mu, Sigma — ТОЙ ЖЕ рабочей функцией, что в eval_gat_mpc
                        # (StochasticGATPredictor.predict: Monte-Carlo роллауты с
                        # sigma_perturb), на гладкой внутри-оконной истории m*.
                        H = HISTORY_LEN
                        win = landmarks.detach().cpu().numpy()          # (W+1, M, 2)
                        hist_arr = np.transpose(win[-H:], (1, 0, 2))    # (M, H, 2)
                        mu, cov = stoch_gat.predict(hist_arr, n_steps=HORIZON)
                        metrics.log_prediction(mpc_step,
                            {lmid: mu[i] for i, lmid in enumerate(lm_ids)})
                        cur_peds = hist_arr[:, -1, :]
                        d = np.linalg.norm(cur_peds - mpc_pose[:2], axis=1)
                        sel = np.argsort(d)[:MAXPEDS_MPC]
                        controller.update_gat_predictions(mu[sel], cov[sel])
                    else:
                        # CVM/MLP: используем предсказания SLAM (mean-траектории)
                        track = [lmid for lmid in lm_ids if lmid in preds]
                        if track:
                            pred_np = {lmid: preds[lmid].cpu().numpy() for lmid in track}
                            metrics.log_prediction(mpc_step, pred_np)
                            cur_arr = np.array([cur_est[lmid] for lmid in track])
                            pred_arr = np.stack([pred_np[lmid] for lmid in track], axis=0)  # (K,H,2)
                            d = np.linalg.norm(cur_arr - mpc_pose[:2], axis=1)
                            sel = np.argsort(d)[:MAXPEDS_MPC]
                            controller.update_predictions(cur_arr[sel], pred_arr[sel])
                except Exception:
                    pass  # при сбое SLAM держим предыдущие TVP/позу

            loc_err_list.append(float(np.linalg.norm(mpc_pose[:2] - robot_pose_gt[:2])))

            try:
                u_pred = controller.predict(mpc_pose)
            except Exception:
                u_pred = np.array([0.0, 0.0])

            # одометрия следующего перехода + дед-реконинг est_pose (между SLAM)
            last_odom = P.noisy_odometry(u_pred[0], u_pred[1], sigma_v=sv, sigma_w=sw)
            v, w = last_odom
            th = est_pose[2]
            est_pose = np.array([est_pose[0] + v * np.cos(th) * MPC_DT,
                                 est_pose[1] + v * np.sin(th) * MPC_DT,
                                 wrap_angle(th + w * MPC_DT)])
            mpc_step += 1
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
        **metrics.finalize(),
    }


def build_slam_and_controller(kind, device, args):
    """Возвращает (slam, stoch_gat, controller_factory)."""
    if kind == 'cvm':
        slam = CVDynamicSLAM(window_size=args.window, dt=MPC_DT,
                             sigma_odom_v=SIGMA_ODOM_V, sigma_odom_w=SIGMA_ODOM_W,
                             sigma_obs_r=SIGMA_OBS_R, sigma_obs_phi=SIGMA_OBS_PHI,
                             sigma_acc=SIGMA_ACC_CVM).to(device)
        stoch = None
        factory = lambda goal: MLPMPCController(dt=MPC_DT, goal=goal, horizon=HORIZON)
    elif kind == 'mlp':
        mlp = MLPVelocityPredictor(history_len=HISTORY_LEN, dt=MPC_DT, hidden_dim=64).to(device)
        mlp.load_state_dict(torch.load(args.mlp_weights, map_location=device))
        mlp.eval()
        slam = NeuralDynamicSLAM(mlp, window_size=args.window, dt=MPC_DT,
                                 sigma_odom_v=SIGMA_ODOM_V, sigma_odom_w=SIGMA_ODOM_W,
                                 sigma_obs_r=SIGMA_OBS_R, sigma_obs_phi=SIGMA_OBS_PHI,
                                 sigma_kin=SIGMA_KIN_MLP).to(device)
        stoch = None
        factory = lambda goal: MLPMPCController(dt=MPC_DT, goal=goal, horizon=HORIZON)
    elif kind == 'gat':
        stoch = StochasticGATPredictor(weights_path=args.gat_weights, history_len=HISTORY_LEN,
                                       dt=MPC_DT, device=device, n_rollouts=args.rollouts,
                                       sigma_perturb=args.sigma_perturb)
        slam = NeuralDynamicSLAM(stoch.model, window_size=args.window, dt=MPC_DT,
                                 sigma_odom_v=SIGMA_ODOM_V, sigma_odom_w=SIGMA_ODOM_W,
                                 sigma_obs_r=SIGMA_OBS_R, sigma_obs_phi=SIGMA_OBS_PHI,
                                 sigma_kin=SIGMA_KIN_GAT).to(device)
        factory = lambda goal: GATMPCController(dt=MPC_DT, goal=goal, horizon=HORIZON)
    else:
        raise ValueError(f"unknown predictor: {kind}")
    return slam, stoch, factory


def main():
    global PED_COUNT, SPEED_RANGE, TAU_RANGE, WORLD_SIZE, MAX_TIME, HORIZON
    parser = argparse.ArgumentParser(description='Closed-loop DynoSLAM + MPC (cvm/mlp/gat)')
    parser.add_argument('--predictor', choices=['cvm', 'mlp', 'gat'], required=True)
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES)
    parser.add_argument('--output',   type=str, default=None)
    parser.add_argument('--gat_weights', type=str, default=GAT_WEIGHTS)
    parser.add_argument('--mlp_weights', type=str, default=MLP_WEIGHTS)
    parser.add_argument('--device',   type=str, default=DEVICE)
    parser.add_argument('--rollouts', type=int, default=N_ROLLOUTS)
    parser.add_argument('--sigma_perturb', type=float, default=SIGMA_PERT)
    parser.add_argument('--avoid_w', type=float, default=egm.AVOID_W,
                        help='вес Mahalanobis-барьера GAT (ниже → менее осторожно → выше reach)')
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--window', type=int, default=WINDOW_SIZE)
    parser.add_argument('--slam_epochs', type=int, default=SLAM_EPOCHS)
    parser.add_argument('--slam_every', type=int, default=SLAM_EVERY)
    parser.add_argument('--pose_init_noise', type=float, default=POSE_INIT_NOISE,
                        help='σ шума на init-позе робота (суррогат static-локализации)')
    parser.add_argument('--noise_scale', type=float, default=1.0,
                        help='множитель шума сенсоров (0 = чистые данные, для проверки)')
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
    egm.AVOID_W = args.avoid_w   # вес барьера GAT (GATMPCController читает его при создании)
    output = args.output or f"results_slam_{args.predictor}.csv"

    print(f"[SLAM+MPC] predictor={args.predictor}  device={args.device}  "
          f"window={args.window}  slam_epochs={args.slam_epochs}  slam_every={args.slam_every}")
    print(f"[SLAM+MPC] episodes={args.episodes}  ped_count={PED_COUNT}  world={WORLD_SIZE}  "
          f"horizon={HORIZON}  pose_init_noise={args.pose_init_noise}")

    slam, stoch, factory = build_slam_and_controller(args.predictor, args.device, args)

    records = []
    for ep in tqdm(range(args.episodes)):
        try:
            rec = run_episode(ep, args.predictor, slam, stoch, factory, args.device,
                              seed=ep + args.seed, slam_every=args.slam_every,
                              slam_epochs=args.slam_epochs, pose_init_noise=args.pose_init_noise,
                              noise_scale=args.noise_scale)
            records.append(rec)
        except Exception as e:
            print(f"  Эпизод {ep} упал: {e}")

    df = pd.DataFrame(records)
    df.to_csv(output, index=False)

    print(f"\n{'='*55}")
    print(f"  DynoSLAM+MPC [{args.predictor}] — итоги ({len(df)} эпизодов)")
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
    print(f"  Ped Est Error      : {df['ped_est_err'].mean():.4f} м")
    print(f"  ADE / FDE          : {df['ade'].mean():.4f} / {df['fde'].mean():.4f} м")
    print(f"{'='*55}")
    print(f"  Сохранено → {output}")


if __name__ == '__main__':
    main()

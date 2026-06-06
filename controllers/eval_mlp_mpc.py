"""
eval_mlp_mpc.py
===============
Single-Agent Neural Prior (MLP) + MPC.

Промежуточный baseline между CVM и GAT: вместо линейной экстраполяции (CVM)
будущие позиции пешеходов предсказывает однопешеходная нейросеть MLP
(без социального контекста). Предсказание ДЕТЕРМИНИРОВАННОЕ, поэтому
ковариации нет — используется тот же жёсткий барьер с фиксированным
радиусом R_SAFE, что и в eval_plain_mpc.py:

  J = sum_t [ goal_weight * dist_to_goal(t)
            + sum_k PENALTY_W * max(0, R_SAFE - ||p_robot - p_k,t||)^3 ]

Предсказание пешеходов (eq. Single Landmark Velocity Prediction):
  v_k,i   = s_theta(H_k,i)             — MLP по истории скоростей пешехода k
  m_k,t   = m_k,i-1 + sum v * dt       — авторегрессивная интеграция

MPC «видит» 10 БЛИЖАЙШИХ к роботу пешеходов (как и eval_plain/eval_gat).
Метрики идентичны остальным скриптам для корректного сравнения.
"""

import sys
import os

# pyminisim лежит локально — добавляем корень проекта в путь
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import argparse
from collections import deque

import numpy as np
import pandas as pd
import casadi
import do_mpc          # пакет называется do_mpc (с подчёркиванием)
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

from models.single_agent_model import MLPVelocityPredictor
import perception_utils as P

# ── гиперпараметры ────────────────────────────────────────────────────────────
SIM_DT      = 0.01
MPC_DT      = 0.1
HORIZON     = 20
MAX_TIME    = 30.0
HISTORY_LEN = 5

GOAL_WEIGHT = 3.0
PENALTY_W   = 5000.0  # штраф за нарушение R_SAFE
R_SAFE      = 0.8     # м, фиксированный радиус мягкого избегания
R_GOAL      = 0.35
R_COLLISION = 0.35
R_NEAR_MISS = 0.8

MAXPEDS_MPC = 10      # максимум пешеходов в TVP (берутся ближайшие к роботу)
HIDDEN_DIM  = 64      # как при обучении (train.py)

# ── distribution-параметры (по умолчанию = IN-DISTRIBUTION, как при генерации
#    обучающих данных: ped 1–15, арена 10×10, speed 0.8–1.5, tau 0.2–0.6).
#    Все переопределяются через CLI (см. main) — для OOD-прогона.
NUM_EPISODES = 100
PED_COUNT    = (1, 15)     # число пешеходов на эпизод [min, max]
SPEED_RANGE  = (0.8, 1.5)  # желаемая скорость пешеходов HSFM
TAU_RANGE    = (0.2, 0.6)  # параметр tau HSFM
WORLD_SIZE   = 10.0        # размер арены waypoint-tracker'а (как в data_generator)
BASE_SEED    = 42          # seed эпизода = BASE_SEED + episode_id

WEIGHTS_PATH = "weights/mlp_best.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_CSV   = "results_mlp_mpc.csv"


# ── история пешеходов (eq. Landmark History H_k,i) ───────────────────────────
class PedHistoryBuffer:
    """
    Sliding-window буфер позиций пешеходов.

    H_k,i = [m_k,i-H, ..., m_k,i-1]  — eq. Landmark History Formulation

    При появлении нового пешехода буфер заполняется его текущей позицией.
    """

    def __init__(self, history_len: int):
        self.history_len = history_len
        self._buf: dict[int, deque] = {}

    def update(self, ped_poses_dict: dict):
        for pid, pose in ped_poses_dict.items():
            pos = np.array(pose[:2], dtype=np.float32)
            if pid not in self._buf:
                self._buf[pid] = deque(
                    [pos.copy() for _ in range(self.history_len)],
                    maxlen=self.history_len
                )
            else:
                self._buf[pid].append(pos.copy())

    def get_tensor(self):
        """
        Возвращает (arr, pids):
          arr  : np.ndarray (K, H, 2)
          pids : list[int]  — отсортированные id пешеходов
        """
        if not self._buf:
            return None, []
        pids = sorted(self._buf.keys())
        arr  = np.stack(
            [np.array(list(self._buf[pid]), dtype=np.float32) for pid in pids],
            axis=0
        )   # (K, H, 2)
        return arr, pids


# ── MLP-предсказатель: детерминированный rollout позиций ──────────────────────
class MLPPredictor:
    """
    Обёртка над MLPVelocityPredictor (single-agent).

    Предсказывает скорости на n_steps вперёд для каждого пешехода независимо
    (без социального контекста) и интегрирует их в позиции:
      v_k,t   = s_theta(H_k)
      m_k,t   = m_k,i-1 + (sum_{<=t} v) * dt
    """

    def __init__(self, weights_path: str, history_len: int, dt: float,
                 device: str = 'cpu', hidden_dim: int = HIDDEN_DIM):
        self.dt          = dt
        self.device      = device
        self.history_len = history_len

        self.model = MLPVelocityPredictor(
            history_len=history_len,
            dt=dt,
            hidden_dim=hidden_dim,
        ).to(device)

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state)
            print(f"[MLP] Веса загружены: {weights_path}")
        else:
            print(f"[MLP] ВНИМАНИЕ: веса не найдены в {weights_path}, "
                  f"используется случайная инициализация.")
        self.model.eval()

    @torch.no_grad()
    def predict(self, ped_history: np.ndarray, n_steps: int) -> np.ndarray:
        """
        ped_history : (K, H, 2)
        n_steps     : T

        Возвращает predicted positions: (K, T, 2)
        """
        hist = torch.tensor(ped_history, dtype=torch.float32, device=self.device)
        vels = self.model(hist, num_steps=n_steps)          # (K, T, 2)
        vels = vels.cpu().numpy()

        last_p = ped_history[:, -1, :]                       # (K, 2)
        disp   = np.cumsum(vels, axis=1) * self.dt           # (K, T, 2)
        positions = last_p[:, None, :] + disp                # (K, T, 2)
        return positions


# ── контроллер MLP + MPC (фиксированный жёсткий барьер) ───────────────────────
class MLPMPCController:
    """
    MPC с детерминированным предсказанием позиций пешеходов и фиксированным
    радиусом R_SAFE (как в Plain MPC, но позиции — из MLP, а не из CVM).

    Unicycle-динамика:
      px_{t+1}  = px_t  + uv * cos(pth_t) * dt
      py_{t+1}  = py_t  + uv * sin(pth_t) * dt
      pth_{t+1} = pth_t + uomg * dt

    TVP: предсказанные MLP позиции пешеходов на каждом шаге горизонта.
    """

    def __init__(self, dt: float, goal: np.ndarray,
                 horizon: int = HORIZON, goal_weight: float = GOAL_WEIGHT):
        self.dt      = dt
        self.horizon = horizon
        self.goal    = goal.copy()

        # ── модель ──────────────────────────────────────────────────────────
        model = do_mpc.model.Model('discrete')
        px    = model.set_variable('_x', 'px')
        py    = model.set_variable('_x', 'py')
        pth   = model.set_variable('_x', 'pth')
        uv    = model.set_variable('_u', 'uv')
        uomg  = model.set_variable('_u', 'uomg')

        # TVP: предсказанные позиции пешеходов (передаются снаружи)
        self.tvp_ped_x = [
            model.set_variable('_tvp', f'ped_x_{i}') for i in range(MAXPEDS_MPC)
        ]
        self.tvp_ped_y = [
            model.set_variable('_tvp', f'ped_y_{i}') for i in range(MAXPEDS_MPC)
        ]

        model.set_rhs('px',  px  + uv * casadi.cos(pth) * dt)
        model.set_rhs('py',  py  + uv * casadi.sin(pth) * dt)
        model.set_rhs('pth', pth + uomg * dt)

        # ── функция стоимости (жёсткий барьер с фиксированным R_SAFE) ────────
        cost = goal_weight * casadi.sqrt(
            (px - goal[0])**2 + (py - goal[1])**2 + 1e-6
        )
        for i in range(MAXPEDS_MPC):
            dist_i = casadi.sqrt(
                (px - self.tvp_ped_x[i])**2 +
                (py - self.tvp_ped_y[i])**2 + 1e-6
            )
            cost += PENALTY_W * casadi.fmax(0.0, R_SAFE - dist_i)**3

        model.set_expression('cost', cost)
        model.setup()

        # ── MPC ─────────────────────────────────────────────────────────────
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_robust=0,
            n_horizon=horizon,
            t_step=dt,
            state_discretization='discrete',
            store_full_solution=True,
            nlpsol_opts={
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
                'print_time': 0,
            }
        )
        mpc.set_objective(mterm=model.aux['cost'], lterm=model.aux['cost'])

        mpc.bounds['lower', '_u', 'uv']   = 0.0
        mpc.bounds['upper', '_u', 'uv']   = 1.8
        mpc.bounds['lower', '_u', 'uomg'] = -np.deg2rad(50.0)
        mpc.bounds['upper', '_u', 'uomg'] =  np.deg2rad(50.0)
        mpc.bounds['lower', '_x', 'pth']  = -np.pi
        mpc.bounds['upper', '_x', 'pth']  =  np.pi
        mpc.set_rterm(uv=1e-4, uomg=1e-4)

        # буфер предсказаний; по умолчанию "уводим" пешеходов за горизонт.
        # ВАЖНО: инициализируем ДО set_tvp_fun/setup, т.к. mpc.setup()
        # вызывает _tvp_fun и обращается к этому буферу.
        self._ped_pred = np.full((MAXPEDS_MPC, horizon + 1, 2), 100.0)

        self.tvp_template = mpc.get_tvp_template()
        mpc.set_tvp_fun(self._tvp_fun)
        mpc.setup()

        self.mpc   = mpc
        self.model = model

    # ── TVP callback ────────────────────────────────────────────────────────
    def _tvp_fun(self, tnow):
        for t in range(self.horizon + 1):
            for j in range(MAXPEDS_MPC):
                self.tvp_template['_tvp', t, f'ped_x_{j}'] = self._ped_pred[j, t, 0]
                self.tvp_template['_tvp', t, f'ped_y_{j}'] = self._ped_pred[j, t, 1]
        return self.tvp_template

    def update_predictions(self, cur_positions: np.ndarray,
                           pred_positions: np.ndarray):
        """
        cur_positions  : (n, 2)     — текущие позиции (ближайшие n<=MAXPEDS пешеходов)
        pred_positions : (n, T, 2)  — предсказанные MLP позиции на горизонт
        """
        self._ped_pred[:] = 100.0
        n        = min(pred_positions.shape[0], MAXPEDS_MPC)
        n_steps  = min(pred_positions.shape[1], self.horizon)
        for j in range(n):
            self._ped_pred[j, 0] = cur_positions[j]
            for t in range(1, n_steps + 1):
                self._ped_pred[j, t] = pred_positions[j, t - 1]

    def predict(self, x_current: np.ndarray) -> np.ndarray:
        self.mpc.x0 = x_current
        self.mpc.set_initial_guess()
        u0 = self.mpc.make_step(x_current)
        return u0.flatten()


# ── генерация сценария ────────────────────────────────────────────────────────
def generate_scenario():
    n_peds      = np.random.randint(PED_COUNT[0], PED_COUNT[1] + 1)
    robot_start = np.array([
        np.random.uniform(-4, -2),
        np.random.uniform(-4,  4),
        0.0
    ])
    robot_goal  = np.array([
        np.random.uniform(2, 4),
        np.random.uniform(-4, 4),
        0.0
    ])
    ped_starts       = np.random.uniform(-3, 3, size=(n_peds, 3))
    ped_starts[:, 2] = np.random.uniform(-np.pi, np.pi, size=n_peds)
    ped_speeds       = np.random.uniform(SPEED_RANGE[0], SPEED_RANGE[1], size=n_peds)
    tau              = np.random.uniform(TAU_RANGE[0], TAU_RANGE[1])
    params           = HSFMParams.create_default()
    params.tau       = tau
    return dict(
        n_peds=n_peds,
        robot_start=robot_start,
        robot_goal=robot_goal,
        ped_starts=ped_starts,
        ped_speeds=ped_speeds,
        hsfm_params=params,
    )


# ── один эпизод ───────────────────────────────────────────────────────────────
def run_episode(episode_id: int, mlp: MLPPredictor, seed: int = None,
                noisy: bool = False) -> dict:
    if seed is not None:
        np.random.seed(seed)

    metrics  = P.PerceptionMetrics()
    mpc_step = 0

    cfg        = generate_scenario()
    robot_goal = cfg['robot_goal']

    robot_model = UnicycleRobotModel(
        initial_pose=cfg['robot_start'],
        initial_control=np.array([0.0, 0.0])
    )
    tracker   = RandomWaypointTracker(world_size=(WORLD_SIZE, WORLD_SIZE))
    ped_model = HeadedSocialForceModelPolicy(
        n_pedestrians=cfg['n_peds'],
        waypoint_tracker=tracker,
        initial_poses=cfg['ped_starts'],
        pedestrian_linear_velocity_magnitude=cfg['ped_speeds'],
        hsfm_params=cfg['hsfm_params'],
    )
    sim = Simulation(
        sim_dt=SIM_DT,
        world_map=EmptyWorld(),
        robot_model=robot_model,
        pedestrians_model=ped_model,
        sensors=[],
        rt_factor=None,
    )

    controller  = MLPMPCController(dt=MPC_DT, goal=robot_goal, horizon=HORIZON)
    ped_history = PedHistoryBuffer(history_len=HISTORY_LEN)

    # прогрев Numba
    sim.step()

    hold_time    = MPC_DT
    time_elapsed = 0.0
    u_pred       = np.array([0.0, 0.0])

    reached_goal    = False
    path_length     = 0.0
    prev_pos        = cfg['robot_start'][:2].copy()
    prev_omega      = 0.0

    smoothness_list = []
    min_dist_list   = []
    collision_steps = 0
    near_miss_steps = 0
    total_steps     = 0

    while time_elapsed < MAX_TIME:
        world = sim.current_state.world
        if world.robot is None or world.pedestrians is None:
            break
        if not np.isfinite(world.robot.pose).all():
            break

        robot_pose     = world.robot.pose.copy()
        ped_poses_dict = world.pedestrians.poses

        ped_poses_arr  = np.array([p[:2] for p in ped_poses_dict.values()])

        # перцепция: GT (perfect) или шумное range-bearing измерение
        if noisy:
            perceived = P.noisy_global_positions(robot_pose, ped_poses_dict)
        else:
            perceived = {int(k): np.array(v[:2], dtype=np.float32)
                         for k, v in ped_poses_dict.items()}

        ped_history.update(perceived)

        # ── метрики ─────────────────────────────────────────────────────────
        if len(ped_poses_arr) > 0:
            dists = np.linalg.norm(ped_poses_arr - robot_pose[:2], axis=1)
            min_d = dists.min()
            min_dist_list.append(min_d)
            if min_d < R_COLLISION:
                collision_steps += 1
            elif min_d < R_NEAR_MISS:
                near_miss_steps += 1
        total_steps += 1

        cur_pos      = robot_pose[:2]
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos     = cur_pos.copy()

        omega = u_pred[1]
        smoothness_list.append((omega - prev_omega) / MPC_DT)
        prev_omega = omega

        if np.linalg.norm(robot_pose[:2] - robot_goal[:2]) < R_GOAL:
            reached_goal = True
            break

        # ── вызов MPC с частотой MPC_DT ─────────────────────────────────────
        if hold_time >= MPC_DT:
            hist_arr, pids = ped_history.get_tensor()

            if hist_arr is not None and len(pids) > 0:
                try:
                    # детерминированный MLP rollout позиций (K, T, 2)
                    pred = mlp.predict(hist_arr, n_steps=HORIZON)

                    # метрики перцепции/прогноза по наблюдаемым пешеходам
                    est_pos = {pid: perceived[pid] for pid in pids if pid in perceived}
                    metrics.log_estimate(mpc_step, est_pos, ped_poses_dict)
                    metrics.log_prediction(mpc_step,
                        {pid: pred[i] for i, pid in enumerate(pids) if pid in perceived})

                    # текущие позиции в том же порядке, что и pred (sorted pids)
                    cur_peds = hist_arr[:, -1, :]                  # (K, 2)
                    # выбираем MAXPEDS_MPC ближайших к роботу
                    d   = np.linalg.norm(cur_peds - robot_pose[:2], axis=1)
                    idx = np.argsort(d)[:MAXPEDS_MPC]
                    controller.update_predictions(cur_peds[idx], pred[idx])
                except Exception:
                    # при ошибке MLP — TVP остаётся с предыдущего шага
                    pass

            try:
                u_pred = controller.predict(robot_pose)
            except Exception:
                u_pred = np.array([0.0, 0.0])

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
        'collision_rate': collision_steps / max(total_steps, 1),
        'near_miss_rate': near_miss_steps / max(total_steps, 1),
        'avg_speed':      path_length / max(time_elapsed, 1e-6),
        'smoothness_rms': float(np.sqrt(np.mean(np.array(smoothness_list)**2)))
                          if smoothness_list else float('nan'),
        **metrics.finalize(),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global PED_COUNT, SPEED_RANGE, TAU_RANGE, WORLD_SIZE, MAX_TIME, HORIZON
    parser = argparse.ArgumentParser(
        description='Evaluate MLP single-agent prior + MPC'
    )
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES)
    parser.add_argument('--output',   type=str, default=OUTPUT_CSV)
    parser.add_argument('--weights',  type=str, default=WEIGHTS_PATH)
    parser.add_argument('--device',   type=str, default=DEVICE)
    parser.add_argument('--seed', type=int, default=BASE_SEED,
                        help='Базовый seed (seed эпизода = base + episode_id)')
    parser.add_argument('--noisy_perception', action='store_true',
                        help='строить историю пешеходов из шумных range-bearing измерений')
    # ── distribution-параметры (in-distribution по умолчанию) ──────────────────
    parser.add_argument('--ped_count', type=int, nargs=2, default=list(PED_COUNT),
                        metavar=('MIN', 'MAX'), help='Диапазон числа пешеходов')
    parser.add_argument('--speed_range', type=float, nargs=2, default=list(SPEED_RANGE),
                        metavar=('MIN', 'MAX'), help='Диапазон скорости пешеходов')
    parser.add_argument('--tau_range', type=float, nargs=2, default=list(TAU_RANGE),
                        metavar=('MIN', 'MAX'), help='Диапазон tau HSFM')
    parser.add_argument('--world_size', type=float, default=WORLD_SIZE,
                        help='Размер арены waypoint-tracker (квадрат)')
    parser.add_argument('--max_time', type=float, default=MAX_TIME,
                        help='Макс. длительность эпизода, с')
    parser.add_argument('--horizon', type=int, default=HORIZON,
                        help='Горизонт MPC, шагов')
    args = parser.parse_args()

    # переопределяем глобальные параметры распределения из CLI
    PED_COUNT   = tuple(args.ped_count)
    SPEED_RANGE = tuple(args.speed_range)
    TAU_RANGE   = tuple(args.tau_range)
    WORLD_SIZE  = args.world_size
    MAX_TIME    = args.max_time
    HORIZON     = args.horizon

    print(f"[MLP MPC] device={args.device}  weights={args.weights}")
    print(f"[MLP MPC] episodes={args.episodes}  ped_count={PED_COUNT}  "
          f"speed={SPEED_RANGE}  tau={TAU_RANGE}  world={WORLD_SIZE}  "
          f"horizon={HORIZON}  max_time={MAX_TIME}  noisy={args.noisy_perception}")

    mlp = MLPPredictor(
        weights_path=args.weights,
        history_len=HISTORY_LEN,
        dt=MPC_DT,
        device=args.device,
    )

    print(f"[MLP MPC] Запуск {args.episodes} эпизодов...")
    records = []
    for ep in tqdm(range(args.episodes)):
        try:
            rec = run_episode(ep, mlp, seed=ep + args.seed, noisy=args.noisy_perception)
            records.append(rec)
        except Exception as e:
            print(f"  Эпизод {ep} упал: {e}")

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*55}")
    print(f"  MLP MPC (single-agent) — итоги ({len(df)} эпизодов)")
    print(f"{'='*55}")
    print(f"  Goal Reach Rate    : {df['goal_reached'].mean():.3f}")
    print(f"  Time to Goal       : {df['time_to_goal'].mean():.2f} с")
    print(f"  Path Length        : {df['path_length'].mean():.2f} м")
    print(f"  Avg Min Dist       : {df['avg_min_dist'].mean():.3f} м")
    print(f"  Collision Rate     : {df['collision_rate'].mean():.4f}")
    print(f"  Near-Miss Rate     : {df['near_miss_rate'].mean():.4f}")
    print(f"  Avg Speed          : {df['avg_speed'].mean():.3f} м/с")
    print(f"  Smoothness RMS     : {df['smoothness_rms'].mean():.4f} rad/s²")
    print(f"  Ped Est Error      : {df['ped_est_err'].mean():.4f} м")
    print(f"  ADE / FDE          : {df['ade'].mean():.4f} / {df['fde'].mean():.4f} м")
    print(f"{'='*55}")
    print(f"  Сохранено → {args.output}")


if __name__ == '__main__':
    main()

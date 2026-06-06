"""
eval_plain_mpc.py
=================
Baseline Social MPC по формулам статьи DynoSLAM (eq. eqmpcbaseline).

Функция стоимости:
  J = sum_t [ goal_weight * dist_to_goal(t)
            + sum_k PENALTY_W * max(0, R_SAFE - ||p_robot - p_k,t||)^3 ]

Предсказание пешеходов: CVM (Constant Velocity Model)
  p_k,t = p_k_current + v_k_current * t * dt

Метрики:
  goal_reach_rate   — доля эпизодов с достижением цели
  time_to_goal      — среднее время до цели (с)
  path_length       — длина пройденного пути (м)
  avg_min_dist      — средняя мин. дистанция до пешехода (м)
  collision_rate    — доля шагов, когда dist < R_COLLISION
  near_miss_rate    — доля шагов, когда R_COLLISION <= dist < R_NEAR_MISS
  avg_speed         — средняя скорость (м/с)
  smoothness_rms    — RMS угловых ускорений (rad/s^2)
"""

import sys
import os

# pyminisim лежит локально — добавляем корень проекта в путь
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import argparse
import numpy as np
import pandas as pd
import casadi
import do_mpc          # пакет называется do_mpc (с подчёркиванием)
from tqdm import tqdm

from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import (
    HeadedSocialForceModelPolicy,
    RandomWaypointTracker,
    HSFMParams,
)

import perception_utils as P

# ── гиперпараметры ────────────────────────────────────────────────────────────
SIM_DT      = 0.01    # с, физический шаг симулятора
MPC_DT      = 0.1     # с, частота вызова MPC
HORIZON     = 20      # шагов горизонта
MAX_TIME    = 30.0    # с, максимальная длительность эпизода
HISTORY_LEN = 5       # длина истории для оценки скорости (noisy режим)

GOAL_WEIGHT = 3.0     # Q — вес целевого члена
PENALTY_W   = 5000.0  # штраф за нарушение R_SAFE
R_SAFE      = 0.8     # м, радиус мягкого избегания
R_GOAL      = 0.35    # м, порог достижения цели
R_COLLISION = 0.35    # м, порог "столкновение" для метрики
R_NEAR_MISS = 0.8     # м, порог "near miss" для метрики

MAXPEDS_MPC  = 10     # максимум пешеходов в TVP

# ── distribution-параметры (по умолчанию = IN-DISTRIBUTION, как при генерации
#    обучающих данных: ped 1–15, арена 10×10, speed 0.8–1.5, tau 0.2–0.6).
#    Все переопределяются через CLI (см. main) — для OOD-прогона.
NUM_EPISODES = 100
PED_COUNT    = (1, 15)     # число пешеходов на эпизод [min, max]
SPEED_RANGE  = (0.8, 1.5)  # желаемая скорость пешеходов HSFM
TAU_RANGE    = (0.2, 0.6)  # параметр tau HSFM
WORLD_SIZE   = 10.0        # размер арены waypoint-tracker'а (как в data_generator)
BASE_SEED    = 42          # seed эпизода = BASE_SEED + episode_id

OUTPUT_CSV   = "results_plain_mpc.csv"


# ── контроллер ────────────────────────────────────────────────────────────────
class PlainMPCController:
    """
    Baseline MPC: CVM-предсказание пешеходов + фиксированный r_safe.

    Unicycle-динамика (eq. robot kinematic model f):
      px_{t+1}  = px_t  + uv * cos(pth_t) * dt
      py_{t+1}  = py_t  + uv * sin(pth_t) * dt
      pth_{t+1} = pth_t + uomg * dt

    TVP (Time-Varying Parameters): предсказанные CVM-позиции пешеходов
    на каждом шаге горизонта передаются через do-mpc TVP.
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

        # TVP: позиции пешеходов (CVM-предсказание передаётся снаружи)
        self.tvp_ped_x = [
            model.set_variable('_tvp', f'ped_x_{i}') for i in range(MAXPEDS_MPC)
        ]
        self.tvp_ped_y = [
            model.set_variable('_tvp', f'ped_y_{i}') for i in range(MAXPEDS_MPC)
        ]

        model.set_rhs('px',  px  + uv * casadi.cos(pth) * dt)
        model.set_rhs('py',  py  + uv * casadi.sin(pth) * dt)
        model.set_rhs('pth', pth + uomg * dt)

        # ── функция стоимости (eq. eqmpcbaseline stage cost) ─────────────
        # J_t = goal_weight * dist_to_goal
        #     + sum_k PENALTY_W * max(0, R_SAFE - dist_k)^3
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

        # ограничения на управление
        mpc.bounds['lower', '_u', 'uv']   = 0.0
        mpc.bounds['upper', '_u', 'uv']   = 1.8
        mpc.bounds['lower', '_u', 'uomg'] = -np.deg2rad(50.0)
        mpc.bounds['upper', '_u', 'uomg'] =  np.deg2rad(50.0)
        mpc.bounds['lower', '_x', 'pth']  = -np.pi
        mpc.bounds['upper', '_x', 'pth']  =  np.pi
        mpc.set_rterm(uv=1e-4, uomg=1e-4)

        # буфер CVM-предсказаний: shape (MAXPEDS_MPC, horizon+1, 2)
        # по умолчанию "уводим" пешеходов за горизонт.
        # ВАЖНО: инициализируем ДО set_tvp_fun/setup, т.к. mpc.setup()
        # вызывает _tvp_fun и обращается к этому буферу.
        self._ped_cvm = np.full((MAXPEDS_MPC, horizon + 1, 2), 100.0)

        self.tvp_template = mpc.get_tvp_template()
        mpc.set_tvp_fun(self._tvp_fun)
        mpc.setup()

        self.mpc   = mpc
        self.model = model

    # ── TVP callback ────────────────────────────────────────────────────────
    def _tvp_fun(self, tnow):
        for t in range(self.horizon + 1):
            for j in range(MAXPEDS_MPC):
                self.tvp_template['_tvp', t, f'ped_x_{j}'] = self._ped_cvm[j, t, 0]
                self.tvp_template['_tvp', t, f'ped_y_{j}'] = self._ped_cvm[j, t, 1]
        return self.tvp_template

    # ── обновление CVM-предсказаний ─────────────────────────────────────────
    def _update_cvm_predictions(self,
                                ped_poses: np.ndarray,
                                ped_vels: np.ndarray):
        """
        CVM (eq. Constant Velocity Model):
          p_k,t = p_k_0 + v_k * t * dt

        ped_poses : (K, 2)
        ped_vels  : (K, 2)
        """
        K = min(len(ped_poses), MAXPEDS_MPC)
        for j in range(MAXPEDS_MPC):
            for t in range(self.horizon + 1):
                if j < K:
                    self._ped_cvm[j, t] = ped_poses[j] + ped_vels[j] * t * self.dt
                else:
                    self._ped_cvm[j, t] = [100.0, 100.0]

    # ── шаг управления ──────────────────────────────────────────────────────
    def predict(self,
                x_current: np.ndarray,
                ped_poses: np.ndarray,
                ped_vels: np.ndarray) -> np.ndarray:
        """
        x_current : [px, py, theta]
        ped_poses : (K, 2) — текущие позиции пешеходов
        ped_vels  : (K, 2) — текущие скорости пешеходов
        returns   : [uv, uomg]
        """
        self._update_cvm_predictions(ped_poses, ped_vels)
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
def run_episode(episode_id: int, seed: int = None, noisy: bool = False) -> dict:
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

    controller  = PlainMPCController(dt=MPC_DT, goal=robot_goal, horizon=HORIZON)
    ped_history = P.PedHistoryBuffer(history_len=HISTORY_LEN)

    # прогрев Numba (первый шаг всегда медленнее)
    sim.step()

    hold_time    = MPC_DT   # сразу вызываем MPC на первом шаге
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

        robot_pose     = world.robot.pose.copy()           # [px, py, theta]
        ped_poses_dict = world.pedestrians.poses           # {id: [px, py, theta]}
        ped_vels_dict  = world.pedestrians.velocities      # {id: [vx, vy, omega]}

        ped_poses_arr = np.array([p[:2] for p in ped_poses_dict.values()])

        # перцепция: GT (perfect) или шумное range-bearing измерение
        if noisy:
            perceived = P.noisy_global_positions(robot_pose, ped_poses_dict)
        else:
            perceived = {int(k): np.array(v[:2], dtype=np.float32)
                         for k, v in ped_poses_dict.items()}
        ped_history.update(perceived)

        # ── метрика: дистанция ──────────────────────────────────────────────
        if len(ped_poses_arr) > 0:
            dists = np.linalg.norm(ped_poses_arr - robot_pose[:2], axis=1)
            min_d = dists.min()
            min_dist_list.append(min_d)
            if min_d < R_COLLISION:
                collision_steps += 1
            elif min_d < R_NEAR_MISS:
                near_miss_steps += 1
        total_steps += 1

        # ── метрика: длина пути ─────────────────────────────────────────────
        cur_pos      = robot_pose[:2]
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos     = cur_pos.copy()

        # ── метрика: плавность (angular jerk) ──────────────────────────────
        omega = u_pred[1]
        smoothness_list.append((omega - prev_omega) / MPC_DT)
        prev_omega = omega

        # ── проверка достижения цели ────────────────────────────────────────
        if np.linalg.norm(robot_pose[:2] - robot_goal[:2]) < R_GOAL:
            reached_goal = True
            break

        # ── вызов MPC с частотой MPC_DT ────────────────────────────────────
        if hold_time >= MPC_DT:
            hist_arr, pids = ped_history.get_tensor()

            # pid-выровненные позиции/скорости (для метрик и noisy-управления)
            if hist_arr is not None and len(pids) > 0:
                if noisy:
                    # noisy: позиция = последнее измерение, скорость = конечная
                    # разность по шумной истории (тут CVM и ломается от шума)
                    pos_pid = hist_arr[:, -1, :]
                    vel_pid = (hist_arr[:, -1, :] - hist_arr[:, -2, :]) / MPC_DT
                else:
                    # clean: GT позиции и GT скорости (поведение без изменений)
                    vel_by_pid = {int(k): np.array(v[:2], dtype=np.float32)
                                  for k, v in ped_vels_dict.items()}
                    pos_pid = np.array([perceived[pid] for pid in pids])
                    vel_pid = np.array([vel_by_pid[pid] for pid in pids])

                # метрики: оценка текущих позиций + CVM-прогноз
                est_pos = {pid: pos_pid[i] for i, pid in enumerate(pids)}
                metrics.log_estimate(mpc_step, est_pos, ped_poses_dict)
                metrics.log_prediction(mpc_step, {
                    pid: pos_pid[i][None, :] + vel_pid[i][None, :] * MPC_DT
                         * np.arange(1, HORIZON + 1)[:, None]
                    for i, pid in enumerate(pids)})

                # MPC «видит» MAXPEDS_MPC ближайших к роботу
                d   = np.linalg.norm(pos_pid - robot_pose[:2], axis=1)
                idx = np.argsort(d)[:MAXPEDS_MPC]
                near_poses, near_vels = pos_pid[idx], vel_pid[idx]
            else:
                near_poses = np.zeros((0, 2))
                near_vels  = np.zeros((0, 2))

            try:
                u_pred = controller.predict(robot_pose, near_poses, near_vels)
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
    parser = argparse.ArgumentParser(description='Evaluate Plain MPC (CVM baseline)')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                        help='Число эпизодов')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                        help='Путь к выходному CSV')
    parser.add_argument('--seed', type=int, default=BASE_SEED,
                        help='Базовый seed (seed эпизода = base + episode_id)')
    parser.add_argument('--noisy_perception', action='store_true',
                        help='позиции пешеходов = шумные измерения; '
                             'скорость CVM = конечная разность по шумной истории')
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

    print(f"[Plain MPC] episodes={args.episodes}  ped_count={PED_COUNT}  "
          f"speed={SPEED_RANGE}  tau={TAU_RANGE}  world={WORLD_SIZE}  "
          f"horizon={HORIZON}  max_time={MAX_TIME}  noisy={args.noisy_perception}")
    records = []
    for ep in tqdm(range(args.episodes)):
        try:
            rec = run_episode(ep, seed=ep + args.seed, noisy=args.noisy_perception)
            records.append(rec)
        except Exception as e:
            print(f"  Эпизод {ep} упал: {e}")

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*55}")
    print(f"  Plain MPC (CVM) — итоги ({len(df)} эпизодов)")
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
import sys
import argparse
import time
import numpy as np
import pandas as pd
import do_mpc
import casadi
import pygame
from tqdm import tqdm

sys.path.append('..')
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker, HSFMParams
from pyminisim.visual import Renderer, CircleDrawing

MAX_PEDS_MPC = 10  # Максимальное количество пешеходов, которое "видит" MPC

class DoMPCController:
    def __init__(self, dt: float, goal: np.ndarray, horizon: int = 20, goal_weight: float = 3.0):
        self.dt = dt
        self.horizon = horizon
        self.goal = goal.copy()
        self.goal_weight = goal_weight
        
        self.model = do_mpc.model.Model('discrete')
        
        pose_x = self.model.set_variable(var_type='_x', var_name='pose_x')
        pose_y = self.model.set_variable(var_type='_x', var_name='pose_y')
        pose_theta = self.model.set_variable(var_type='_x', var_name='pose_theta')
        
        u_v = self.model.set_variable(var_type='_u', var_name='u_v')
        u_omega = self.model.set_variable(var_type='_u', var_name='u_omega')
        
        # Задаем координаты пешеходов как Time-Varying Parameters (TVP)
        self.ped_x = [self.model.set_variable(var_type='_tvp', var_name=f'ped_x_{i}') for i in range(MAX_PEDS_MPC)]
        self.ped_y = [self.model.set_variable(var_type='_tvp', var_name=f'ped_y_{i}') for i in range(MAX_PEDS_MPC)]
        
        self.model.set_rhs('pose_x', pose_x + u_v * casadi.cos(pose_theta) * self.dt)
        self.model.set_rhs('pose_y', pose_y + u_v * casadi.sin(pose_theta) * self.dt)
        self.model.set_rhs('pose_theta', pose_theta + u_omega * self.dt)
        
        # Штраф за удаление от цели (увеличен вес)
        cost = self.goal_weight * (casadi.sqrt((pose_x - goal[0])**2 + (pose_y - goal[1])**2 + (pose_theta - goal[2])**2)**2)

        # dx = goal[0] - pose_x
        # dy = goal[1] - pose_y
        # angle_to_goal = casadi.atan2(dy, dx)
        # angle_error = casadi.fmod(angle_to_goal - pose_theta + casadi.pi, 2*casadi.pi) - casadi.pi
        # cost += 2.0 * angle_error**2   # вес 2.0, можно подобрать
        
        # Динамический штраф за приближение к пешеходам (margin = 0.8 метра)
        for i in range(MAX_PEDS_MPC):
            dist = casadi.sqrt((pose_x - self.ped_x[i])**2 + (pose_y - self.ped_y[i])**2)
            cost += casadi.fmax(0, 0.8 - dist)**3 * 5000.0

        

        self.model.set_expression(expr_name='cost', expr=cost)
        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {'n_robust': 0, 'n_horizon': horizon, 't_step': 0.1, 'state_discretization': 'discrete',
                     'store_full_solution': True, 'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}}
        self.mpc.set_param(**setup_mpc)
        
        self.mpc.set_objective(mterm=self.model.aux['cost'], lterm=self.model.aux['cost'])
        
        self.mpc.bounds['lower', '_x', 'pose_theta'] = -np.pi
        self.mpc.bounds['upper', '_x', 'pose_theta'] = np.pi
        self.mpc.bounds['lower', '_u', 'u_v'] = 0.0
        self.mpc.bounds['upper', '_u', 'u_v'] = 1.8
        self.mpc.bounds['lower', '_u', 'u_omega'] = -np.deg2rad(50.)
        self.mpc.bounds['upper', '_u', 'u_omega'] = np.deg2rad(50.)
        self.mpc.set_rterm(u_v=1e-4, u_omega=1e-4)
        
        # Инициализация для TVP
        self.current_ped_poses = []
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self._tvp_fun_controller)
        self.mpc.setup()

    def _tvp_fun_controller(self, t_now):
        for i in range(self.horizon + 1):
            for j in range(MAX_PEDS_MPC):
                if j < len(self.current_ped_poses):
                    self.tvp_template['_tvp', i, f'ped_x_{j}'] = self.current_ped_poses[j][0]
                    self.tvp_template['_tvp', i, f'ped_y_{j}'] = self.current_ped_poses[j][1]
                else:
                    self.tvp_template['_tvp', i, f'ped_x_{j}'] = 100.0
                    self.tvp_template['_tvp', i, f'ped_y_{j}'] = 100.0
        return self.tvp_template

    def predict(self, x_current: np.ndarray, ped_poses: np.ndarray) -> np.ndarray:
        self.current_ped_poses = ped_poses
        self.mpc.x0 = x_current
        self.mpc.set_initial_guess()
        u0 = self.mpc.make_step(x_current)
        return u0.flatten()


def generate_scenario(args):
    """Генерирует случайные параметры для одной симуляции."""
    n_peds = np.random.randint(args.ped_count[0], args.ped_count[1] + 1)
    
    # Робот
    robot_start = np.array([np.random.uniform(-4, -2), np.random.uniform(-4, 4), 0.0])
    robot_goal = np.array([np.random.uniform(2, 4), np.random.uniform(-4, 4), 0.0])
    
    # Пешеходы
    ped_starts = np.random.uniform(-3, 3, size=(n_peds, 3))
    ped_starts[:, 2] = np.random.uniform(-np.pi, np.pi, size=n_peds) # Угол
    ped_speeds = np.random.uniform(args.speed_range[0], args.speed_range[1], size=n_peds)
    
    # Параметры HSFM. tau - время релаксации (обратно пропорционально силе притяжения к цели)
    tau = np.random.uniform(args.tau_range[0], args.tau_range[1])
    params = HSFMParams.create_default()
    params.tau = tau
    
    return {
        "n_peds": n_peds, "robot_start": robot_start, "robot_goal": robot_goal,
        "ped_starts": ped_starts, "ped_speeds": ped_speeds, "hsfm_params": params
    }


def run_simulation(episode_id, config, args, visual=False):
    """Запускает симуляцию и собирает данные."""
    sim_dt = 0.01
    mpc_dt = 0.1
    
    robot_model = UnicycleRobotModel(initial_pose=config["robot_start"])
    tracker = RandomWaypointTracker(world_size=(10.0, 10.0))
    pedestrians_model = HeadedSocialForceModelPolicy(
        n_pedestrians=config["n_peds"],
        waypoint_tracker=tracker,
        pedestrian_linear_velocity_magnitude=config["ped_speeds"],
        initial_poses=config["ped_starts"],
        hsfm_params=config["hsfm_params"]
    )

    sim = Simulation(sim_dt=sim_dt, world_map=EmptyWorld(), robot_model=robot_model,
                     pedestrians_model=pedestrians_model, sensors=[])
    
    controller = DoMPCController(dt=mpc_dt, goal=config["robot_goal"], horizon=40)
    
    renderer = None
    if visual:
        renderer = Renderer(simulation=sim, resolution=40.0, screen_size=(600, 600), camera="fixed")
        renderer.initialize()
        renderer.draw("goal", CircleDrawing(controller.goal[:2], 0.1, (255, 0, 0), 0))
    
    records = []
    sim.step()
    hold_time = sim_dt
    u_pred = np.array([0., 0.])
    
    time_elapsed = 0.0
    running = True
    episode_valid = True  # флаг, что в эпизоде нет NaN
    
    while running and time_elapsed < args.max_time:
        # Обработка событий в визуальном режиме
        if visual:
            renderer.render()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
        
        # Проверка на NaN/Inf в состоянии
        if sim.current_state.world.pedestrians is None:
            episode_valid = False
            break
        
        robot_pose = sim.current_state.world.robot.pose
        robot_vel = sim.current_state.world.robot.velocity
        ped_poses = np.stack([p for p in sim.current_state.world.pedestrians.poses.values()])
        ped_vels = np.stack([v for v in sim.current_state.world.pedestrians.velocities.values()])

        if (not np.isfinite(robot_pose).all() or
            not np.isfinite(robot_vel).all() or
            not np.isfinite(ped_poses).all() or
            not np.isfinite(ped_vels).all()):
            episode_valid = False
            break
        
        # Проверка достижения цели (до записи и шага)
        dist_to_goal = np.linalg.norm(robot_pose[:2] - config["robot_goal"][:2])
        if dist_to_goal < 0.3:
            # Достигли цели, выходим из цикла без дополнительного шага
            break
        
        # Запись состояния (раз в mpc_dt)
        if hold_time >= mpc_dt:
            # Предсказание нового контроля
            u_pred = controller.predict(robot_pose, ped_poses)
            hold_time = 0.
            
            # Запись в датасет
            # records.append({
            #     "episode": episode_id, "time": time_elapsed, "agent_type": "robot", "agent_id": 0,
            #     "x": robot_pose[0], "y": robot_pose[1], "theta": robot_pose[2],
            #     "vx": robot_vel[0], "vy": robot_vel[1], "goal_x": config["robot_goal"][0], "goal_y": config["robot_goal"][1]
            # })
            records.append({
                "episode": episode_id, "time": time_elapsed, "agent_type": "robot", "agent_id": 0,
                "x": robot_pose[0], "y": robot_pose[1], "theta": robot_pose[2],
                "vx": robot_vel[0], "vy": robot_vel[1],
                "u_v": u_pred[0],          # <-- добавлено
                "u_omega": u_pred[1],      # <-- добавлено
                "goal_x": config["robot_goal"][0], "goal_y": config["robot_goal"][1]
            })
            for p_id in range(config["n_peds"]):
                records.append({
                    "episode": episode_id, "time": time_elapsed, "agent_type": "pedestrian", "agent_id": p_id+1,
                    "x": ped_poses[p_id][0], "y": ped_poses[p_id][1], "theta": ped_poses[p_id][2],
                    "vx": ped_vels[p_id][0], "vy": ped_vels[p_id][1], "goal_x": np.nan, "goal_y": np.nan,
                    "u_v": np.nan,
                    "u_omega": np.nan
                })
        
        # Шаг симуляции с применением управления
        sim.step(u_pred)
        hold_time += sim_dt
        time_elapsed += sim_dt

    if visual:
        renderer.close()
        
    return records if episode_valid else []


def main():
    parser = argparse.ArgumentParser(description="Simulation data generator")
    parser.add_argument('--mode', choices=['visual', 'dataset'], default='visual', 
                        help='visual: одно окно с отрисовкой; dataset: генерация датасета')
    parser.add_argument('--num_episodes', type=int, default=5, 
                        help='Количество эпизодов (траекторий) для режима dataset')
    parser.add_argument('--ped_count', type=int, nargs=2, default=[3, 7], 
                        help='Диапазон количества пешеходов [min, max]')
    parser.add_argument('--speed_range', type=float, nargs=2, default=[0.8, 1.5], 
                        help='Диапазон желаемой скорости пешеходов [min, max]')
    parser.add_argument('--tau_range', type=float, nargs=2, default=[0.2, 0.6], 
                        help='Диапазон параметра tau HSFM (чем меньше, тем агрессивнее притяжение к цели)')
    parser.add_argument('--max_time', type=float, default=25.0, 
                        help='Максимальное время симуляции одного эпизода в секундах')
    parser.add_argument('--output', type=str, default='trajectories_dataset.csv', 
                        help='Имя выходного файла CSV')
    
    args = parser.parse_args()

    if args.mode == 'visual':
        print("Запуск в визуальном режиме (1 эпизод)...")
        config = generate_scenario(args)
        run_simulation(0, config, args, visual=True)
    
    elif args.mode == 'dataset':
        # print(f"Запуск генерации датасета: {args.num_episodes} эпизодов...")
        # all_data = []
        # for ep in tqdm(range(args.num_episodes)):
        #     # print(f"Генерация эпизода {ep + 1}/{args.num_episodes}...")
        #     config = generate_scenario(args)
        #     records = run_simulation(ep, config, args, visual=False)
        #     if len(records) == 0:
        #         print(f"Эпизод {ep + 1} отброшен из-за NaN/Inf в HSFM")
        #         continue
        #     all_data.extend(records)
            
        #     df = pd.DataFrame(all_data)
        #     df.to_csv(args.output, index=False)
        # print(f"Готово! Сохранено {len(df)} записей в файл '{args.output}'")

        print(f"Запуск генерации датасета: {args.num_episodes} эпизодов...")
        import os
        from tqdm import tqdm

        batch_size = 100  # количество строк для одной пакетной записи
        file_exists = os.path.isfile(args.output)
        first_batch = not file_exists  # первый раз пишем заголовок
        all_data = []  # накопитель строк для текущей порции

        for ep in tqdm(range(args.num_episodes)):
            config = generate_scenario(args)
            records = run_simulation(ep, config, args, visual=False)
            if len(records) == 0:
                print(f"Эпизод {ep + 1} отброшен из-за NaN/Inf в HSFM")
                continue
            
            all_data.extend(records)
            
            # Если набралось batch_size записей или это последний эпизод – записываем
            if len(all_data) >= batch_size or ep == args.num_episodes - 1:
                df_batch = pd.DataFrame(all_data)
                # Записываем: первый раз с заголовком, последующие без
                df_batch.to_csv(args.output, index=False, mode='a', header=first_batch)
                first_batch = False
                all_data = []  # очищаем накопитель

        # Итоговая статистика
        if os.path.isfile(args.output):
            df_total = pd.read_csv(args.output)
            print(f"Готово! Сохранено {len(df_total)} записей в файл '{args.output}'")
        else:
            print("Нет сохранённых эпизодов.")


if __name__ == '__main__':
    main()

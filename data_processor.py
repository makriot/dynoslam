import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, csv_file, window_size=10, dt=0.11, max_range=10.0,
                 noise_odom_v=0.05, noise_odom_w=0.05,
                 noise_obs_r=0.05, noise_obs_phi=0.02):
        """
        parameters:
        ---
        csv_file: dataset
        window_size:
        dt:
        max_range: maximum range from robot to landmarks 
        """
        self.csv_file = csv_file
        self.window_size = window_size
        self.dt = dt
        self.max_range = max_range
        
        # Параметры шума
        self.noise_odom_v = noise_odom_v
        self.noise_odom_w = noise_odom_w
        self.noise_obs_r = noise_obs_r
        self.noise_obs_phi = noise_obs_phi
        
        print("Loading data...")
        self.df = pd.read_csv(csv_file)

    def get_episode_windows(self, episode_id):
        ep_df = self.df[self.df['episode'] == episode_id].copy()
        times = np.sort(ep_df['time'].unique())
        
        time_data = []
        for t in times:
            step_df = ep_df[ep_df['time'] == t]
            
            robot_row = step_df[step_df['agent_type'] == 'robot'].iloc[0]
            ped_df = step_df[step_df['agent_type'] == 'pedestrian']
            
            robot_state = np.array([robot_row['x'], robot_row['y'], robot_row['theta']])
            u_v = robot_row['u_v'] if not np.isnan(robot_row['u_v']) else 0.0
            u_omega = robot_row['u_omega'] if not np.isnan(robot_row['u_omega']) else 0.0
            
            true_lms = {int(row['agent_id']): np.array([row['x'], row['y']]) for _, row in ped_df.iterrows()}
            
            time_data.append({
                'time': t, 'robot_state': robot_state,
                'u_v_true': u_v, 'u_omega_true': u_omega, 'true_lms': true_lms
            })

        W = self.window_size
        windows = []
        
        # Формируем окна (размер W требует W+1 состояний: от t=0 до t=W)
        for i in range(len(time_data) - W):
            window_steps = time_data[i : i + W + 1]
            init_robot_pose = window_steps[0]['robot_state']
            odometry, observations = [], []
            
            for j in range(1, W + 1):
                prev_step = window_steps[j-1]
                curr_step = window_steps[j]
                
                # Добавляем шум к управлению (Одометрия)
                u_v_noisy = prev_step['u_v_true'] + np.random.normal(0, self.noise_odom_v)
                u_w_noisy = prev_step['u_omega_true'] + np.random.normal(0, self.noise_odom_w)
                odometry.append([u_v_noisy, u_w_noisy])
                
                # Генерируем локальные наблюдения (Z) из глобальных координат + шум
                rx, ry, rth = curr_step['robot_state']
                step_obs = []
                for lm_id, lm_pos in curr_step['true_lms'].items():
                    dx, dy = lm_pos[0] - rx, lm_pos[1] - ry
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r <= self.max_range:  # Робот видит пешехода
                        phi = (np.arctan2(dy, dx) - rth + np.pi) % (2*np.pi) - np.pi
                        step_obs.append({
                            'lm_id': lm_id,
                            'range': r + np.random.normal(0, self.noise_obs_r),
                            'bearing': phi + np.random.normal(0, self.noise_obs_phi)
                        })
                observations.append(step_obs)
                
            windows.append({
                'init_robot_pose': init_robot_pose,
                'odometry': np.array(odometry),
                'observations': observations,
                'true_trajectory': [step['robot_state'] for step in window_steps],
                'true_landmarks': [step['true_lms'] for step in window_steps]
            })
        return windows

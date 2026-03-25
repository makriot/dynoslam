import torch
import torch.nn as nn
from .base_dynamic_slam import BaseDynamicSLAM

def wrap_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

class CVDynamicSLAM(BaseDynamicSLAM):
    def __init__(self, window_size: int, dt: float = 0.1, 
                 sigma_odom_v=0.1, sigma_odom_w=0.1, 
                 sigma_obs_r=0.1, sigma_obs_phi=0.05,
                 sigma_acc=0.5):  # Оставили только штраф за ускорение (рывки)
        super().__init__(window_size, dt)
        self.sigma_odom_v = sigma_odom_v
        self.sigma_odom_w = sigma_odom_w
        self.sigma_obs_r = sigma_obs_r
        self.sigma_obs_phi = sigma_obs_phi
        self.sigma_acc = sigma_acc

    def forward(self, init_robot_pose, odometry, observations, num_epochs=50, prediction_horizon=10):
        W = self.window_size
        
        lm_ids = set()
        for obs_step in observations:
            for obs in obs_step:
                lm_ids.add(obs['lm_id'])
        lm_ids = sorted(list(lm_ids))
        M = len(lm_ids)
        lm_id_to_idx = {lmid: idx for idx, lmid in enumerate(lm_ids)}

        # ИСКЛЮЧИЛИ velocities из параметров
        robot_poses = nn.Parameter(torch.zeros(W + 1, 3))
        landmarks = nn.Parameter(torch.zeros(W + 1, M, 2))

        with torch.no_grad():
            robot_poses[0] = init_robot_pose
            for i in range(W):
                v, w = odometry[i, 0], odometry[i, 1]
                th = robot_poses[i, 2]
                robot_poses[i+1, 0] = robot_poses[i, 0] + v * torch.cos(th) * self.dt
                robot_poses[i+1, 1] = robot_poses[i, 1] + v * torch.sin(th) * self.dt
                robot_poses[i+1, 2] = wrap_angle(th + w * self.dt)
            
            initialized_lms = set()
            for i in range(W):
                rx, ry, rth = robot_poses[i+1]
                for obs in observations[i]:
                    idx = lm_id_to_idx[obs['lm_id']]
                    if idx not in initialized_lms:
                        r, phi = obs['range'], obs['bearing']
                        lx = rx + r * torch.cos(rth + phi)
                        ly = ry + r * torch.sin(rth + phi)
                        # Задаем константой на всё окно (m_0 = m_1 = ... = m_W)
                        # Это означает, что начальное ускорение равно 0, что идеально для prior'а
                        landmarks[:, idx, 0] = lx  
                        landmarks[:, idx, 1] = ly
                        initialized_lms.add(idx)

        optimizer = torch.optim.Adam([robot_poses, landmarks], lr=0.05)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0
            
            # 1. Привязка к начальной позе
            loss += torch.sum((robot_poses[0] - init_robot_pose)**2) * 1e6

            # 2. Odometry Factor
            for i in range(W):
                v, w = odometry[i, 0], odometry[i, 1]
                th_prev = robot_poses[i, 2]
                pred_x = robot_poses[i, 0] + v * torch.cos(th_prev) * self.dt
                pred_y = robot_poses[i, 1] + v * torch.sin(th_prev) * self.dt
                pred_th = robot_poses[i, 2] + w * self.dt
                
                loss += ((robot_poses[i+1, 0] - pred_x) / self.sigma_odom_v)**2
                loss += ((robot_poses[i+1, 1] - pred_y) / self.sigma_odom_v)**2
                loss += (wrap_angle(robot_poses[i+1, 2] - pred_th) / self.sigma_odom_w)**2

            # 3. Observation Factor
            for i in range(W):
                rx, ry, rth = robot_poses[i+1]
                for obs in observations[i]:
                    idx = lm_id_to_idx[obs['lm_id']]
                    r_meas, phi_meas = obs['range'], obs['bearing']
                    
                    lx, ly = landmarks[i+1, idx]
                    dx, dy = lx - rx, ly - ry
                    
                    r_pred = torch.sqrt(dx**2 + dy**2)
                    phi_pred = wrap_angle(torch.atan2(dy, dx) - rth)
                    
                    loss += ((r_pred - r_meas) / self.sigma_obs_r)**2
                    loss += (wrap_angle(phi_pred - phi_meas) / self.sigma_obs_phi)**2

            # 4. Kinematic Factor (Zero Acceleration / Smoothness Prior)
            # Векторизованный лосс: m_{i+1} - 2m_{i} + m_{i-1} -> 0
            if W >= 2:
                # landmarks имеет размер [W+1, M, 2]
                # landmarks[2:] — это элементы со 2 по W (т.е. i+1)
                # landmarks[1:-1] — элементы с 1 по W-1 (т.е. i)
                # landmarks[:-2] — элементы с 0 по W-2 (т.е. i-1)
                accel = landmarks[2:] - 2 * landmarks[1:-1] + landmarks[:-2]
                loss += torch.sum((accel / self.sigma_acc)**2)

            loss.backward()
            optimizer.step()

        # Вычисляем финальные скорости из двух последних кадров окна
        # для генерации предиктов в MPC
        predictions = {}
        with torch.no_grad():
            if W >= 1:
                # v = (m_W - m_{W-1}) / dt
                last_velocities = (landmarks[-1] - landmarks[-2]) / self.dt
            else:
                last_velocities = torch.zeros((M, 2))

            for lmid, idx in lm_id_to_idx.items():
                m_last = landmarks[-1, idx]
                v_last = last_velocities[idx]
                
                pred_traj = []
                for t in range(1, prediction_horizon + 1):
                    pred_traj.append(m_last + v_last * self.dt * t)
                predictions[lmid] = torch.stack(pred_traj)

        return robot_poses.detach(), landmarks.detach(), predictions

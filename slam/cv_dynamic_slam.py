import torch
import torch.nn as nn
from .base_dynamic_slam import BaseDynamicSLAM

def wrap_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

class CVDynamicSLAM(BaseDynamicSLAM):
    def __init__(self, window_size: int, dt: float = 0.1, 
                 sigma_odom_v=0.1, sigma_odom_w=0.1, 
                 sigma_obs_r=0.1, sigma_obs_phi=0.05,
                 sigma_acc=0.5):
        super().__init__(window_size, dt)
        self.sigma_odom_v = sigma_odom_v
        self.sigma_odom_w = sigma_odom_w
        self.sigma_obs_r = sigma_obs_r
        self.sigma_obs_phi = sigma_obs_phi
        self.sigma_acc = sigma_acc

    def forward(self, init_robot_pose, odometry, observations, lm_history=None, num_epochs=100, prediction_horizon=10):
        W = self.window_size

        device = "cpu"
        
        lm_ids = set()
        for obs_step in observations:
            for obs in obs_step:
                lm_ids.add(obs['lm_id'])
        lm_ids = sorted(list(lm_ids))
        M = len(lm_ids)
        lm_id_to_idx = {lmid: idx for idx, lmid in enumerate(lm_ids)}

        robot_poses = nn.Parameter(torch.zeros(W + 1, 3, device=device))
        landmarks = nn.Parameter(torch.zeros(W + 1, M, 2, device=device))

        # =================================================================
        # ПРЕДОБРАБОТКА (Вне цикла оптимизации!)
        # Собираем разреженные наблюдения в плотные тензоры с маской
        # =================================================================
        meas_r = torch.zeros(W, M)
        meas_phi = torch.zeros(W, M)
        obs_mask = torch.zeros(W, M, dtype=torch.bool)
        
        for i in range(W):
            for obs in observations[i]:
                idx = lm_id_to_idx[obs['lm_id']]
                meas_r[i, idx] = obs['range']
                meas_phi[i, idx] = obs['bearing']
                obs_mask[i, idx] = True

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
                        lx = rx + obs['range'] * torch.cos(rth + obs['bearing'])
                        ly = ry + obs['range'] * torch.sin(rth + obs['bearing'])
                        landmarks[:, idx, 0] = lx  
                        landmarks[:, idx, 1] = ly
                        initialized_lms.add(idx)

        optimizer = torch.optim.Adam([robot_poses, landmarks], lr=0.05)

        # Подготовим тензоры одометрии для быстрого доступа
        odom_v = odometry[:, 0]
        odom_w = odometry[:, 1]

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0
            
            # 1. Привязка к начальной позе
            loss += torch.sum((robot_poses[0] - init_robot_pose)**2) * 1e6

            # 2. Odometry Factor (ПОЛНОСТЬЮ ВЕКТОРИЗОВАНО)
            th_prev = robot_poses[:-1, 2]
            pred_x = robot_poses[:-1, 0] + odom_v * torch.cos(th_prev) * self.dt
            pred_y = robot_poses[:-1, 1] + odom_v * torch.sin(th_prev) * self.dt
            pred_th = robot_poses[:-1, 2] + odom_w * self.dt
            
            loss += torch.sum(((robot_poses[1:, 0] - pred_x) / self.sigma_odom_v)**2)
            loss += torch.sum(((robot_poses[1:, 1] - pred_y) / self.sigma_odom_v)**2)
            loss += torch.sum((wrap_angle(robot_poses[1:, 2] - pred_th) / self.sigma_odom_w)**2)

            # 3. Observation Factor (ПОЛНОСТЬЮ ВЕКТОРИЗОВАНО)
            if M > 0:
                rx = robot_poses[1:, 0].unsqueeze(1) # [W, 1]
                ry = robot_poses[1:, 1].unsqueeze(1) # [W, 1]
                rth = robot_poses[1:, 2].unsqueeze(1) # [W, 1]
                
                lx = landmarks[1:, :, 0] # [W, M]
                ly = landmarks[1:, :, 1] # [W, M]
                
                dx = lx - rx
                dy = ly - ry
                
                r_pred = torch.sqrt(dx**2 + dy**2)
                phi_pred = wrap_angle(torch.atan2(dy, dx) - rth)
                
                # Считаем лосс только там, где obs_mask == True
                loss += torch.sum((((r_pred - meas_r) / self.sigma_obs_r)**2)[obs_mask])
                loss += torch.sum(((wrap_angle(phi_pred - meas_phi) / self.sigma_obs_phi)**2)[obs_mask])

            # 4. Kinematic Factor (Уже было векторизовано)
            # if W >= 2 and M > 0:
            #     accel = landmarks[2:] - 2 * landmarks[1:-1] + landmarks[:-2]
            #     loss += torch.sum((accel / self.sigma_acc)**2)

            loss.backward()
            optimizer.step()

        predictions = {}
        with torch.no_grad():
            if W >= 1 and M > 0:
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

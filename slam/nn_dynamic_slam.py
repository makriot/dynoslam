import torch
import torch.nn as nn
from .base_dynamic_slam import BaseDynamicSLAM

def wrap_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

class NeuralDynamicSLAM(BaseDynamicSLAM):
    def __init__(self, predictor_model, window_size: int, dt: float = 0.1, 
                 sigma_odom_v=0.1, sigma_odom_w=0.1, 
                 sigma_obs_r=0.1, sigma_obs_phi=0.05,
                 sigma_kin=0.2): # Штраф за отклонение от предсказания нейронки
        super().__init__(window_size, dt)
        
        self.predictor = predictor_model
        self.predictor.eval() # Замораживаем веса! Мы не учим сеть внутри SLAM
        for param in self.predictor.parameters():
            param.requires_grad = False
            
        self.sigma_odom_v = sigma_odom_v
        self.sigma_odom_w = sigma_odom_w
        self.sigma_obs_r = sigma_obs_r
        self.sigma_obs_phi = sigma_obs_phi
        self.sigma_kin = sigma_kin

    def forward(self, init_robot_pose, odometry, observations, lm_history=None, num_epochs=100, prediction_horizon=10):
        """
        :param lm_history: dict {lm_id: tensor[history_len, 2]} - история координат перед окном
        """
        W = self.window_size
        
        # 1. Извлекаем ID всех наблюдаемых landmarks
        lm_ids = set()
        for obs_step in observations:
            for obs in obs_step:
                lm_ids.add(obs['lm_id'])
        lm_ids = sorted(list(lm_ids))
        M = len(lm_ids)
        lm_id_to_idx = {lmid: idx for idx, lmid in enumerate(lm_ids)}

        robot_poses = nn.Parameter(torch.zeros(W + 1, 3))
        landmarks = nn.Parameter(torch.zeros(W + 1, M, 2))

        # ---------------------------------------------------------
        # ШАГ 1: ПРЕДСКАЗАНИЕ НЕЙРОНКИ (Prior для графа)
        # ---------------------------------------------------------
        # Готовим батч истории для нейронки [M, history_len, 2]
        H_len = self.predictor.history_len
        hist_batch = torch.zeros((M, H_len, 2))
        
        if lm_history is not None:
            for lmid, idx in lm_id_to_idx.items():
                if lmid in lm_history:
                    hist_batch[idx] = lm_history[lmid]
                else:
                    # Если пешеход только появился, история = 0 (будет стоять на месте)
                    hist_batch[idx] = 0.0 

        # Получаем предсказанные скорости на всё окно W: shape [M, W, 2]
        with torch.no_grad():
            predicted_velocities = self.predictor(hist_batch, num_steps=W)

        # ---------------------------------------------------------
        # ШАГ 2: ИНИЦИАЛИЗАЦИЯ И ПРЕДОБРАБОТКА МАТРИЦ 
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # ШАГ 3: ОПТИМИЗАЦИЯ ФАКТОРНОГО ГРАФА
        # ---------------------------------------------------------
        optimizer = torch.optim.Adam([robot_poses, landmarks], lr=0.05)
        odom_v = odometry[:, 0]
        odom_w = odometry[:, 1]

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0
            
            # Prior
            loss += torch.sum((robot_poses[0] - init_robot_pose)**2) * 1e6

            # Odometry Factor
            th_prev = robot_poses[:-1, 2]
            pred_x = robot_poses[:-1, 0] + odom_v * torch.cos(th_prev) * self.dt
            pred_y = robot_poses[:-1, 1] + odom_v * torch.sin(th_prev) * self.dt
            pred_th = robot_poses[:-1, 2] + odom_w * self.dt
            
            loss += torch.sum(((robot_poses[1:, 0] - pred_x) / self.sigma_odom_v)**2)
            loss += torch.sum(((robot_poses[1:, 1] - pred_y) / self.sigma_odom_v)**2)
            loss += torch.sum((wrap_angle(robot_poses[1:, 2] - pred_th) / self.sigma_odom_w)**2)

            # Observation Factor
            if M > 0:
                rx = robot_poses[1:, 0].unsqueeze(1)
                ry = robot_poses[1:, 1].unsqueeze(1)
                rth = robot_poses[1:, 2].unsqueeze(1)
                
                lx = landmarks[1:, :, 0]
                ly = landmarks[1:, :, 1]
                
                r_pred = torch.sqrt((lx - rx)**2 + (ly - ry)**2)
                phi_pred = wrap_angle(torch.atan2(ly - ry, lx - rx) - rth)
                
                loss += torch.sum((((r_pred - meas_r) / self.sigma_obs_r)**2)[obs_mask])
                loss += torch.sum(((wrap_angle(phi_pred - meas_phi) / self.sigma_obs_phi)**2)[obs_mask])

            # Neural Kinematic Factor (Уравнение 9 из статьи: Variant 1)
            if W >= 1 and M > 0:
                # landmarks[1:] - landmarks[:-1] это реальные смещения в окне, shape [W, M, 2]
                # predicted_velocities имеет форму [M, W, 2], транспонируем в [W, M, 2]
                v_pred_W_M_2 = predicted_velocities.transpose(0, 1)
                
                # Считаем разницу между ожидаемым сдвигом нейронки и оптимизируемым сдвигом
                kin_err = (landmarks[1:] - landmarks[:-1]) - (v_pred_W_M_2 * self.dt)
                loss += torch.sum((kin_err / self.sigma_kin)**2)

            loss.backward()
            optimizer.step()

        # ---------------------------------------------------------
        # ШАГ 4: ПРЕДСКАЗАНИЕ ДЛЯ MPC (На основе оптимизированных данных)
        # ---------------------------------------------------------
        predictions = {}
        if M > 0:
            with torch.no_grad():
                # Берем последние H_len кадров из только что оптимизированных координат
                # Если W < H_len, нужно склеить с прошлой историей (для простоты пока берем из landmarks)
                available_len = min(W + 1, H_len)
                pad_len = H_len - available_len
                
                opt_hist = torch.zeros((M, H_len, 2))
                if pad_len > 0:
                    opt_hist[:, :pad_len, :] = hist_batch[:, -pad_len:, :]
                opt_hist[:, pad_len:, :] = landmarks[-available_len:].transpose(0, 1)

                # Прогоняем нейронку еще раз для предсказания будущего!
                mpc_velocities = self.predictor(opt_hist, num_steps=prediction_horizon) # [M, horizon, 2]

                for lmid, idx in lm_id_to_idx.items():
                    m_curr = landmarks[-1, idx]
                    v_traj = mpc_velocities[idx] # [horizon, 2]
                    
                    pred_traj = []
                    for t in range(prediction_horizon):
                        m_curr = m_curr + v_traj[t] * self.dt
                        pred_traj.append(m_curr)
                        
                    predictions[lmid] = torch.stack(pred_traj)

        return robot_poses.detach(), landmarks.detach(), predictions
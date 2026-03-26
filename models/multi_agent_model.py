import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseVelocityPredictor

class SimpleGATPredictor(BaseVelocityPredictor):
    def __init__(self, history_len: int, dt: float, hidden_dim=64, interaction_radius=5.0):
        super().__init__(history_len, dt)
        self.interaction_radius = interaction_radius
        self.hidden_dim = hidden_dim
        
        input_dim = (history_len - 1) * 2
        
        # RNN/MLP для сжатия истории
        self.hist_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GAT (Graph Attention Network) слои
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.a = nn.Linear(2 * hidden_dim, 1, bias=False)
        
        # Выходной слой для простой регрессии скорости
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def compute_gat_context(self, h, positions):
        """Строит граф на лету по радиусу и считает Attention"""
        M = h.shape[0]
        if M <= 1:
            return h # Если агент один, контекст равен его же истории
            
        # 1. Строим граф по расстоянию (Матрица смежности)
        dist = torch.cdist(positions, positions) # [M, M]
        adj = (dist <= self.interaction_radius).float()
        
        # 2. Считаем Attention
        Wh = self.W(h) # [M, hidden_dim]
        Wh_i = Wh.unsqueeze(1).expand(M, M, self.hidden_dim)
        Wh_j = Wh.unsqueeze(0).expand(M, M, self.hidden_dim)
        
        # a^T [W h_i || W h_j]
        e = self.a(torch.cat([Wh_i, Wh_j], dim=2)).squeeze(-1) # [M, M]
        e = F.leaky_relu(e, 0.2)
        
        # 3. Маскируем связи, которых нет в радиусе R
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1) # [M, M]
        
        # 4. Агрегируем контекст: c_k = sum(alpha_{k,j} * W h_j)
        c = torch.matmul(attention, Wh) # [M, hidden_dim]
        return c

    def forward(self, history_positions, num_steps):
        M, H, _ = history_positions.shape
        device = history_positions.device
        
        curr_hist = history_positions.clone()
        predictions = []
        
        # Авторегрессионный цикл (для инференса)
        for _ in range(num_steps):
            vel = (curr_hist[:, 1:, :] - curr_hist[:, :-1, :]) / self.dt
            v_flat = vel.reshape(M, -1)
            
            h = self.hist_encoder(v_flat)
            last_pos = curr_hist[:, -1, :]
            
            c = self.compute_gat_context(h, last_pos)
            next_v = self.out_layer(c) # [M, 2]
            predictions.append(next_v)
            
            # Обновляем координаты для следующего шага
            next_pos = last_pos + next_v * self.dt
            curr_hist = torch.cat([curr_hist[:, 1:, :], next_pos.unsqueeze(1)], dim=1)
            
        return torch.stack(predictions, dim=1)


class FlowMatchingGATPredictor(SimpleGATPredictor):
    def __init__(self, history_len: int, dt: float, hidden_dim=64, interaction_radius=5.0):
        super().__init__(history_len, dt, hidden_dim, interaction_radius)
        
        # Заменяем out_layer на Нейросеть Векторного Поля v_theta(v_t, t, c)
        self.out_layer = None 
        self.v_theta = nn.Sequential(
            nn.Linear(2 + 1 + hidden_dim, hidden_dim), # v_t(2) + t(1) + c_k(hidden)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def get_vector_field(self, v_t, t, c):
        if isinstance(t, float):
            t_vec = torch.full((v_t.shape[0], 1), t, device=v_t.device)
        else:
            t_vec = t # Уже тензор [M, 1]
        x = torch.cat([v_t, t_vec, c], dim=-1)
        return self.v_theta(x)

    def forward(self, history_positions, num_steps):
        """Инференс через ODE решатель (Euler)"""
        M, H, _ = history_positions.shape
        device = history_positions.device
        
        curr_hist = history_positions.clone()
        predictions = []
        
        for _ in range(num_steps):
            vel = (curr_hist[:, 1:, :] - curr_hist[:, :-1, :]) / self.dt
            v_flat = vel.reshape(M, -1)
            h = self.hist_encoder(v_flat)
            last_pos = curr_hist[:, -1, :]
            c = self.compute_gat_context(h, last_pos)
            
            # Сэмплирование (ODE Euler step) из t=0 в t=1
            v_t = torch.randn((M, 2), device=device) # Изначальный шум N(0, I)
            steps = 10
            dt_ode = 1.0 / steps
            
            for step in range(steps):
                t = step * dt_ode
                v_vec = self.get_vector_field(v_t, t, c)
                v_t = v_t + v_vec * dt_ode # Шаг Эйлера
                
            next_v = v_t
            predictions.append(next_v)
            next_pos = last_pos + next_v * self.dt
            curr_hist = torch.cat([curr_hist[:, 1:, :], next_pos.unsqueeze(1)], dim=1)
            
        return torch.stack(predictions, dim=1)

    def compute_fm_loss(self, history_positions, target_next_v):
        """Уравнение 8 из статьи: Обучение Flow Matching"""
        M = history_positions.shape[0]
        device = history_positions.device
        
        vel = (history_positions[:, 1:, :] - history_positions[:, :-1, :]) / self.dt
        v_flat = vel.reshape(M, -1)
        h = self.hist_encoder(v_flat)
        last_pos = history_positions[:, -1, :]
        c = self.compute_gat_context(h, last_pos)
        
        # Генерируем случайное время t и шум v_0
        t = torch.rand((M, 1), device=device)
        v_0 = torch.randn((M, 2), device=device)
        v_1 = target_next_v # Целевая скорость
        
        # Линейная интерполяция
        v_t = t * v_1 + (1 - t) * v_0
        
        # Предсказываем векторное поле
        pred_vec = self.get_vector_field(v_t, t, c)
        
        # Таргет для векторного поля = v_1 - v_0
        target_vec = v_1 - v_0
        
        return F.mse_loss(pred_vec, target_vec)
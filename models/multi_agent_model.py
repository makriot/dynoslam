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
        
        self.hist_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.a = nn.Linear(2 * hidden_dim, 1, bias=False)
        
        # ВАЖНО: Теперь вход в out_layer равен hidden_dim * 2 
        # (потому что мы будем конкатенировать h и c)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def compute_gat_context(self, h, positions):
        # ... (ВЕСЬ КОД ЭТОЙ ФУНКЦИИ ОСТАЕТСЯ БЕЗ ИЗМЕНЕНИЙ, КАК В ПРОШЛЫЙ РАЗ) ...
        # (включая проверку на dim() == 3 и device=positions.device)
        device = positions.device
        
        if positions.dim() == 3:  
            B, M, _ = positions.shape
            dist = torch.cdist(positions, positions) 
            adj = (dist <= self.interaction_radius).float().to(device)
            
            Wh = self.W(h)
            Wh = Wh.view(B, M, self.hidden_dim)
            
            Wh_i = Wh.unsqueeze(2).expand(B, M, M, self.hidden_dim)
            Wh_j = Wh.unsqueeze(1).expand(B, M, M, self.hidden_dim)
            
            e = self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)
            e = F.leaky_relu(e, 0.2)
            
            zero_vec = torch.full_like(e, -9e15, device=device) 
            
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=-1)
            
            c = torch.bmm(attention, Wh) 
            return c.view(B * M, self.hidden_dim)
        else:
            M = h.shape[0]
            if M <= 1:
                return h
                
            dist = torch.cdist(positions, positions)
            adj = (dist <= self.interaction_radius).float().to(device)
            
            Wh = self.W(h)
            Wh_i = Wh.unsqueeze(1).expand(M, M, self.hidden_dim)
            Wh_j = Wh.unsqueeze(0).expand(M, M, self.hidden_dim)
            
            e = self.a(torch.cat([Wh_i, Wh_j], dim=2)).squeeze(-1)
            e = F.leaky_relu(e, 0.2)
            
            zero_vec = torch.full_like(e, -9e15, device=device)
            
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            
            c = torch.matmul(attention, Wh)
            return c

    def forward(self, history_positions, num_steps, mc_noise=True):
        is_batched = history_positions.dim() == 4
        if not is_batched:
            history_positions = history_positions.unsqueeze(0)
            
        B, M, H, _ = history_positions.shape
        device = history_positions.device
        
        curr_hist = history_positions.clone()
        predictions = []
        
        for _ in range(num_steps):
            vel = (curr_hist[:, :, 1:, :] - curr_hist[:, :, :-1, :]) / self.dt
            v_flat = vel.reshape(B * M, -1)
            
            h = self.hist_encoder(v_flat)
            last_pos = curr_hist[:, :, -1, :]
            
            c = self.compute_gat_context(h, last_pos)
            
            # === ФИКС ПРОБЛЕМЫ ИНЕРЦИИ ===
            # Конкатенируем личную историю (h) и контекст графа (c)
            h_combined = torch.cat([h, c], dim=-1) # shape: [B*M, hidden_dim * 2]
            
            next_v = self.out_layer(h_combined).view(B, M, 2)

            if mc_noise:
                # Добавляем шум к СКОРОСТИ (намерениям), а не к координатам
                # 0.2 м/с - это легкое отклонение в шаге человека
                noise = torch.randn_like(next_v) * 0.05 
                next_v = next_v + noise
            # =============================
            
            predictions.append(next_v)
            
            next_pos = last_pos + next_v * self.dt
            curr_hist = torch.cat([curr_hist[:, :, 1:, :], next_pos.unsqueeze(2)], dim=2)
            
        preds_stacked = torch.stack(predictions, dim=2)
        
        if not is_batched:
            preds_stacked = preds_stacked.squeeze(0)
            
        return preds_stacked



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
            nn.Linear(hidden_dim, hidden_dim),
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

    # def forward(self, history_positions, num_steps):
    #     """Инференс через ODE решатель (Euler)"""
    #     M, H, _ = history_positions.shape
    #     device = history_positions.device
        
    #     curr_hist = history_positions.clone()
    #     predictions = []
        
    #     for _ in range(num_steps):
    #         vel = (curr_hist[:, 1:, :] - curr_hist[:, :-1, :]) / self.dt
    #         v_flat = vel.reshape(M, -1)
    #         h = self.hist_encoder(v_flat)
    #         last_pos = curr_hist[:, -1, :]
    #         c = self.compute_gat_context(h, last_pos)
            
    #         # Сэмплирование (ODE Euler step) из t=0 в t=1
    #         v_t = torch.randn((M, 2), device=device) # Изначальный шум N(0, I)
    #         steps = 10
    #         dt_ode = 1.0 / steps
            
    #         for step in range(steps):
    #             t = step * dt_ode
    #             v_vec = self.get_vector_field(v_t, t, c)
    #             v_t = v_t + v_vec * dt_ode # Шаг Эйлера
                
    #         next_v = v_t
    #         predictions.append(next_v)
    #         next_pos = last_pos + next_v * self.dt
    #         curr_hist = torch.cat([curr_hist[:, 1:, :], next_pos.unsqueeze(1)], dim=1)
            
    #     return torch.stack(predictions, dim=1)

    def forward(self, history_positions, num_steps):
        is_batched = history_positions.dim() == 4
        if not is_batched:
            history_positions = history_positions.unsqueeze(0)
            
        B, M, H, _ = history_positions.shape
        device = history_positions.device
        
        curr_hist = history_positions.clone()
        predictions = []
        
        for _ in range(num_steps):
            vel = (curr_hist[:, :, 1:, :] - curr_hist[:, :, :-1, :]) / self.dt
            v_flat = vel.reshape(B * M, -1)
            
            h = self.hist_encoder(v_flat)
            last_pos = curr_hist[:, :, -1, :] # [B, M, 2]
            
            c = self.compute_gat_context(h, last_pos) # [B*M, hidden_dim]
            
            # Сэмплирование (ODE Euler step)
            v_t = torch.randn((B * M, 2), device=device) # Изначальный шум N(0, I)
            steps = 10
            dt_ode = 1.0 / steps
            
            for step in range(steps):
                t = step * dt_ode
                v_vec = self.get_vector_field(v_t, t, c)
                v_t = v_t + v_vec * dt_ode # Шаг Эйлера
                
            next_v = v_t.view(B, M, 2)
            predictions.append(next_v)
            
            next_pos = last_pos + next_v * self.dt
            curr_hist = torch.cat([curr_hist[:, :, 1:, :], next_pos.unsqueeze(2)], dim=2)
            
        preds_stacked = torch.stack(predictions, dim=2) # [B, M, num_steps, 2]
        
        if not is_batched:
            preds_stacked = preds_stacked.squeeze(0)
            
        return preds_stacked

    def compute_fm_loss(self, history_positions, target_next_v):
        """Уравнение 8 из статьи: Обучение Flow Matching"""
        # M = history_positions.shape[0]
        # device = history_positions.device
        
        # vel = (history_positions[:, 1:, :] - history_positions[:, :-1, :]) / self.dt
        # v_flat = vel.reshape(M, -1)
        # h = self.hist_encoder(v_flat)
        # last_pos = history_positions[:, -1, :]
        # c = self.compute_gat_context(h, last_pos)
        
        # # Генерируем случайное время t и шум v_0
        # t = torch.rand((M, 1), device=device)
        # v_0 = torch.randn((M, 2), device=device)
        # v_1 = target_next_v # Целевая скорость
        
        # # Линейная интерполяция
        # v_t = t * v_1 + (1 - t) * v_0
        
        # # Предсказываем векторное поле
        # pred_vec = self.get_vector_field(v_t, t, c)
        
        # # Таргет для векторного поля = v_1 - v_0
        # target_vec = v_1 - v_0
        
        # return F.mse_loss(pred_vec, target_vec)
        # history_positions shape: [B, M, H, 2]
        B, M, H, _ = history_positions.shape
        device = history_positions.device
        
        # 1. Находим, кто реальный, а кто фантомный (добавленный через pad_collate)
        # Фантомные пешеходы стоят на 9999.0
        valid_mask = (history_positions[:, :, 0, 0] != 9999.0).view(B * M, 1) # [B*M, 1]
        
        vel = (history_positions[:, :, 1:, :] - history_positions[:, :, :-1, :]) / self.dt
        v_flat = vel.view(B * M, -1)
        h = self.hist_encoder(v_flat)
        last_pos = history_positions[:, :, -1, :] 
        
        c = self.compute_gat_context(h, last_pos) # [B*M, hidden]
        
        v_1 = target_next_v.view(B * M, 2)
        
        t = torch.rand((B * M, 1), device=device)
        v_0 = torch.randn((B * M, 2), device=device)
        v_t = t * v_1 + (1 - t) * v_0
        
        pred_vec = self.get_vector_field(v_t, t, c)
        target_vec = v_1 - v_0
        
        # 2. Обнуляем ошибку для фантомных пешеходов
        sq_err = ((pred_vec - target_vec) * valid_mask)**2
        
        # 3. Делим только на число реальных пешеходов
        num_valid = valid_mask.sum() * 2
        return sq_err.sum() / (num_valid + 1e-8)
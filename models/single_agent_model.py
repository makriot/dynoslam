import torch
import torch.nn as nn
from .base_model import BaseVelocityPredictor

class MLPVelocityPredictor(BaseVelocityPredictor):
    def __init__(self, history_len: int, dt: float, hidden_dim=64):
        super().__init__(history_len, dt)
        # На вход подаем (history_len - 1) скоростей, вычисленных из координат
        input_dim = (history_len - 1) * 2 
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out_layer = nn.Linear(hidden_dim, 2) # Предсказание скорости на 1 шаг

    def forward(self, history_positions, num_steps):
        B, H, _ = history_positions.shape
        device = history_positions.device
        
        # Переводим координаты в скорости
        # shape: [B, H-1, 2]
        velocities = (history_positions[:, 1:, :] - history_positions[:, :-1, :]) / self.dt
        v_flat = velocities.reshape(B, -1)
        
        predictions = []
        current_v_flat = v_flat
        
        # Авторегрессионное предсказание на num_steps вперед
        for _ in range(num_steps):
            feat = self.net(current_v_flat)
            next_v = self.out_layer(feat) # [B, 2]
            predictions.append(next_v)
            
            # Обновляем окно истории для следующего шага (сдвигаем)
            current_v_flat = torch.cat([current_v_flat[:, 2:], next_v], dim=1)
            
        return torch.stack(predictions, dim=1) # [B, num_steps, 2]
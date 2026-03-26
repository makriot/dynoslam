import torch
import torch.nn as nn

class BaseVelocityPredictor(nn.Module):
    def __init__(self, history_len: int, dt: float):
        super().__init__()
        self.history_len = history_len
        self.dt = dt

    def forward(self, history_positions, num_steps):
        """
        Предсказывает скорости на `num_steps` шагов вперед.
        
        :param history_positions: Тензор формы [Batch, history_len, 2] (координаты X, Y)
        :param num_steps: Сколько шагов скорости предсказать в будущее
        :return: Тензор предсказанных скоростей [Batch, num_steps, 2]
        """
        raise NotImplementedError("Нужно реализовать в дочернем классе")
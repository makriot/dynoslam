import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

class DynamicSLAMDataset:
    def __init__(self, csv_file, history_len=5, dt=0.1):
        print(f"Подготовка датасета для обучения из {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.history_len = history_len
        self.dt = dt
        self.samples = self._extract_samples()

    def _extract_samples(self):
        samples = []
        episodes = self.df['episode'].unique()
        
        for ep_id in tqdm(episodes, total=len(episodes)):
            ep_df = self.df[self.df['episode'] == ep_id].copy()
            times = np.sort(ep_df['time'].unique())
            
            time_data = []
            for t in times:
                ped_df = ep_df[(ep_df['time'] == t) & (ep_df['agent_type'] == 'pedestrian')]
                true_lms = {int(row['agent_id']): np.array([row['x'], row['y']]) for _, row in ped_df.iterrows()}
                time_data.append(true_lms)
            
            # Ищем окна H + 1
            for i in range(len(time_data) - self.history_len - 1):
                hist_steps = time_data[i : i + self.history_len]
                target_step = time_data[i + self.history_len]
                
                # Находим агентов, которые существовали на протяжении всей истории + таргетного кадра
                valid_ids = []
                for lmid in target_step.keys():
                    if all(lmid in step for step in hist_steps):
                        valid_ids.append(lmid)
                        
                if len(valid_ids) == 0:
                    continue
                    
                hist_tensor = torch.zeros((len(valid_ids), self.history_len, 2))
                target_v_tensor = torch.zeros((len(valid_ids), 2))
                
                for idx, lmid in enumerate(valid_ids):
                    for h_idx in range(self.history_len):
                        hist_tensor[idx, h_idx] = torch.tensor(hist_steps[h_idx][lmid], dtype=torch.float32)
                        
                    # Скорость = (m_{t+1} - m_t) / dt
                    pos_t = hist_tensor[idx, -1]
                    pos_t_next = torch.tensor(target_step[lmid], dtype=torch.float32)
                    target_v_tensor[idx] = (pos_t_next - pos_t) / self.dt
                    
                samples.append((hist_tensor, target_v_tensor))
                
        print(f"Собрано {len(samples)} обучающих сцен (каждая сцена содержит M агентов).")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
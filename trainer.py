import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.multi_agent_model import FlowMatchingGATPredictor

def train_model(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    # Переносим веса модели на GPU
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    is_flow_matching = isinstance(model, FlowMatchingGATPredictor)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # tqdm теперь оборачивает dataloader напрямую
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for hist_pos, target_v in pbar:
            # Сразу перекидываем батч с CPU на GPU
            hist_pos = hist_pos.to(device)
            target_v = target_v.to(device)
            
            optimizer.zero_grad()
            
            if is_flow_matching:
                loss = model.compute_fm_loss(hist_pos, target_v)
            else:
                # pred_v имеет форму [Batch, M, num_steps, 2] (т.е. [B, M, 1, 2])
                pred_v = model(hist_pos, num_steps=1)
                
                # Убираем измерение num_steps=1. 
                # Так как это размерность 2 (0=B, 1=M, 2=num_steps, 3=coords), мы делаем .squeeze(2)
                pred_v = pred_v.squeeze(2) 
                
                # Принудительно маскируем фиктивных пешеходов (которые мы добавили в pad_collate)
                # Фантомные пешеходы имеют target_v == 0.0 (мы так задали в pad_collate)
                # Мы не хотим, чтобы они влияли на лосс
                valid_mask = (hist_pos[:, :, 0, 0] != 9999.0).unsqueeze(-1) # [B, M, 1]
                
                pred_v_valid = pred_v * valid_mask
                target_v_valid = target_v * valid_mask
                
                # Считаем MSE только по реальным пешеходам
                # Суммируем квадраты ошибок и делим на количество РЕАЛЬНЫХ пешеходов
                sq_err = (pred_v_valid - target_v_valid)**2
                num_valid = valid_mask.sum() * 2 # Умножаем на 2, т.к. 2 координаты
                
                loss = sq_err.sum() / (num_valid + 1e-8) # +1e-8 чтобы избежать деления на 0
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} завершена. Avg Loss: {total_loss / len(dataloader):.4f}")
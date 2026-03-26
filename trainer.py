import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.multi_agent_model import FlowMatchingGATPredictor

def train_model(model, dataset, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Определяем, используем ли мы Flow Matching
    is_flow_matching = isinstance(model, FlowMatchingGATPredictor)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Перемешиваем индексы для эпохи
        indices = torch.randperm(len(dataset)).tolist()
        
        pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}")
        for idx in pbar:
            hist_pos, target_v = dataset[idx]
            hist_pos = hist_pos.to(device)
            target_v = target_v.to(device)
            
            optimizer.zero_grad()
            
            if is_flow_matching:
                loss = model.compute_fm_loss(hist_pos, target_v)
            else:
                # MLP или SimpleGAT
                pred_v = model(hist_pos, num_steps=1).squeeze(1) # [M, 1, 2] -> [M, 2]
                loss = F.mse_loss(pred_v, target_v)
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} | Average Loss: {total_loss / len(dataset):.4f}")
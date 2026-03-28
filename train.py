import torch
import os
from train_dataset import DynamicSLAMDataset
from torch.utils.data import DataLoader
from trainer import train_model
from models.single_agent_model import MLPVelocityPredictor
from models.multi_agent_model import SimpleGATPredictor, FlowMatchingGATPredictor

def pad_collate(batch):
    """
    Добивает количество агентов (M) в каждой сцене до максимального в батче,
    чтобы PyTorch мог сложить их в один прямоугольный тензор.
    """
    # batch - это список из 32 кортежей (по размеру batch_size)
    # Каждый кортеж: (hist_tensor [M_i, H, 2], target_v [M_i, 2])
    max_m = max([item[0].shape[0] for item in batch])
    
    hist_padded = []
    target_padded = []
    
    for hist, target in batch:
        m = hist.shape[0]
        pad_len = max_m - m
        
        if pad_len > 0:
            # Фантомные пешеходы (координата 9999.0, чтобы GAT их игнорировал)
            pad_h = torch.ones((pad_len, hist.shape[1], 2)) * 9999.0
            hist = torch.cat([hist, pad_h], dim=0)
            
            # Нулевая скорость для фантомов
            pad_t = torch.zeros((pad_len, 2))
            target = torch.cat([target, pad_t], dim=0)
            
        hist_padded.append(hist)
        target_padded.append(target)
        
    return torch.stack(hist_padded), torch.stack(target_padded)

def main():
    os.makedirs("weights", exist_ok=True)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # device = "cuda:1"
    print(f"Using device: {device}")

    # 1. Загружаем тренировочный датасет
    dataset = DynamicSLAMDataset("data/data_train.csv", history_len=5, dt=0.1)
    
    # ==========================================
    # ОБУЧЕНИЕ МОДЕЛИ 1: MLP (Single Agent)
    # ==========================================
    # print("\n--- Training MLP Predictor ---")
    # mlp = MLPVelocityPredictor(history_len=5, dt=0.1, hidden_dim=64)
    # train_model(mlp, dataset, epochs=5, lr=1e-3, device=device)
    # torch.save(mlp.state_dict(), "weights/mlp_best.pth")
    
    # ==========================================
    # ОБУЧЕНИЕ МОДЕЛИ 2: Simple GAT
    # ==========================================
    # print("\n--- Training Simple GAT Predictor ---")
    # gat = SimpleGATPredictor(history_len=5, dt=0.1, hidden_dim=64, interaction_radius=5.0)
    # train_model(gat, dataset, epochs=5, lr=1e-3, device=device)
    # torch.save(gat.state_dict(), "weights/gat_best.pth")

    # ==========================================
    # ОБУЧЕНИЕ МОДЕЛИ 3: GAT + Flow Matching
    # ==========================================
    # print("\n--- Training GAT + Flow Matching Predictor ---")
    # fm_gat = FlowMatchingGATPredictor(history_len=5, dt=0.1, hidden_dim=64, interaction_radius=5.0)
    # train_model(fm_gat, dataset, epochs=5, lr=1e-3, device=device)
    # torch.save(fm_gat.state_dict(), "weights/fm_gat_best.pth")

    dataloader = DataLoader(
        dataset, 
        batch_size=32,       
        shuffle=True, 
        collate_fn=pad_collate, # <---- ВОТ ЭТА СТРОЧКА ИСПРАВЛЯЕТ ОШИБКУ
        num_workers=4, 
        pin_memory=True
    )

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    gat = SimpleGATPredictor(history_len=5, dt=0.1, hidden_dim=64, interaction_radius=5.0)
    train_model(gat, dataloader, epochs=40, lr=1e-3, device=device)
    torch.save(gat.state_dict(), "weights/gat_best.pth")

    print("\nОбучение завершено! Веса сохранены в папку weights/")

if __name__ == '__main__':
    main()
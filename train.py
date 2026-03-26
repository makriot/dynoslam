import torch
import os
from train_dataset import DynamicSLAMDataset
from trainer import train_model
from models.single_agent_model import MLPVelocityPredictor
from models.multi_agent_model import SimpleGATPredictor, FlowMatchingGATPredictor

def main():
    os.makedirs("weights", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Загружаем тренировочный датасет
    dataset = DynamicSLAMDataset("data/data_train.csv", history_len=5, dt=0.1)
    
    # ==========================================
    # ОБУЧЕНИЕ МОДЕЛИ 1: MLP (Single Agent)
    # ==========================================
    print("\n--- Training MLP Predictor ---")
    mlp = MLPVelocityPredictor(history_len=5, dt=0.1, hidden_dim=64)
    train_model(mlp, dataset, epochs=5, lr=1e-3, device=device)
    torch.save(mlp.state_dict(), "weights/mlp_best.pth")
    
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

    print("\nОбучение завершено! Веса сохранены в папку weights/")

if __name__ == '__main__':
    main()
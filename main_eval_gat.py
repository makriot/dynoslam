import torch
import os
from data_processor import DataProcessor
from evaluate import evaluate_dataset, evaluate_episode

from slam.cv_dynamic_slam import CVDynamicSLAM
from slam.nn_dynamic_slam import NeuralDynamicSLAM

from models.single_agent_model import MLPVelocityPredictor
from models.multi_agent_model import SimpleGATPredictor, FlowMatchingGATPredictor

def main():
    print("Инициализация DataProcessor...")
    
    # Настройки окна и симулятора
    window_size = 20 # Окно оптимизации SLAM
    history_len = 5  # Окно истории для нейросетей
    DT = 0.1         # СТРОГО 0.1, как при обучении!
    
    # Используй data_test.csv для честных метрик!
    dp = DataProcessor(
        "data/data_test.csv", 
        window_size=window_size, 
        dt=DT, 
        history_len=history_len
    )
    
    # Устройство (cpu или cuda)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство для инференса: {device}")
    
    # # =========================================================================
    # # 1. Baseline: Constant Velocity
    # # =========================================================================
    # print("\n=== Тестирование Constant Velocity Model (Baseline) ===")
    cv_slam = CVDynamicSLAM(window_size=window_size, dt=DT, sigma_acc=0.5).to(device)
    evaluate_dataset(cv_slam, dp, num_epochs=100, output_csv="graphslam_metrics_horizon20.csv")


    # =========================================================================
    # 2. Neural SLAM: Simple GAT (Твоя обученная модель)
    # =========================================================================
    # print("\n=== Тестирование Neural Dynamic SLAM (Simple GAT Predictor) ===")
    
    # Создаем архитектуру точно с такими же параметрами, как при обучении
    # gat_model = SimpleGATPredictor(
    #     history_len=history_len, 
    #     dt=DT, 
    #     hidden_dim=64, 
    #     interaction_radius=5.0
    # )
    
    # # Загружаем обученные веса
    # weights_path = "weights/gat_best.pth"
    # if os.path.exists(weights_path):
    #     # map_location позволяет грузить веса, обученные на GPU, даже на CPU
    #     gat_model.load_state_dict(torch.load(weights_path, map_location=device))
    #     print(f"Успешно загружены веса: {weights_path}")
    # else:
    #     print(f"ВНИМАНИЕ: Файл {weights_path} не найден! Модель будет со случайными весами.")
    
    # gat_model.to(device)
    # gat_model.eval() # Обязательно переводим в режим инференса!
    
    # # # Оборачиваем GAT в GraphSLAM
    # gat_slam = NeuralDynamicSLAM(
    #     predictor_model=gat_model, 
    #     window_size=window_size, 
    #     dt=DT, 
    #     sigma_kin=0.5 # Доверие к нейросети
    # ).to(device)
    
    # # Запускаем метрики
    # gat_r_rmse, gat_lm_rmse = evaluate_dataset(gat_slam, dp, num_epochs=100, output_csv="gat_metrics_horizon20_straight.csv")
    
    
    # # =========================================================================
    # # ВЫВОД РЕЗУЛЬТАТОВ (Можно скопировать в статью)
    # # =========================================================================
    # print("\n====================================================")
    # print(f"                 ИТОГОВОЕ СРАВНЕНИЕ (W={window_size})")
    # print("====================================================")
    # # print(f"CV Baseline -> Robot RMSE: {cv_r_rmse:.4f}, Landmarks RMSE: {cv_lm_rmse:.4f}")
    # print(f"GAT SLAM    -> Robot RMSE: {gat_r_rmse:.4f}, Landmarks RMSE: {gat_lm_rmse:.4f}")
    
    
    # (Опционально) Если хочешь отрисовать одно окно для статьи, раскомментируй:
    # print("\nГенерация графика для GAT...")
    # evaluate_slam(gat_slam, dp, episode_id=0, num_epochs=100, plot_window_idx=5)

    # mlp_model = MLPVelocityPredictor(history_len=history_len, dt=DT, hidden_dim=64)
    
    # mlp_model.load_state_dict(torch.load("weights/mlp_best.pth"))
    
    # mlp_slam = NeuralDynamicSLAM(
    #     predictor_model=mlp_model, 
    #     window_size=window_size, 
    #     dt=DT, 
    #     sigma_kin=0.1
    # )

    # evaluate_dataset(mlp_slam, dp, num_epochs=100, output_csv="mlp_metrics_horizon20_new.csv")

    episode_id = 7

    # evaluate_episode(
    #     cv_slam, dp, episode_id=episode_id, num_epochs=100, 
    #     plot_window_idx=6, prediction_horizon=10, save_prefix="cv_baseline"
    # )

    # # 2. MLP Model
    # evaluate_episode(
    #     mlp_slam, dp, episode_id=episode_id, num_epochs=100, 
    #     plot_window_idx=6, prediction_horizon=10, save_prefix="mlp_model"
    # )

    # 3. GAT Model
    # evaluate_episode(
    #     gat_slam, dp, episode_id=episode_id, num_epochs=100, 
    #     plot_window_idx=6, prediction_horizon=10, save_prefix="gat_model_stochastic"
    # )

if __name__ == '__main__':
    main()

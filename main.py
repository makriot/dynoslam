

from data_processor import DataProcessor
from slam.cv_dynamic_slam import CVDynamicSLAM
from slam.nn_dynamic_slam import NeuralDynamicSLAM
from models.single_agent_model import MLPVelocityPredictor
from evaluate import evaluate_slam, evaluate_dataset

def main():
    print("Инициализация DataProcessor...")
    # Берем данные. Если у тебя есть реальный all_data.csv, укажи путь к нему
    window_size = 20
    history_len = 5
    dp = DataProcessor("data/mock_data.csv", window_size=window_size, dt=0.11)
    
    # 1. Тестируем Baseline (Constant Velocity)
    print("\n=== Тестирование Constant Velocity Model (Baseline) ===")
    # cv_slam = CVDynamicSLAM(
    #     window_size=window_size, 
    #     dt=0.11, 
    #     sigma_acc=0.5 # Штраф за ускорение. Попробуй поменять на 0.1 или 1.0
    # )
    
    # Запускаем оценку (выбираем эпизод 0, рисуем окно 5)
    # evaluate_slam(cv_slam, dp, episode_id=0, num_epochs=100, plot_window_idx=6)

    # print("\n=== Полный бенчмарк ===")
    # evaluate_dataset(cv_slam, dp, num_epochs=100, output_csv=f"constant_vel_horizon{window_size}.csv")

    print("\n=== Тестирование Neural Dynamic SLAM (MLP Predictor) ===")
    
    # Инициализируем модель (пока со случайными весами!)
    mlp_model = MLPVelocityPredictor(history_len=history_len, dt=0.1, hidden_dim=64)
    
    # TODO: В будущем здесь будет загрузка обученных весов:
    # mlp_model.load_state_dict(torch.load("weights/mlp_best.pth"))
    
    nn_slam = NeuralDynamicSLAM(
        predictor_model=mlp_model, 
        window_size=window_size, 
        dt=0.11, 
        sigma_kin=0.1 # Насколько сильно доверять нейронке
    )
    
    nn_r_rmse, nn_lm_rmse = evaluate_dataset(nn_slam, dp, num_epochs=100, output_csv="single_agent_mock.csv")
    
    # print(f"CV Baseline -> Robot RMSE: {cv_r_rmse:.4f}, Landmarks RMSE: {cv_lm_rmse:.4f}")
    print(f"Neural SLAM -> Robot RMSE: {nn_r_rmse:.4f}, Landmarks RMSE: {nn_lm_rmse:.4f}")



if __name__ == '__main__':
    main()

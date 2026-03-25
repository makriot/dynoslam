import torch
from data_processor import DataProcessor
from slam.cv_dynamic_slam import CVDynamicSLAM

def main():
    # 1. Загружаем данные и формируем окна

    dt = 0.11
    window_size = 10

    print("Инициализация DataProcessor...")
    dp = DataProcessor("data/mock_data.csv", window_size=window_size, dt=dt)
    
    # Достаем все окна для Эпизода 0
    windows = dp.get_episode_windows(episode_id=0)
    print(f"Извлечено окон: {len(windows)}\n")
    
    # 2. Инициализируем Constant Velocity GraphSLAM
    slam = CVDynamicSLAM(window_size=window_size, dt=dt)
    
    # Возьмем для теста одно окно
    w = windows[0]
    
    init_robot_pose = torch.tensor(w['init_robot_pose'], dtype=torch.float32)
    odometry = torch.tensor(w['odometry'], dtype=torch.float32)
    
    # Переводим наблюдения в тензоры
    observations = []
    for obs_step in w['observations']:
        obs_step_tensors = []
        for o in obs_step:
            obs_step_tensors.append({
                'lm_id': o['lm_id'],
                'range': torch.tensor(o['range'], dtype=torch.float32),
                'bearing': torch.tensor(o['bearing'], dtype=torch.float32)
            })
        observations.append(obs_step_tensors)
        
    print("Запуск оптимизации через PyTorch (Градиентный спуск)...")
    robot_poses, landmarks, predictions = slam(
        init_robot_pose, odometry, observations, num_epochs=100, prediction_horizon=10
    )

    
    print("\n[Результаты]")
    # print(f"Финальные позы робота:\n{robot_poses[-1]}")
    print("Robot poses", robot_poses)
    # if predictions:
    #     first_lm = list(predictions.keys())[0]
    #     print(f"Предсказанная траектория (на 10 шагов вперед) для пешехода {first_lm}:\n{predictions[first_lm]}")
    
    print("landmarks", landmarks)

if __name__ == '__main__':
    main()

import torch
import os
import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_processor import DataProcessor
from evaluate import evaluate_episode  # Импортируем именно evaluate_episode!

from slam.nn_dynamic_slam import NeuralDynamicSLAM
from models.multi_agent_model import SimpleGATPredictor

# Глобальные настройки
WINDOW_SIZE = 20
HISTORY_LEN = 5
DT = 0.1
DATA_CSV = "data/data_test.csv"
WEIGHTS_PATH = "weights/gat_best.pth"
NUM_WORKERS = 8  # Количество параллельных процессов
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV = "gat_stochastic_metrics_horizon20.csv"

def init_worker_model(device):
    """
    Каждый процесс должен создать СВОЮ копию модели.
    Если процессы делят одну модель, PyTorch может упасть с ошибкой CUDA.
    """
    gat_model = SimpleGATPredictor(
        history_len=HISTORY_LEN, 
        dt=DT, 
        hidden_dim=64, 
        interaction_radius=5.0
    )
    
    if os.path.exists(WEIGHTS_PATH):
        gat_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        print(f"ВНИМАНИЕ: Файл {WEIGHTS_PATH} не найден. Используем случайные веса.")
        
    gat_model.to(device)
    gat_model.eval()
    
    gat_slam = NeuralDynamicSLAM(
        predictor_model=gat_model, 
        window_size=WINDOW_SIZE, 
        dt=DT, 
        sigma_kin=0.5 
    ).to(device)
    
    return gat_slam

def process_single_episode(args):
    """
    Эта функция выполняется в отдельном процессе.
    Она берет ID эпизода, прогоняет его и возвращает результаты.
    """
    ep_id, device = args
    
    # 1. Загружаем данные ТОЛЬКО для этого эпизода
    dp = DataProcessor(
        DATA_CSV, 
        window_size=WINDOW_SIZE, 
        dt=DT, 
        history_len=HISTORY_LEN
    )
    
    # 2. Создаем модель для этого потока
    slam_model = init_worker_model(device)
    
    # 3. Запускаем базовую функцию оценки из evaluate.py
    # ТЕПЕРЬ ОНА ВОЗВРАЩАЕТ 8 СПИСКОВ (добавился pred_rmse_list)
    r_rmse, l_rmse, ate, rpe_m, rpe_s, sde, time_ms, pred_rmse = evaluate_episode(
        slam_model, 
        dp, 
        episode_id=ep_id, 
        num_epochs=100, 
        plot_window_idx=None,
        prediction_horizon=10 # Передаем горизонт предсказания
    )
    
    # Возвращаем кортеж со всеми 9 значениями (1 ID + 8 массивов)
    return ep_id, r_rmse, l_rmse, ate, rpe_m, rpe_s, sde, time_ms, pred_rmse

def main():
    print("=== Параллельный инференс Neural Dynamic SLAM (GAT) ===")
    print(f"Устройство: {DEVICE} | Рабочих процессов: {NUM_WORKERS}")
    
    df = pd.read_csv(DATA_CSV, usecols=['episode'])
    unique_episodes = df['episode'].unique().tolist()
    print(f"Найдено эпизодов для оценки: {len(unique_episodes)}")
    
    tasks = [(ep_id, DEVICE) for ep_id in unique_episodes]
    
    mp.set_start_method('spawn', force=True)
    
    # Списки для агрегации всех 8 метрик
    all_robot_rmse = []
    all_lm_rmse = []
    all_ate = []
    all_rpe_m = []
    all_rpe_s = []
    all_sde = []
    all_time_ms = []
    all_pred_rmse = [] # НОВЫЙ СПИСОК ДЛЯ БУДУЩЕГО
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results_iter = pool.imap_unordered(process_single_episode, tasks)
        
        # Распаковываем 9 значений
        for result in tqdm(results_iter, total=len(tasks), desc="Обработка эпизодов"):
            ep_id, r_rmse, l_rmse, ate, rpe_m, rpe_s, sde, time_ms, pred_rmse = result
            
            # Расширяем глобальные списки
            all_robot_rmse.extend(r_rmse)
            all_lm_rmse.extend(l_rmse)
            all_ate.extend(ate)
            all_rpe_m.extend(rpe_m)
            all_rpe_s.extend(rpe_s)
            all_sde.extend(sde)
            all_time_ms.extend(time_ms)
            all_pred_rmse.extend(pred_rmse)

    if not all_robot_rmse:
        print("Ошибка: Не получено никаких результатов!")
        return
        
    # Вспомогательная функция для подсчета среднего без учета NaN
    def get_mean(lst):
        arr = np.array(lst)
        valid_vals = arr[~np.isnan(arr)]
        if len(valid_vals) == 0:
            return 0.0
        return np.mean(valid_vals)

    # Считаем средние для вывода в консоль
    global_r_rmse = get_mean(all_robot_rmse)
    global_lm_rmse = get_mean(all_lm_rmse)
    global_ate = get_mean(all_ate)
    global_pred_rmse = get_mean(all_pred_rmse) # Средняя ошибка предсказания
    
    # 5. Сохраняем все метрики в CSV
    max_len = len(all_robot_rmse)
    
    def pad_list(lst):
        if len(lst) == max_len:
            return lst
        return lst + [np.nan] * (max_len - len(lst))

    res_df = pd.DataFrame({
        'robot_rmse': pad_list(all_robot_rmse),
        'landmark_rmse': pad_list(all_lm_rmse),
        'ate': pad_list(all_ate),
        'rpe_mean': pad_list(all_rpe_m),
        'rpe_std': pad_list(all_rpe_s),
        'sde': pad_list(all_sde),
        'time_ms': pad_list(all_time_ms),
        'prediction_rmse': pad_list(all_pred_rmse) # ДОБАВЛЯЕМ В CSV
    })
    
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nДетальные метрики сохранены в: {OUTPUT_CSV}")

    # 6. Вывод итогов
    print("\n====================================================")
    print(f"                 ИТОГОВОЕ СРАВНЕНИЕ (W={WINDOW_SIZE})")
    print("====================================================")
    print(f"GAT SLAM -> Robot RMSE: {global_r_rmse:.4f}")
    print(f"GAT SLAM -> Landmarks RMSE: {global_lm_rmse:.4f}")
    print(f"GAT SLAM -> ATE: {global_ate:.4f}")
    print(f"GAT SLAM -> PREDICTION RMSE: {global_pred_rmse:.4f}  <--- (Ошибка прогноза будущего)")

if __name__ == '__main__':
    main()
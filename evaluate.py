import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


import numpy as np
import time
from scipy.linalg import orthogonal_procrustes

def align_trajectory(est, gt):
    """
    Выравнивание траектории по Umeyama (трансляция + вращение).
    est, gt: numpy arrays (N, 2)
    Возвращает выровненную est.
    """
    est_centered = est - np.mean(est, axis=0)
    gt_centered = gt - np.mean(gt, axis=0)
    R, _ = orthogonal_procrustes(est_centered, gt_centered)  # R — матрица вращения 2x2
    est_aligned = est_centered @ R + np.mean(gt, axis=0)
    return est_aligned

def compute_ate(est, gt):
    """
    RMSE после выравнивания траектории.
    est, gt: (N, 2)
    """
    est_aligned = align_trajectory(est, gt)
    return np.sqrt(np.mean(np.sum((est_aligned - gt)**2, axis=1)))

def compute_rpe_trans(est, gt, interval_frames=1):
    """
    Относительная ошибка трансляции (RMSE) для заданного интервала в кадрах.
    Возвращает mean, std.
    """
    errors = []
    for i in range(len(est) - interval_frames):
        est_delta = est[i + interval_frames] - est[i]
        gt_delta = gt[i + interval_frames] - gt[i]
        errors.append(np.linalg.norm(est_delta - gt_delta))
    return np.mean(errors), np.std(errors)

def compute_sde(robot_est, robot_gt, ped_est, ped_gt, time_steps):
    """
    Safety Distance Error: разница между оценённым и истинным расстоянием
    от робота до ближайшего пешехода.
    robot_est, robot_gt: (T,2)
    ped_est: (T, M, 2) — позиции всех пешеходов (NaN если не виден)
    ped_gt: список из T словарей {lm_id: (x,y)}
    time_steps: количество шагов в окне (T)
    Возвращает среднюю ошибку расстояния по всем шагам.
    """
    errors = []
    for t in range(time_steps):
        # Истинное расстояние до ближайшего пешехода
        if not ped_gt[t]:
            continue  # нет пешеходов в этот момент — пропускаем
        true_positions = np.array(list(ped_gt[t].values()))
        true_dists = np.linalg.norm(true_positions - robot_gt[t], axis=1)
        true_min_dist = np.min(true_dists)
        
        # Оценённое расстояние до ближайшего пешехода (из тех, что есть в ped_est[t])
        # В ped_est[t] могут быть NaN для невидимых — нужно фильтровать
        est_positions = ped_est[t][~np.isnan(ped_est[t]).any(axis=1)]
        if len(est_positions) == 0:
            continue
        est_dists = np.linalg.norm(est_positions - robot_est[t], axis=1)
        est_min_dist = np.min(est_dists)
        
        errors.append(abs(true_min_dist - est_min_dist))
    return np.mean(errors) if errors else np.nan


def plot_window_results(window_data, est_robot_poses, est_landmarks, predictions, lm_ids, save_path="slam_result.png"):
    """
    Отрисовывает одно окно: Ground Truth vs Estimates vs Predictions.
    """
    plt.figure(figsize=(10, 10))
    
    # 1. Истинная траектория робота
    true_robot = np.array([step[:2] for step in window_data['true_trajectory']])
    plt.plot(true_robot[:, 0], true_robot[:, 1], 'b-', linewidth=2, label='True Robot Path', alpha=0.6)
    plt.scatter(true_robot[0, 0], true_robot[0, 1], c='blue', marker='o', s=100, label='Robot Start')
    
    # 2. Оцененная траектория робота
    est_robot = est_robot_poses.numpy()[:, :2]
    plt.plot(est_robot[:, 0], est_robot[:, 1], 'r--', linewidth=2, label='Estimated Robot Path')
    
    # 3. Истинные траектории пешеходов
    W = len(window_data['true_trajectory']) - 1
    true_ped_paths = {lmid: [] for lmid in lm_ids}
    for t in range(W + 1):
        true_lms = window_data['true_landmarks'][t]
        for lmid in lm_ids:
            if lmid in true_lms:
                true_ped_paths[lmid].append(true_lms[lmid])
    
    for lmid, path in true_ped_paths.items():
        if len(path) > 0:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, alpha=0.5)
            plt.scatter(path[0, 0], path[0, 1], c='green', marker='s')
            
    plt.plot([], [], 'g-', linewidth=2, label='True Pedestrians')
    
    # 4. Оцененные траектории пешеходов
    est_lms = est_landmarks.numpy()
    for idx, lmid in enumerate(lm_ids):
        path = est_lms[:, idx, :]
        plt.plot(path[:, 0], path[:, 1], color='orange', linestyle='--', linewidth=2)
        
    plt.plot([], [], color='orange', linestyle='--', linewidth=2, label='Estimated Pedestrians')

    # 5. Предсказания
    if predictions:
        for lmid, pred_tensor in predictions.items():
            pred_path = pred_tensor.numpy()
            idx = lm_ids.index(lmid)
            last_est = est_lms[-1, idx, :]
            full_pred = np.vstack([last_est, pred_path])
            
            plt.plot(full_pred[:, 0], full_pred[:, 1], color='purple', linestyle=':', linewidth=2)
        plt.plot([], [], color='purple', linestyle=':', linewidth=2, label='Predictions (Future)')

    plt.title("Dynamic GraphSLAM: Ground Truth vs Estimates")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_episode(slam_model, dp, episode_id, num_epochs=100, plot_window_idx=None):
    """
    Базовая функция: прогоняет SLAM по всем окнам одного эпизода.
    Возвращает списки RMSE для каждого окна. Не делает выводов в консоль (кроме отрисовки).
    """
    windows = dp.get_episode_windows(episode_id)
    if not windows:
        return [], [], [], [], [], [], []

    robot_rmse_list = []
    lm_rmse_list = []
    ate_list = []
    rpe_mean_list = []
    rpe_std_list = []
    sde_list = []
    time_list = []

    for i, w in enumerate(windows):
        init_pose = torch.tensor(w['init_robot_pose'], dtype=torch.float32)
        odom = torch.tensor(w['odometry'], dtype=torch.float32)
        
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
        
        lm_history = w.get('lm_history', None)
            
        start_time = time.time()
        robot_poses, landmarks, predictions = slam_model(
            init_pose, odom, observations, lm_history=lm_history, num_epochs=num_epochs, prediction_horizon=10
        )
        elapsed_ms = (time.time() - start_time) * 1000
        time_list.append(elapsed_ms)

        # РАСЧЕТ ОШИБОК
        true_robot = np.array([step[:2] for step in w['true_trajectory']])
        est_robot = robot_poses.numpy()[:, :2]

        # 1. Robot RMSE (уже есть)
        r_rmse = np.sqrt(np.mean(np.sum((true_robot - est_robot)**2, axis=1)))
        robot_rmse_list.append(r_rmse)

        # 2. ATE
        ate = compute_ate(est_robot, true_robot)
        ate_list.append(ate)

        # 3. RPE (например, интервал 1 кадр = 0.11 с)
        rpe_mean, rpe_std = compute_rpe_trans(est_robot, true_robot, interval_frames=1)
        rpe_mean_list.append(rpe_mean)
        rpe_std_list.append(rpe_std)

        # 4. Landmark RMSE (как раньше)
        lm_ids = set()
        for obs_step in w['observations']:
            for o in obs_step:
                lm_ids.add(o['lm_id'])
        lm_ids = sorted(list(lm_ids))
        
        lm_sq_errors = []
        for t_idx in range(len(w['true_landmarks'])):
            true_lms_t = w['true_landmarks'][t_idx]
            for lm_idx, lmid in enumerate(lm_ids):
                if lmid in true_lms_t:
                    true_pos = true_lms_t[lmid]
                    est_pos = landmarks[t_idx, lm_idx].numpy()
                    dist_sq = np.sum((true_pos - est_pos)**2)
                    lm_sq_errors.append(dist_sq)
                    
        if lm_sq_errors:
            l_rmse = np.sqrt(np.mean(lm_sq_errors))
            lm_rmse_list.append(l_rmse)
        else:
            lm_rmse_list.append(np.nan)
        
        # 5. SDE
        # Сформируем ped_est как (T, M, 2) с NaN для отсутствующих объектов
        T = len(w['true_landmarks'])
        M = len(lm_ids)
        ped_est = np.full((T, M, 2), np.nan)
        for t_idx in range(T):
            for lm_idx, lmid in enumerate(lm_ids):
                if not np.isnan(landmarks[t_idx, lm_idx, 0]):  # предполагаем, что если видим, то не NaN
                    ped_est[t_idx, lm_idx] = landmarks[t_idx, lm_idx].numpy()
        ped_gt = w['true_landmarks']  # список словарей
        sde = compute_sde(est_robot, true_robot, ped_est, ped_gt, T)
        sde_list.append(sde)

        # Отрисовка
        if plot_window_idx is not None and i == plot_window_idx:
            plot_window_results(w, robot_poses, landmarks, predictions, lm_ids, save_path=f"window_ep{episode_id}_{i}_results.png")
            print(f"График для эпизода {episode_id}, окна {i} сохранен.")

    return (robot_rmse_list, lm_rmse_list, ate_list, rpe_mean_list, rpe_std_list, sde_list, time_list)


def evaluate_slam(slam_model, dp, episode_id=0, num_epochs=100, plot_window_idx=0):
    """
    Запускает оценку для одного эпизода (идеально для дебага и визуализации).
    """
    print(f"--- Запуск оценки (Эпизод {episode_id}) ---")
    r_list, l_list, ate_list, rpe_mean_list, rpe_std_list, sde_list, time_list = evaluate_episode(slam_model, dp, episode_id, num_epochs, plot_window_idx)
    
    if not r_list:
        print("Окна не найдены!")
        return 0.0, 0.0

    mean_r = np.mean(r_list)
    mean_l = np.mean(l_list) if l_list else 0.0

    print("\n[Итоговые метрики качества для эпизода]")
    print(f"Количество обработанных окон: {len(r_list)}")
    print(f"Robot Trajectory RMSE: {mean_r:.4f} м")
    print(f"Landmarks (Pedestrians) RMSE: {mean_l:.4f} м")
    print("ate_list, rpe_mean_list, rpe_std_list, sde_list, time_list:", ate_list, rpe_mean_list, rpe_std_list, sde_list, time_list)
    
    return mean_r, mean_l


def evaluate_dataset(slam_model, dp, num_epochs=100, output_csv="evaluation_metrics.csv"):
    """
    Прогоняет алгоритм по ВСЕМ эпизодам в датасете, агрегирует ошибки и выдает глобальную метрику.
    Отключает любые графики для скорости.
    """
    unique_episodes = dp.df['episode'].unique()
    print(f"--- Запуск оценки всего датасета ({len(unique_episodes)} эпизодов) ---")
    
    all_robot_rmse = []
    all_lm_rmse = []
    all_ate = []
    all_rpe_mean = []
    all_rpe_std = []
    all_sde = []
    all_time = []
    
    # tqdm создаст красивый прогресс-бар в консоли
    for ep_id in tqdm(unique_episodes, desc="Оценка эпизодов", unit="ep"):
        # plot_window_idx=None гарантирует, что мы не будем тратить время на графики

        (r_list, l_list, ate_list, rpe_mean_list, rpe_std_list, sde_list, time_list) = \
            evaluate_episode(slam_model, dp, ep_id, num_epochs, plot_window_idx=None)
        
        all_robot_rmse.extend(r_list)
        all_lm_rmse.extend(l_list)
        all_ate.extend(ate_list)
        all_rpe_mean.extend(rpe_mean_list)
        all_rpe_std.extend(rpe_std_list)
        all_sde.extend(sde_list)
        all_time.extend(time_list)
    
    def stats(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan, np.nan, np.nan, np.nan
        return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

    metrics = {
        'metric': ['robot_rmse', 'landmark_rmse', 'ate', 'rpe_mean', 'rpe_std', 'sde', 'time_ms'],
        'count': [len(all_robot_rmse), len(all_lm_rmse), len(all_ate), len(all_rpe_mean), len(all_rpe_std), len(all_sde), len(all_time)],
        'mean': [stats(all_robot_rmse)[0], stats(all_lm_rmse)[0], stats(all_ate)[0], stats(all_rpe_mean)[0], stats(all_rpe_std)[0], stats(all_sde)[0], stats(all_time)[0]],
        'std':  [stats(all_robot_rmse)[1], stats(all_lm_rmse)[1], stats(all_ate)[1], stats(all_rpe_mean)[1], stats(all_rpe_std)[1], stats(all_sde)[1], stats(all_time)[1]],
        'min':  [stats(all_robot_rmse)[2], stats(all_lm_rmse)[2], stats(all_ate)[2], stats(all_rpe_mean)[2], stats(all_rpe_std)[2], stats(all_sde)[2], stats(all_time)[2]],
        'max':  [stats(all_robot_rmse)[3], stats(all_lm_rmse)[3], stats(all_ate)[3], stats(all_rpe_mean)[3], stats(all_rpe_std)[3], stats(all_sde)[3], stats(all_time)[3]],
    }
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_csv, index=False)
    print(f"\nГлобальные метрики сохранены в {output_csv}")
    print(metrics_df.to_string())
    
    # Возвращаем основные значения для обратной совместимости
    return metrics_df.loc[metrics_df['metric']=='ate', 'mean'].values[0], \
           metrics_df.loc[metrics_df['metric']=='sde', 'mean'].values[0]

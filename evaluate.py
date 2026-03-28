import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.linalg import orthogonal_procrustes

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Ошибки)
# ==============================================================================
def align_trajectory(est, gt):
    est_centered = est - np.mean(est, axis=0)
    gt_centered = gt - np.mean(gt, axis=0)
    R, _ = orthogonal_procrustes(est_centered, gt_centered)
    est_aligned = est_centered @ R + np.mean(gt, axis=0)
    return est_aligned

def compute_ate(est, gt):
    est_aligned = align_trajectory(est, gt)
    return np.sqrt(np.mean(np.sum((est_aligned - gt)**2, axis=1)))

def compute_rpe_trans(est, gt, interval_frames=1):
    errors = []
    for i in range(len(est) - interval_frames):
        est_delta = est[i + interval_frames] - est[i]
        gt_delta = gt[i + interval_frames] - gt[i]
        errors.append(np.linalg.norm(est_delta - gt_delta))
    return np.mean(errors) if errors else np.nan, np.std(errors) if errors else np.nan

def compute_sde(robot_est, robot_gt, ped_est, ped_gt, time_steps):
    errors = []
    for t in range(time_steps):
        if not ped_gt[t]:
            continue
        true_positions = np.array(list(ped_gt[t].values()))
        true_dists = np.linalg.norm(true_positions - robot_gt[t], axis=1)
        true_min_dist = np.min(true_dists)
        
        est_positions = ped_est[t][~np.isnan(ped_est[t]).any(axis=1)]
        if len(est_positions) == 0:
            continue
        est_dists = np.linalg.norm(est_positions - robot_est[t], axis=1)
        est_min_dist = np.min(est_dists)
        
        errors.append(abs(true_min_dist - est_min_dist))
    return np.mean(errors) if errors else np.nan

# НОВАЯ МЕТРИКА: Prediction RMSE
def compute_prediction_rmse(predictions, dp, episode_id, current_window_end_time, prediction_horizon):
    """
    Сравнивает предсказанные траектории пешеходов с их реальными 
    координатами строго в будущем (начиная со следующего кадра после конца окна).
    
    :param current_window_end_time: Время (t) последнего кадра в текущем окне.
    """
    if not predictions:
        return np.nan
        
    # Достаем все данные этого эпизода
    ep_df = dp.df[dp.df['episode'] == episode_id]
    
    # Ищем кадры, которые идут СТРОГО ПОСЛЕ current_window_end_time
    times = np.sort(ep_df['time'].unique())
    future_times = times[times > current_window_end_time][:prediction_horizon]
    
    if len(future_times) == 0:
        return np.nan
        
    sq_errors = []
    
    for lmid, pred_traj in predictions.items():
        pred_traj_np = pred_traj.cpu().numpy() # [Horizon, 2]
        
        for t_idx, f_time in enumerate(future_times):
            # Если предсказания закончились раньше, чем найденные кадры
            if t_idx >= len(pred_traj_np):
                break
                
            # Ищем координаты пешехода lmid в момент времени f_time
            ped_row = ep_df[(ep_df['time'] == f_time) & (ep_df['agent_id'] == lmid)]
            if not ped_row.empty:
                true_pos = ped_row[['x', 'y']].values[0]
                est_pos = pred_traj_np[t_idx]
                sq_errors.append(np.sum((true_pos - est_pos)**2))
                
    if not sq_errors:
        return np.nan
        
    return np.sqrt(np.mean(sq_errors))

# ==============================================================================
# ОТРИСОВКА
# ==============================================================================
def plot_window_results(window_data, dp, episode_id, current_window_end_time, est_robot_poses, est_landmarks, predictions, lm_ids, save_path="slam_result.png"):
    plt.figure(figsize=(12, 12))
    
    # 1. Истинная траектория робота
    true_robot = np.array([step[:2] for step in window_data['true_trajectory']])
    plt.plot(true_robot[:, 0], true_robot[:, 1], 'b-', linewidth=2, label='True Robot Path (Past/Current)', alpha=0.6)
    plt.scatter(true_robot[0, 0], true_robot[0, 1], c='blue', marker='o', s=100)
    
    # 2. Оцененная траектория робота
    est_robot = est_robot_poses.cpu().numpy()[:, :2]
    plt.plot(est_robot[:, 0], est_robot[:, 1], 'r--', linewidth=2, label='Estimated Robot Path')
    
    # 3. Истинные траектории пешеходов (Текущее окно)
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
            plt.scatter(path[-1, 0], path[-1, 1], c='green', marker='s') # Отмечаем конец текущего окна
            
    plt.plot([], [], 'g-', linewidth=2, label='True Pedestrians (Past/Current)')
    
    # 4. Оцененные траектории пешеходов
    est_lms = est_landmarks.cpu().numpy()
    for idx, lmid in enumerate(lm_ids):
        path = est_lms[:, idx, :]
        plt.plot(path[:, 0], path[:, 1], color='orange', linestyle='--', linewidth=2)
    plt.plot([], [], color='orange', linestyle='--', linewidth=2, label='Estimated Pedestrians')

    # 5. ПРЕДСКАЗАНИЯ (Будущее)
    prediction_horizon = 10
    if predictions:
        for lmid, pred_tensor in predictions.items():
            pred_path = pred_tensor.cpu().numpy()
            idx = lm_ids.index(lmid)
            last_est = est_lms[-1, idx, :]
            
            # Соединяем последнюю оцененную точку с предсказанием
            full_pred = np.vstack([last_est, pred_path])
            prediction_horizon = len(pred_path)
            
            plt.plot(full_pred[:, 0], full_pred[:, 1], color='purple', linestyle=':', linewidth=2)
            plt.scatter(full_pred[-1, 0], full_pred[-1, 1], c='purple', marker='x')
        plt.plot([], [], color='purple', linestyle=':', linewidth=2, label='Predictions (Future)')

    # 6. ИСТИННОЕ БУДУЩЕЕ (Ground Truth Future - Честно извлекаем из датасета)
    ep_df = dp.df[dp.df['episode'] == episode_id]
    times = np.sort(ep_df['time'].unique())
    future_times = times[times > current_window_end_time][:prediction_horizon]
    
    if len(future_times) > 0:
        for lmid in lm_ids:
            future_path = []
            for f_time in future_times:
                ped_row = ep_df[(ep_df['time'] == f_time) & (ep_df['agent_id'] == lmid)]
                if not ped_row.empty:
                    future_path.append(ped_row[['x', 'y']].values[0])
                    
            if len(future_path) > 0:
                future_path = np.array(future_path)
                # Соединяем конец истинной траектории окна с началом истинного будущего, чтобы не было разрыва
                if len(true_ped_paths[lmid]) > 0:
                    last_known_gt = true_ped_paths[lmid][-1]
                    future_path = np.vstack([last_known_gt, future_path])
                
                plt.plot(future_path[:, 0], future_path[:, 1], color='cyan', linestyle='-.', linewidth=2, alpha=0.8)
                plt.scatter(future_path[-1, 0], future_path[-1, 1], c='cyan', marker='*')
                
        plt.plot([], [], color='cyan', linestyle='-.', linewidth=2, label='Ground Truth Future')

    plt.title(f"Dynamic GraphSLAM: W={W}, Predict Horizon={prediction_horizon}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# ОЦЕНКА ЭПИЗОДА И ДАТАСЕТА
# ==============================================================================
def evaluate_episode(slam_model, dp, episode_id, num_epochs=100, plot_window_idx=None, prediction_horizon=10, save_prefix="slam"):
    windows = dp.get_episode_windows(episode_id)
    if not windows:
        return [], [], [], [], [], [], [], [] 

    robot_rmse_list, lm_rmse_list, ate_list = [], [], []
    rpe_mean_list, rpe_std_list, sde_list = [], [], []
    time_list, pred_rmse_list = [], []

    try:
        device = next(slam_model.parameters()).device
    except:
        device = "cpu"
        
    for i, w in enumerate(windows):
        init_pose = torch.tensor(w['init_robot_pose'], dtype=torch.float32).to(device)
        odom = torch.tensor(w['odometry'], dtype=torch.float32).to(device)
        
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
            init_pose, odom, observations, lm_history=lm_history, num_epochs=num_epochs, prediction_horizon=prediction_horizon
        )
        elapsed_ms = (time.time() - start_time) * 1000
        time_list.append(elapsed_ms)

        true_robot = np.array([step[:2] for step in w['true_trajectory']])
        est_robot = robot_poses.cpu().detach().numpy()[:, :2]

        r_rmse = np.sqrt(np.mean(np.sum((true_robot - est_robot)**2, axis=1)))
        robot_rmse_list.append(r_rmse)

        ate = compute_ate(est_robot, true_robot)
        ate_list.append(ate)

        rpe_mean, rpe_std = compute_rpe_trans(est_robot, true_robot, interval_frames=1)
        rpe_mean_list.append(rpe_mean)
        rpe_std_list.append(rpe_std)

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
                    est_pos = landmarks[t_idx, lm_idx].cpu().detach().numpy()
                    dist_sq = np.sum((true_pos - est_pos)**2)
                    lm_sq_errors.append(dist_sq)
                    
        if lm_sq_errors:
            l_rmse = np.sqrt(np.mean(lm_sq_errors))
            lm_rmse_list.append(l_rmse)
        else:
            lm_rmse_list.append(np.nan)
        
        T = len(w['true_landmarks'])
        M = len(lm_ids)
        ped_est = np.full((T, M, 2), np.nan)
        for t_idx in range(T):
            for lm_idx, lmid in enumerate(lm_ids):
                if not np.isnan(landmarks[t_idx, lm_idx, 0].cpu().detach()): 
                    ped_est[t_idx, lm_idx] = landmarks[t_idx, lm_idx].cpu().detach().numpy()
        sde = compute_sde(est_robot, true_robot, ped_est, w['true_landmarks'], T)
        sde_list.append(sde)

        # ----------------------------------------------------------------------
        # ОЦЕНКА БУДУЩЕГО И ОТРИСОВКА (Используя твой end_time)
        # ----------------------------------------------------------------------
        current_window_end_time = w.get('end_time', None)
        
        if current_window_end_time is not None:
            # Считаем RMSE предсказания
            pred_rmse = compute_prediction_rmse(predictions, dp, episode_id, current_window_end_time, prediction_horizon)
            pred_rmse_list.append(pred_rmse)

            # РИСУЕМ ГРАФИК, ЕСЛИ ИНДЕКС СОВПАДАЕТ
            if plot_window_idx is not None and i == plot_window_idx:
                plot_window_results(
                    window_data=w, 
                    dp=dp,
                    episode_id=episode_id,
                    current_window_end_time=current_window_end_time, 
                    est_robot_poses=robot_poses, 
                    est_landmarks=landmarks, 
                    predictions=predictions, 
                    lm_ids=lm_ids, 
                    save_path=f"{save_prefix}_ep{episode_id}_win{i}.png"
                )
                print(f"✅ Успех: График сохранен в файл {save_prefix}_ep{episode_id}_win{i}.png")
        else:
            pred_rmse_list.append(np.nan)
            if plot_window_idx is not None and i == plot_window_idx:
                print(f"❌ Ошибка: В окне {i} нет 'end_time', график пропущен!")

    return (robot_rmse_list, lm_rmse_list, ate_list, rpe_mean_list, rpe_std_list, sde_list, time_list, pred_rmse_list)

def evaluate_dataset(slam_model, dp, num_epochs=100, output_csv="evaluation_metrics.csv"):
    unique_episodes = dp.df['episode'].unique()
    print(f"--- Запуск оценки всего датасета ({len(unique_episodes)} эпизодов) ---")
    
    all_metrics = {k: [] for k in ['robot', 'lm', 'ate', 'rpe_m', 'rpe_s', 'sde', 'time', 'pred']}
    
    for ep_id in tqdm(unique_episodes, desc="Оценка эпизодов", unit="ep"):
        res = evaluate_episode(slam_model, dp, ep_id, num_epochs, plot_window_idx=None)
        
        for k, v_list in zip(all_metrics.keys(), res):
            all_metrics[k].extend(v_list)
    
    def stats(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan, np.nan, np.nan, np.nan
        return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

    metrics_names = ['robot_rmse', 'landmark_rmse', 'ate', 'rpe_mean', 'rpe_std', 'sde', 'time_ms', 'prediction_rmse']
    
    metrics = {
        'metric': metrics_names,
        'count': [len(all_metrics[k]) for k in all_metrics.keys()],
        'mean': [stats(all_metrics[k])[0] for k in all_metrics.keys()],
        'std':  [stats(all_metrics[k])[1] for k in all_metrics.keys()],
        'min':  [stats(all_metrics[k])[2] for k in all_metrics.keys()],
        'max':  [stats(all_metrics[k])[3] for k in all_metrics.keys()],
    }
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_csv, index=False)
    print(f"\nГлобальные метрики сохранены в {output_csv}")
    print(metrics_df.to_string())
    
    # Возвращаем ATE и SDE для обратной совместимости, а также Prediction RMSE
    return (
        metrics_df.loc[metrics_df['metric']=='ate', 'mean'].values[0], 
        metrics_df.loc[metrics_df['metric']=='sde', 'mean'].values[0],
        metrics_df.loc[metrics_df['metric']=='prediction_rmse', 'mean'].values[0]
    )
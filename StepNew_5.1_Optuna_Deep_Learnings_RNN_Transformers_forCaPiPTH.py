# ==============================================================================
# Section 0: 匯入所有必要的函式庫
# ==============================================================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import math
import optuna

# (此處程式碼與前一版本相同)
print(f"TensorFlow 版本: {tf.__version__}")
print(f"Optuna 版本: {optuna.__version__}")
print(f"目前時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================================
# Section 1: 設定 Configuration
# ==============================================================================
# --- 檔案與數據設定 ---
home_dir = os.path.expanduser('~')
folder_path = os.path.join(home_dir, "TEST3_CKD")
FILE_PATH = os.path.join(folder_path, "Lab_and_Med_Imputed_with_training_split.xlsx")
main_output_dir = os.path.join(folder_path, "output_results_optuna", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(main_output_dir, exist_ok=True)
print(f"所有 Optuna 結果將儲存於: {main_output_dir}")

# --- ★★★★★ 新增：模型名稱與超參數存檔路徑設定 ★★★★★ ---
MODEL_NAME = 'Transformer_CaPiPTH' # 用於在 CSV 中標識此模型的名稱
BEST_HYPERPARAMS_CSV_PATH = 'C:/TEST3_CKD/Hyperparameters_CaPiPTH_bestMarkers.csv'
print(f"找到的最佳超參數將會被儲存或更新至: {BEST_HYPERPARAMS_CSV_PATH}")


n_trials=5
n_repeat=1

TARGET_COLUMNS = ['Ca', 'P', 'iPTH']
SEQUENCE_LENGTH = 10

# --- 指標與損失函數設定 ---
OPTIMIZATION_METRIC_NAME = 'MSE'
TRAINING_LOSS_NAME = 'MSE'
LOSS_FUNCTION_MAP = {'MSE': 'mean_squared_error', 'MAE': 'mean_absolute_error'}
if TRAINING_LOSS_NAME not in LOSS_FUNCTION_MAP:
    raise ValueError(f"不支援的訓練損失函數: '{TRAINING_LOSS_NAME}'.")
optuna_direction = 'maximize' if OPTIMIZATION_METRIC_NAME == 'R2' else 'minimize'

print(f"Optuna 優化目標: {optuna_direction} '{OPTIMIZATION_METRIC_NAME}'")
print(f"模型訓練損失函數: '{TRAINING_LOSS_NAME}'")

# --- 訓練超參數 (固定部分) ---
EPOCHS = 200
PATIENCE = 10

# (Section 2, 3, 4, 5 的程式碼與前一版本完全相同，此處為簡潔省略)
# ... (載入數據、準備 Scaler、定義模型架構、定義全域輔助函數) ...
# ==============================================================================
# Section 2: 載入數據與特徵工程
# ==============================================================================
print("\n--- Section 2: 載入數據與特徵工程 ---")
try:
    df = pd.read_excel(FILE_PATH)
except FileNotFoundError: print(f"錯誤：找不到檔案 '{FILE_PATH}'。"); exit()
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values(by=['ID', 'DATE']).reset_index(drop=True)
start_date = df['DATE'].min()
df['time_stamp'] = (df['DATE'] - start_date).dt.days
feature_columns = [col for col in df.columns if col not in ['ID', 'DATE', 'training']]
num_features = len(feature_columns)
df_train = df[df['training'] == 1].copy(); df_val = df[df['training'] == 2].copy(); df_test = df[df['training'] == 3].copy()
print("數據載入與分割完成。")

# ==============================================================================
# Section 3: 數據生成器與 Scaler
# ==============================================================================
print("\n--- Section 3: 準備數據生成器與 Scaler ---")
def create_data_generator(data, sequence_length, target_columns):
    feature_columns_gen = [col for col in data.columns if col not in ['ID', 'DATE', 'training']]
    min_len = (sequence_length // 2) + 1
    for _, group in data.groupby('ID'):
        if len(group) >= min_len:
            for i in range(min_len - 1, len(group)):
                target_index = i
                target_values = group[target_columns].iloc[target_index].values.astype(np.float32)
                history_start_index = max(0, target_index - sequence_length)
                seq_df = group.iloc[history_start_index:target_index]
                actual_seq_len = len(seq_df)
                padding_size = sequence_length - actual_seq_len
                padded_seq = np.pad(seq_df[feature_columns_gen].values, pad_width=((padding_size, 0), (0, 0)), mode='constant', constant_values=0).astype(np.float32)
                padded_ts = np.pad(seq_df['time_stamp'].values, pad_width=(padding_size, 0), mode='constant', constant_values=0).astype(np.float32)
                mask = np.concatenate([np.zeros(padding_size), np.ones(actual_seq_len)]).astype(np.float32)
                if actual_seq_len > 0:
                    last_input_timestamp = group['time_stamp'].iloc[target_index - 1]
                    target_timestamp = group['time_stamp'].iloc[target_index]
                    future_delta = np.array([target_timestamp - last_input_timestamp], dtype=np.float32)
                else:
                    future_delta = np.array([0], dtype=np.float32)
                inputs = {'features': padded_seq, 'timestamps': padded_ts, 'future_delta': future_delta, 'attention_mask': mask}
                yield inputs, target_values
x_scaler, y_scaler, fd_scaler = StandardScaler(), StandardScaler(), StandardScaler()
print("正在準備 Scaler (僅使用訓練數據)...")
temp_features, temp_targets, temp_deltas = [], [], []
for inputs, target in create_data_generator(df_train, SEQUENCE_LENGTH, TARGET_COLUMNS):
    real_features = inputs['features'][inputs['attention_mask'] == 1]
    if real_features.shape[0] > 0: temp_features.extend(real_features)
    temp_targets.append(target); temp_deltas.append(inputs['future_delta'])
x_scaler.fit(temp_features); y_scaler.fit(temp_targets); fd_scaler.fit(temp_deltas)
del temp_features, temp_targets, temp_deltas
print("Scaler 準備完成。")

# ==============================================================================
# Section 4: 模型定義
# ==============================================================================
print("\n--- Section 4: 模型架構定義 ---")
class CTLPE(layers.Layer):
    def __init__(self, d_model, **kwargs): super().__init__(**kwargs); self.d_model = d_model; self.time_embedder = layers.Dense(d_model, name='time_embedder')
    def call(self, inputs): feature_embedding, timestamps = inputs; time_deltas = timestamps - timestamps[:, 0:1]; time_embedding = self.time_embedder(tf.expand_dims(time_deltas, -1)); return feature_embedding + time_embedding
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, nhead, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs); self.att = layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model); self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model),]); self.layernorm1 = layers.LayerNormalization(epsilon=1e-6); self.layernorm2 = layers.LayerNormalization(epsilon=1e-6); self.dropout1 = layers.Dropout(dropout_rate); self.dropout2 = layers.Dropout(dropout_rate)
    def call(self, inputs, mask=None, training=False):
        final_mask = None
        if mask is not None: bool_mask = tf.cast(mask, tf.bool); final_mask = bool_mask[:, tf.newaxis, tf.newaxis, :]
        attn_output = self.att(inputs, inputs, attention_mask=final_mask); attn_output = self.dropout1(attn_output, training=training); out1 = self.layernorm1(inputs + attn_output); ffn_output = self.ffn(out1); ffn_output = self.dropout2(ffn_output, training=training); return self.layernorm2(out1 + ffn_output)
def create_transformer_model(input_shape, ts_shape, fd_shape, mask_shape, output_dim, d_model, nhead, ff_dim, nlayers, dropout=0.1):
    feature_input = keras.Input(shape=input_shape, name='features'); timestamp_input = keras.Input(shape=ts_shape, name='timestamps'); future_delta_input = keras.Input(shape=fd_shape, name='future_delta'); mask_input = keras.Input(shape=mask_shape, dtype=tf.float32, name='attention_mask')
    x = layers.Dense(d_model, name='feature_encoder')(feature_input); x = CTLPE(d_model, name='ctlpe_layer')([x, timestamp_input])
    for _ in range(nlayers): x = TransformerBlock(d_model, nhead, ff_dim, dropout)(x, mask=mask_input)
    context_vector = layers.GlobalAveragePooling1D(data_format="channels_first")(x); combined_input = layers.Concatenate()([context_vector, future_delta_input]); z = layers.Dropout(dropout)(combined_input); z = layers.Dense(d_model // 2, activation='relu')(z); output = layers.Dense(output_dim, name='output_regressor')(z)
    return keras.Model(inputs=[feature_input, timestamp_input, future_delta_input, mask_input], outputs=output)

# ==============================================================================
# Section 5: 全域輔助函數 (Dataset, Metrics)
# ==============================================================================
print("\n--- Section 5: 準備全域輔助函數 ---")
output_signature = (
    {'features': tf.TensorSpec(shape=(SEQUENCE_LENGTH, num_features), dtype=tf.float32), 'timestamps': tf.TensorSpec(shape=(SEQUENCE_LENGTH,), dtype=tf.float32), 'future_delta': tf.TensorSpec(shape=(1,), dtype=tf.float32), 'attention_mask': tf.TensorSpec(shape=(SEQUENCE_LENGTH,), dtype=tf.float32)},
    tf.TensorSpec(shape=(len(TARGET_COLUMNS),), dtype=tf.float32)
)
def scale_and_prepare_tf(inputs, targets):
    def scale_py_fn(features, mask):
        features_val = features.numpy(); scaled_features = np.zeros_like(features_val, dtype=np.float32); non_pad_indices = mask.numpy() == 1
        if np.any(non_pad_indices): real_data = features_val[non_pad_indices]; scaled_data = x_scaler.transform(real_data); scaled_features[non_pad_indices] = scaled_data
        return scaled_features
    scaled_features = tf.py_function(func=scale_py_fn, inp=[inputs['features'], inputs['attention_mask']], Tout=tf.float32); scaled_features.set_shape(inputs['features'].get_shape())
    scaled_delta = tf.py_function(lambda x: fd_scaler.transform(x.numpy().reshape(1, -1)).flatten(), [inputs['future_delta']], tf.float32); scaled_delta.set_shape(inputs['future_delta'].get_shape())
    scaled_targets = tf.py_function(lambda y: y_scaler.transform(y.numpy().reshape(1, -1)).flatten(), [targets], tf.float32); scaled_targets.set_shape(targets.get_shape())
    final_inputs = {'features': scaled_features, 'timestamps': inputs['timestamps'], 'future_delta': scaled_delta, 'attention_mask': inputs['attention_mask']}
    return final_inputs, scaled_targets
def count_samples(data, sequence_length):
    count = 0; min_len = (sequence_length // 2) + 1
    for _, group in data.groupby('ID'):
        if len(group) >= min_len: count += (len(group) - (min_len - 1))
    return count
def calculate_metrics_on_original_scale(y_true_orig, y_pred_orig):
    non_zero_mask = y_true_orig != 0
    mse = mean_squared_error(y_true_orig, y_pred_orig); rmse = np.sqrt(mse); mae = mean_absolute_error(y_true_orig, y_pred_orig); r2 = r2_score(y_true_orig, y_pred_orig)
    mape = np.mean(np.abs((y_true_orig[non_zero_mask] - y_pred_orig[non_zero_mask]) / y_true_orig[non_zero_mask])) * 100 if np.sum(non_zero_mask) > 0 else np.inf
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

# ==============================================================================
# Section 6: Optuna Objective Function
# ==============================================================================
print("\n--- Section 6: 定義 Optuna Objective Function ---")

# ★★★★★ 新增：將超參數建議邏輯獨立成函數 ★★★★★
def suggest_transformer_hyperparameters(trial: optuna.trial.Trial) -> dict:
    """根據 Optuna trial 物件建議一組超參數。"""
    params = {}
    params['d_model'] = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead_options = [h for h in [2, 4, 8] if params['d_model'] % h == 0]
    if not nhead_options:
        raise optuna.exceptions.TrialPruned(f"d_model={params['d_model']} is not divisible by any nhead options.")
    params['nhead'] = trial.suggest_categorical('nhead', nhead_options)
    params['nlayers'] = trial.suggest_int('nlayers', 1, 4)
    params['d_ff'] = trial.suggest_categorical('d_ff', [params['d_model'] * 2, params['d_model'] * 4])
    params['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    return params

def objective(trial):
    # --- 1. 建議超參數 ---
    # ★★★★★ 修改處：呼叫獨立的建議函數 ★★★★★
    hyperparams = suggest_transformer_hyperparameters(trial)
    batch_size = hyperparams['batch_size']

    keras.backend.clear_session()

    # --- 2. 建立資料管線 ---
    steps_per_epoch = math.ceil(count_samples(df_train, SEQUENCE_LENGTH) / batch_size)
    validation_steps = math.ceil(count_samples(df_val, SEQUENCE_LENGTH) / batch_size)
    train_dataset = tf.data.Dataset.from_generator(lambda: create_data_generator(df_train, SEQUENCE_LENGTH, TARGET_COLUMNS), output_signature=output_signature).map(scale_and_prepare_tf, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()
    val_dataset_for_fit = tf.data.Dataset.from_generator(lambda: create_data_generator(df_val, SEQUENCE_LENGTH, TARGET_COLUMNS), output_signature=output_signature).map(scale_and_prepare_tf, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()
    val_dataset_for_eval = tf.data.Dataset.from_generator(lambda: create_data_generator(df_val, SEQUENCE_LENGTH, TARGET_COLUMNS), output_signature=output_signature).map(scale_and_prepare_tf, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # --- 3. 建立與編譯模型 ---
    # ★★★★★ 修改處：使用解包語法傳入參數 ★★★★★
    model = create_transformer_model(
        (SEQUENCE_LENGTH, num_features), (SEQUENCE_LENGTH,), (1,), (SEQUENCE_LENGTH,), len(TARGET_COLUMNS),
        d_model=hyperparams['d_model'], nhead=hyperparams['nhead'], ff_dim=hyperparams['d_ff'],
        nlayers=hyperparams['nlayers'], dropout=hyperparams['dropout']
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=hyperparams['learning_rate']),
        loss=LOSS_FUNCTION_MAP[TRAINING_LOSS_NAME]
    )

    # --- 4. 訓練模型 ---
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
    model.fit(
        train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset_for_fit, validation_steps=validation_steps,
        callbacks=[early_stopping, pruning_callback], verbose=0
    )

    # --- 5. 在驗證集上計算 Optuna 的目標指標 ---
    y_true_scaled = np.concatenate([y for x, y in val_dataset_for_eval], axis=0)
    predictions_scaled = model.predict(val_dataset_for_eval, verbose=0)
    y_true_original = y_scaler.inverse_transform(y_true_scaled)
    predictions_original = y_scaler.inverse_transform(predictions_scaled)
    metrics = calculate_metrics_on_original_scale(y_true_original, predictions_original)
    objective_value = metrics[OPTIMIZATION_METRIC_NAME]
    
    return objective_value

# ★★★★★ 新增：儲存超參數的專用函數 ★★★★★
def save_best_hyperparameters_to_csv(params: dict, score: float, model_name: str, metric_name: str, file_path: str):
    """
    將找到的最佳超參數智慧地儲存到 CSV 檔案中。
    如果檔案或模型記錄已存在，則更新；否則，建立新檔案或附加新紀錄。
    """
    # 建立一個包含所有資訊的新紀錄
    new_record = {'model_name': model_name, 'optimized_metric': metric_name, 'score': score, **params}
    
    try:
        # 嘗試讀取現有的 CSV 檔案
        df_existing = pd.read_csv(file_path)
        
        # 檢查模型是否已經存在
        if model_name in df_existing['model_name'].values:
            print(f"在 '{file_path}' 中找到現有模型 '{model_name}' 的記錄，將進行更新...")
            # 找到該模型的索引
            idx = df_existing.index[df_existing['model_name'] == model_name][0]
            # 更新該行的值
            for key, value in new_record.items():
                df_existing.loc[idx, key] = value
            df_to_save = df_existing
        else:
            print(f"在 '{file_path}' 中未找到模型 '{model_name}' 的記錄，將附加新紀錄...")
            # 將新紀錄轉換為 DataFrame 並附加
            df_new_record = pd.DataFrame([new_record])
            df_to_save = pd.concat([df_existing, df_new_record], ignore_index=True)
            
    except FileNotFoundError:
        # 如果檔案不存在，則建立一個新的 DataFrame
        print(f"檔案 '{file_path}' 不存在，將建立新檔案...")
        df_to_save = pd.DataFrame([new_record])

    # 確保目錄存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 儲存 DataFrame 到 CSV
    df_to_save.to_csv(file_path, index=False)
    print(f"最佳超參數已成功儲存至 '{file_path}'")


# ==============================================================================
# Section 7: 執行 Optuna 超參數搜尋
# ==============================================================================
if __name__ == "__main__":
    print(f"\n--- Section 7: 開始執行 Optuna 超參數搜尋 (目標: {optuna_direction} {OPTIMIZATION_METRIC_NAME}) ---")
    for i in range(n_repeat):
        study_Transformer = optuna.create_study(direction=optuna_direction, pruner=optuna.pruners.MedianPruner())
        study_Transformer.optimize(objective, n_trials=n_trials, timeout=10800)
  
        print("\n--- Optuna 搜尋完畢 ---")
        print('This is repeat(round)', i+1)
        print(f"總共完成的 Trial 數量: {len(study_Transformer.trials)}")

        best_trial = study_Transformer.best_trial
        print("最佳 Trial:")
        print(f"  - 數值 (Best Validation {OPTIMIZATION_METRIC_NAME}): {best_trial.value:.6f}")

        print("  - 最佳超參數:")
        best_params = best_trial.params
        for key, value in best_params.items():
            print(f"    - {key}: {value}")

    study_results_df = study_Transformer.trials_dataframe()
    study_results_path = os.path.join(main_output_dir, "optuna_study_results.csv")
    study_results_df.to_csv(study_results_path, index=False)
    print(f"\nOptuna 所有 trial 的詳細結果已儲存至: {study_results_path}")

    # ★★★★★ 修改處：呼叫存檔函數 ★★★★★
    save_best_hyperparameters_to_csv(
        params=best_params,
        score=best_trial.value,
        model_name=MODEL_NAME,
        metric_name=OPTIMIZATION_METRIC_NAME,
        file_path=BEST_HYPERPARAMS_CSV_PATH
    )

    # ==============================================================================
    # Section 8: 使用最佳參數重新訓練與評估最終模型
    # ==============================================================================
    print("\n--- Section 8: 使用找到的最佳參數重新訓練最終模型 ---")
    
    # ... (此區塊與前一版本完全相同，使用 best_params 訓練和評估) ...
    keras.backend.clear_session()
    final_batch_size = best_params['batch_size']
    final_steps_per_epoch = math.ceil(count_samples(df_train, SEQUENCE_LENGTH) / final_batch_size)
    final_validation_steps = math.ceil(count_samples(df_val, SEQUENCE_LENGTH) / final_batch_size)
    train_dataset_final = tf.data.Dataset.from_generator(lambda: create_data_generator(df_train, SEQUENCE_LENGTH, TARGET_COLUMNS), output_signature=output_signature).map(scale_and_prepare_tf, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(final_batch_size).prefetch(tf.data.AUTOTUNE).repeat()
    val_dataset_final = tf.data.Dataset.from_generator(lambda: create_data_generator(df_val, SEQUENCE_LENGTH, TARGET_COLUMNS), output_signature=output_signature).map(scale_and_prepare_tf, num_parallel_calls=tf.data.AUTOTUNE).batch(final_batch_size).prefetch(tf.data.AUTOTUNE).repeat()

    final_model = create_transformer_model(
        (SEQUENCE_LENGTH, num_features), (SEQUENCE_LENGTH,), (1,), (SEQUENCE_LENGTH,), len(TARGET_COLUMNS),
        **best_params
    )
    final_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=best_params['learning_rate']),
        loss=LOSS_FUNCTION_MAP[TRAINING_LOSS_NAME]
    )
    final_model.summary()
    
    log_dir = os.path.join(main_output_dir, "logs", "final_fit")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_final = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    print(f"\n開始訓練最終模型 (訓練損失: {TRAINING_LOSS_NAME})...")
    history_final = final_model.fit(
        train_dataset_final, epochs=EPOCHS, steps_per_epoch=final_steps_per_epoch,
        validation_data=val_dataset_final, validation_steps=final_validation_steps,
        callbacks=[early_stopping_final, tensorboard_callback], verbose=2
    )
    model_save_path = os.path.join(main_output_dir, "final_model.keras")
    final_model.save(model_save_path)
    print(f"最終模型已儲存至: {model_save_path}")
    
    # ==============================================================================
    # Section 9: 在測試集上評估並視覺化最終模型
    # ==============================================================================
    # (此區塊與前一版本完全相同)
    print("\n--- Section 9: 在測試集上評估與視覺化最終模型 ---")
    if not df_test.empty:
        X_test_list, y_test_list = [], []
        for inputs, targets in create_data_generator(df_test, SEQUENCE_LENGTH, TARGET_COLUMNS):
            features_val = inputs['features']; mask_val = inputs['attention_mask']; scaled_features = np.zeros_like(features_val, dtype=np.float32); non_pad_indices = mask_val == 1
            if np.any(non_pad_indices): real_data = features_val[non_pad_indices]; scaled_data = x_scaler.transform(real_data); scaled_features[non_pad_indices] = scaled_data
            scaled_delta = fd_scaler.transform(inputs['future_delta'].reshape(1, -1)).flatten(); scaled_inputs = {'features': scaled_features, 'timestamps': inputs['timestamps'], 'future_delta': scaled_delta, 'attention_mask': inputs['attention_mask']}
            X_test_list.append(scaled_inputs); y_test_list.append(targets)
        if X_test_list: X_test_dict = {key: np.array([d[key] for d in X_test_list]) for key in X_test_list[0]}; y_test = np.array(y_test_list)
        else: X_test_dict, y_test = None, None
    else: X_test_dict, y_test = None, None

    if y_test is not None and len(y_test) > 0:
        predictions_scaled = final_model.predict(X_test_dict)
        predictions_original = y_scaler.inverse_transform(predictions_scaled)
        y_test_original = y_test
        results_data = []
        for i, name in enumerate(TARGET_COLUMNS):
            print(f"\n--- 測試集評估 - 目標: {name} ---")
            metrics = calculate_metrics_on_original_scale(y_test_original[:, i], predictions_original[:, i])
            for key, value in metrics.items(): print(f"  {key:<5}: {value:.4f}")
            result_row = {'Target': name, **{k: round(v, 4) for k, v in metrics.items()}}; results_data.append(result_row)
        df_results = pd.DataFrame(results_data)
        results_path_xlsx = os.path.join(main_output_dir, "final_evaluation_metrics_per_target.xlsx")
        df_results.to_excel(results_path_xlsx, index=False)
        print(f"\n最終評估指標 (分目標) 已儲存為 Excel 檔案: {results_path_xlsx}")
        
        # ... 視覺化 ...
        plt.figure(figsize=(10, 6)); plt.plot(history_final.history['loss'], label='Training Loss'); plt.plot(history_final.history['val_loss'], label='Validation Loss'); plt.axvline(x=early_stopping_final.best_epoch, color='r', linestyle='--', label=f'Best Epoch ({early_stopping_final.best_epoch+1})'); plt.title('Final Model - Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel(f"Loss ({TRAINING_LOSS_NAME})"); plt.legend(); plt.grid(True)
        loss_curve_path = os.path.join(main_output_dir, 'final_loss_curve.png'); plt.savefig(loss_curve_path, dpi=300); plt.close()
        fig_test, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(18, 5.5), squeeze=False); axes = axes.flatten()
        for i, name in enumerate(TARGET_COLUMNS):
            ax = axes[i]; ax.scatter(y_test_original[:, i], predictions_original[:, i], alpha=0.6, s=50, edgecolors='k', linewidths=0.5); lims = [min(np.min(y_test_original[:, i]), np.min(predictions_original[:, i])), max(np.max(y_test_original[:, i]), np.max(predictions_original[:, i]))]; ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x'); ax.set_xlabel("True Values"); ax.set_ylabel("Predicted Values"); ax.set_title(f"Test Set: Prediction vs. True ({name})"); ax.grid(True); ax.legend(); ax.set_aspect('equal', adjustable='box')
        plt.tight_layout(pad=2.0); predictions_plot_path = os.path.join(main_output_dir, 'final_predictions_vs_true.png'); plt.savefig(predictions_plot_path, dpi=300); plt.close(fig_test)
        print(f"損失曲線圖與預測散點圖已儲存至: {main_output_dir}")
    else:
        print("測試集為空，跳過最終評估。")

    print("\n--- 程式執行完畢 ---")





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
INPUT_FILE = "./mcm26Train-B-Data_clean/task1_traffic_flow_5min.csv"
BASE_OUT_DIR = "./task1-The_Crystal_Ball/output/model_B_SARIMA"
IMG_DIR = os.path.join(BASE_OUT_DIR, "images")
RES_DIR = os.path.join(BASE_OUT_DIR, "results")
FILE_PREFIX = "model_B_SARIMA"
plt.style.use('bmh') 
# ===========================================

def create_fourier_features(df, period, order):
    t = np.arange(len(df))
    k = 2 * np.pi * t / period
    for i in range(1, order + 1):
        df[f'sin_{period}_{i}'] = np.sin(i * k)
        df[f'cos_{period}_{i}'] = np.cos(i * k)
    return df

def run_sarima_enhanced():
    print(f"🚀 启动方案 B (SARIMAX + Fourier) 增强可视化版...")
    
    # 1. 创建目录
    for directory in [IMG_DIR, RES_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 2. 加载数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 3. 特征工程
    print("🛠️ 正在构建傅里叶特征...")
    df = create_fourier_features(df, period=288, order=10)   # 日周期
    df = create_fourier_features(df, period=288*7, order=5)  # 周周期
    
    exog_cols = [c for c in df.columns if 'sin_' in c or 'cos_' in c]
    
    # 4. 划分训练集与测试集
    test_days = 5
    cutoff_date = df['ds'].max() - pd.Timedelta(days=test_days)

    train_df = df[df['ds'] <= cutoff_date]
    test_df = df[df['ds'] > cutoff_date]

    y_train = train_df['y']
    X_train = train_df[exog_cols]
    
    print(f"   训练集截止: {cutoff_date}")
    
    # 5. 训练模型
    print("⏳ 正在训练 SARIMAX 模型 (约 1-2 分钟)...")
    model = SARIMAX(endog=y_train, 
                    exog=X_train, 
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    print("   ✅ 模型训练完成")

    # 6. 预测
    print("🔮 正在进行全量预测...")
    # 只需传入 out-of-sample 的 exog，模型会自动处理 in-sample
    full_pred = model_fit.get_prediction(start=0, end=len(df)-1, exog=test_df[exog_cols])
    
    predicted_mean = full_pred.predicted_mean
    conf_int = full_pred.conf_int()

    # 整理结果
    result_df = df[['ds', 'y']].copy()
    result_df['yhat'] = predicted_mean.values
    result_df['yhat_lower'] = conf_int.iloc[:, 0].values
    result_df['yhat_upper'] = conf_int.iloc[:, 1].values
    result_df['yhat_clean'] = result_df['yhat'].clip(lower=0)

    # 7. 评估
    test_res = result_df.iloc[-len(test_df):]
    mae = mean_absolute_error(test_res['y'], test_res['yhat_clean'])
    rmse = np.sqrt(mean_squared_error(test_res['y'], test_res['yhat_clean']))
    r2 = r2_score(test_res['y'], test_res['yhat_clean'])

    print("\n📊 === 方案 B (SARIMA) 评估结果 ===")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R2 Score: {r2:.4f}")

    # 8. 保存数据
    full_res_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_full_forecast.csv")
    result_df.to_csv(full_res_path, index=False)
    
    comp_df = test_res[['ds', 'y', 'yhat', 'yhat_clean']].copy()
    comp_df.rename(columns={'y': 'Actual', 'yhat': 'Predicted_Raw', 'yhat_clean': 'Predicted_Clean'}, inplace=True)
    comp_df['Error'] = comp_df['Actual'] - comp_df['Predicted_Clean']
    comp_res_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_test_comparison.csv")
    comp_df.to_csv(comp_res_path, index=False)
    print(f"   💾 数据已保存至: {RES_DIR}")

    # 9. 生成 5 张图表
    print("🎨 正在生成 5 张可视化图表...")
    def get_img_path(name):
        return os.path.join(IMG_DIR, f"{FILE_PREFIX}_{name}")

    # --- 图 1: 全局预测 ---
    plt.figure(figsize=(14, 6))
    plt.plot(result_df['ds'], result_df['y'], label='Actual', color='gray', alpha=0.5)
    plt.plot(result_df['ds'], result_df['yhat_clean'], label='SARIMAX Prediction', color='blue', alpha=0.7)
    plt.title(f'Global Forecast (Model B: SARIMAX) - RMSE: {rmse:.2f}')
    plt.legend()
    plt.savefig(get_img_path("1_global_forecast.png"), dpi=300)
    plt.close()

    # --- 图 2: 放大测试集 ---
    plt.figure(figsize=(14, 7))
    plt.plot(test_res['ds'], test_res['y'], label='Actual', color='gray', alpha=0.6, linewidth=1.5)
    plt.plot(test_res['ds'], test_res['yhat_clean'], label='Predicted', color='blue', linewidth=2)
    plt.fill_between(test_res['ds'], test_res['yhat_lower'], test_res['yhat_upper'], color='blue', alpha=0.1)
    plt.title('Validation Zoom-in (Model B: SARIMAX)', fontsize=14)
    plt.legend()
    plt.savefig(get_img_path("2_test_set_zoom_in.png"), dpi=300)
    plt.close()

    # --- 图 3: 拟合散点图 ---
    plt.figure(figsize=(8, 8))
    plt.scatter(test_res['y'], test_res['yhat_clean'], alpha=0.3, color='purple')
    max_val = max(test_res['y'].max(), test_res['yhat_clean'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.title('Actual vs Predicted (SARIMAX)', fontsize=14)
    plt.savefig(get_img_path("3_fit_scatter.png"), dpi=300)
    plt.close()

    # --- 图 4: 残差分布图 (新增) ---
    residuals = test_res['y'] - test_res['yhat_clean']
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='purple', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Error Distribution (Model B: SARIMAX)', fontsize=14)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig(get_img_path("4_error_distribution.png"), dpi=300)
    plt.close()

    # --- 图 5: 残差时间序列图 (新增) ---
    plt.figure(figsize=(14, 6))
    plt.plot(test_res['ds'], residuals, color='purple', alpha=0.8)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residuals over Time (Model B: SARIMAX)', fontsize=14)
    plt.ylabel('Error (Actual - Predicted)')
    plt.xlabel('Date')
    plt.savefig(get_img_path("5_residuals_over_time.png"), dpi=300)
    plt.close()

    print(f"   🖼️ 所有 5 张图表已保存至: {IMG_DIR}")
    print("\n✅ 任务一 (方案B - 增强版) 运行完毕！")

if __name__ == "__main__":
    run_sarima_enhanced()
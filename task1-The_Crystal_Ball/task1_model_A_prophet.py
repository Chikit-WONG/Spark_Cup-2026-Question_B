import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ================= 配置区域 (Configuration) =================
# 输入数据路径 (保持不变)
INPUT_FILE = "./mcm26Train-B-Data_clean/task1_traffic_flow_5min.csv"

# === 修改点: 最终确定的输出路径结构 ===
# 主任务目录 -> 输出 -> 模型A目录
BASE_OUT_DIR = "./task1-The_Crystal_Ball/output/model_A_prophet"
IMG_DIR = os.path.join(BASE_OUT_DIR, "images")
RES_DIR = os.path.join(BASE_OUT_DIR, "results")

# 文件名前缀 (用于区分后续的模型B和C)
FILE_PREFIX = "model_A_prophet"

# 绘图风格设置
plt.style.use('bmh') 
# ==========================================================

def run_prophet_final():
    print(f"🚀 启动方案 A (Prophet) 预测模型 [路径最终版]...")
    print(f"   📂 目标输出目录: {BASE_OUT_DIR}")
    
    # 1. 创建输出目录结构
    for directory in [IMG_DIR, RES_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ 已创建目录: {directory}")
    
    # 2. 加载数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        print("   请确保你已运行过数据预处理脚本。")
        return

    df = pd.read_csv(INPUT_FILE)
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 3. 划分训练集与测试集 (最后 5 天做验证)
    test_days = 5
    cutoff_date = df['ds'].max() - pd.Timedelta(days=test_days)

    train_df = df[df['ds'] <= cutoff_date]
    test_df = df[df['ds'] > cutoff_date]

    print(f"   训练集截止: {cutoff_date}")
    
    # 4. 训练模型
    print("⏳ 正在训练模型 (Training)...")
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(train_df)

    # 5. 预测
    future = model.make_future_dataframe(periods=len(test_df), freq='5T')
    forecast = model.predict(future)

    # === 数据修正 ===
    # 创建 clean 列，将负数修正为0
    forecast['yhat_clean'] = forecast['yhat'].clip(lower=0)

    # 6. 评估准备
    prediction_slice = forecast.iloc[-len(test_df):].copy()
    y_true = test_df['y'].values
    y_pred = prediction_slice['yhat_clean'].values # 使用修正后的值

    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📊 === 最终评估结果 ===")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R2 Score: {r2:.4f}")

    # ==========================================
    # 7. 保存结果数据 (CSV)
    # ==========================================
    
    # 保存完整的预测表
    full_res_filename = f"{FILE_PREFIX}_full_forecast.csv"
    full_res_path = os.path.join(RES_DIR, full_res_filename)
    # 保存关键列
    cols_to_save = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'yhat_clean', 'trend', 'daily', 'weekly']
    forecast[cols_to_save].to_csv(full_res_path, index=False)
    
    # 保存对比表
    comp_res_filename = f"{FILE_PREFIX}_test_comparison.csv"
    comp_res_path = os.path.join(RES_DIR, comp_res_filename)
    comparison_df = pd.DataFrame({
        'ds': test_df['ds'],
        'Actual': y_true,
        'Predicted_Raw': prediction_slice['yhat'],
        'Predicted_Clean': y_pred,
        'Error': y_true - y_pred
    })
    comparison_df.to_csv(comp_res_path, index=False)
    
    print(f"   💾 结果数据已保存至: {RES_DIR}")

    # ==========================================
    # 8. 生成增强可视化图片
    # ==========================================
    print("🎨 正在生成可视化图表...")

    # 辅助函数：生成带前缀的图片路径
    def get_img_path(name):
        return os.path.join(IMG_DIR, f"{FILE_PREFIX}_{name}")

    # --- 图 1: 全局预测概览 ---
    fig1 = model.plot(forecast)
    plt.title(f'Global Forecast (Model A: Prophet) - RMSE: {rmse:.2f}')
    plt.savefig(get_img_path("1_global_forecast.png"), dpi=300)
    plt.close(fig1)

    # --- 图 2: 成分分解 ---
    fig2 = model.plot_components(forecast)
    plt.savefig(get_img_path("2_model_components.png"), dpi=300)
    plt.close(fig2)

    # --- 图 3: 测试集细节放大 ---
    plt.figure(figsize=(14, 7))
    plt.plot(test_df['ds'], test_df['y'], label='Actual', color='gray', alpha=0.6)
    plt.plot(prediction_slice['ds'], prediction_slice['yhat_clean'], label='Predicted (Cleaned)', color='#d62728', linewidth=2)
    plt.fill_between(prediction_slice['ds'], prediction_slice['yhat_lower'], prediction_slice['yhat_upper'], color='#d62728', alpha=0.1)
    plt.title('Validation Zoom-in (Model A: Prophet)', fontsize=14)
    plt.legend()
    plt.savefig(get_img_path("3_test_set_zoom_in.png"), dpi=300)
    plt.close()

    # --- 图 4: 残差分布 ---
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Error Distribution (Model A: Prophet)', fontsize=14)
    plt.savefig(get_img_path("4_error_distribution.png"), dpi=300)
    plt.close()

    # --- 图 5: 拟合回归 ---
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue')
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.title('Actual vs Predicted (Model A: Prophet)', fontsize=14)
    plt.savefig(get_img_path("5_fit_scatter.png"), dpi=300)
    plt.close()

    print(f"   🖼️ 所有图表已保存至: {IMG_DIR}")
    print("\n✅ 任务一 (方案A - Prophet) 运行完毕！")

if __name__ == "__main__":
    run_prophet_final()
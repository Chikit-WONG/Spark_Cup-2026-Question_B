import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ================= 配置区域 =================
# 各模型结果文件的路径 (根据你之前的设定)
PATH_A = "./task1-The_Crystal_Ball/output/model_A_prophet/results/model_A_prophet_test_comparison.csv"
PATH_B = "./task1-The_Crystal_Ball/output/model_B_SARIMA/results/model_B_SARIMA_test_comparison.csv"
PATH_C = "./task1-The_Crystal_Ball/output/model_C_LSTM/results/model_C_LSTM_test_comparison.csv"

# 对比结果输出路径
OUT_DIR = "./task1-The_Crystal_Ball/output/comparison"
IMG_DIR = os.path.join(OUT_DIR, "images")
RES_DIR = os.path.join(OUT_DIR, "results")

# 绘图风格
plt.style.use('bmh') 
# ===========================================

def compare_models():
    print(f"⚔️ 启动模型终极对比程序...")
    
    # 1. 创建输出目录
    for d in [IMG_DIR, RES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # 2. 读取数据
    models_data = {}
    try:
        if os.path.exists(PATH_A): models_data['Prophet (A)'] = pd.read_csv(PATH_A)
        if os.path.exists(PATH_B): models_data['SARIMA (B)'] = pd.read_csv(PATH_B)
        if os.path.exists(PATH_C): models_data['LSTM (C)'] = pd.read_csv(PATH_C)
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return

    if not models_data:
        print("❌ 没有找到任何模型的结果文件，请先运行前面的模型代码。")
        return

    print(f"   已加载模型: {list(models_data.keys())}")

    # 3. 计算指标 (Metrics Calculation)
    metrics_list = []
    
    for name, df in models_data.items():
        # 确保时间列格式正确
        df['ds'] = pd.to_datetime(df['ds'])
        
        y_true = df['Actual']
        y_pred = df['Predicted_Clean'] # 统一使用清洗后的预测列
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics_list.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2_Score': r2
        })

    # 转换为 DataFrame 并排序 (按 RMSE 从小到大，越小越好)
    metrics_df = pd.DataFrame(metrics_list).sort_values(by='RMSE')
    
    # 打印冠军
    best_model = metrics_df.iloc[0]['Model']
    print(f"\n🏆 综合表现最佳模型: {best_model}")
    print("\n📊 === 详细指标对比 ===")
    print(metrics_df.to_string(index=False))

    # 保存指标表
    metrics_df.to_csv(os.path.join(RES_DIR, "final_metrics_comparison.csv"), index=False)

    # ==========================================
    # 4. 可视化对比 (Visualization)
    # ==========================================
    print("\n🎨 正在生成对比图表...")

    # --- 图 1: 预测曲线对比 (Zoom-in) ---
    plt.figure(figsize=(15, 8))
    
    # 画真实值 (只画一次，取第一个模型的时间轴)
    first_model_df = list(models_data.values())[0]
    plt.plot(first_model_df['ds'], first_model_df['Actual'], label='Ground Truth', color='black', linewidth=2, alpha=0.3)
    
    # 画各模型预测值
    colors = {'Prophet (A)': '#d62728', 'SARIMA (B)': '#1f77b4', 'LSTM (C)': '#2ca02c'}
    linestyles = {'Prophet (A)': '--', 'SARIMA (B)': '-', 'LSTM (C)': ':'}
    
    for name, df in models_data.items():
        plt.plot(df['ds'], df['Predicted_Clean'], 
                 label=f'{name}', 
                 color=colors.get(name, 'blue'), 
                 linestyle=linestyles.get(name, '-'),
                 linewidth=1.5, alpha=0.8)

    plt.title('Final Showdown: Prediction Comparison (Test Set)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Passenger Flow (kg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMG_DIR, "1_prediction_comparison.png"), dpi=300)
    plt.close()

    # --- 图 2: RMSE 柱状图对比 ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_df['Model'], metrics_df['RMSE'], color=['gold', 'silver', '#cd7f32'])
    
    plt.title('Model Performance Ranking (RMSE - Lower is Better)', fontsize=14)
    plt.ylabel('RMSE (kg)')
    plt.grid(axis='y', alpha=0.3)
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f}',
                 ha='center', va='bottom')
                 
    plt.savefig(os.path.join(IMG_DIR, "2_rmse_ranking.png"), dpi=300)
    plt.close()

    print(f"   ✅ 对比完成！结果已保存至: {OUT_DIR}")

if __name__ == "__main__":
    compare_models()
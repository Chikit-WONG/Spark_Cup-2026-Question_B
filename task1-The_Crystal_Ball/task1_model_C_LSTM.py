import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import random

# ================= 配置区域 =================
INPUT_FILE = "./mcm26Train-B-Data_clean/task1_traffic_flow_5min.csv"
BASE_OUT_DIR = "./task1-The_Crystal_Ball/output/model_C_LSTM"
IMG_DIR = os.path.join(BASE_OUT_DIR, "images")
RES_DIR = os.path.join(BASE_OUT_DIR, "results")
FILE_PREFIX = "model_C_LSTM"

# 超参数 (Hyperparameters)
SEQ_LENGTH = 60      # 输入序列长度 (看过去 60个点/5小时 来预测下一个点)
HIDDEN_SIZE = 64     # 隐藏层神经元数量
NUM_LAYERS = 2       # LSTM 层数
LEARNING_RATE = 0.001
EPOCHS = 100         # 训练轮数
BATCH_SIZE = 64

# 绘图风格
plt.style.use('bmh') 

# 固定随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
# ===========================================

# 定义 LSTM 模型结构
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True: 输入格式为 (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态 (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 取序列最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_lstm_model():
    print(f"🚀 启动方案 C (LSTM 深度学习) 预测模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ⚙️ 运行设备: {device}")
    
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
    
    # 提取目标值并进行归一化 (神经网络对数值范围非常敏感，必须缩放到 0-1)
    data = df['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # 3. 划分训练集与测试集 (最后 5 天)
    test_days = 5
    # 计算测试集的行数
    test_size = 5 * 24 * 12 # 5天 * 24小时 * 12个5分钟
    train_size = len(data_scaled) - test_size
    
    # 注意：LSTM需要序列作为输入，所以训练集和测试集切分要小心
    train_data = data_scaled[:train_size]
    # 为了预测测试集的第一个点，我们需要训练集最后 SEQ_LENGTH 个点作为输入
    # 所以这里的 test_data_input 包含了用于生成的上下文
    
    print(f"   数据总长: {len(data_scaled)}, 训练集: {train_size}, 测试集: {test_size}")

    # 4. 构建序列数据 (Sliding Window)
    print("🛠️ 正在构建时间序列切片 (Sequence Windowing)...")
    X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
    
    # 转换为 PyTorch Tensor
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    # 5. 初始化模型
    model = TrafficLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 训练模型
    print(f"⏳ 正在训练 LSTM 模型 (共 {EPOCHS} 轮)...")
    model.train()
    for epoch in range(EPOCHS):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    print("   ✅ 模型训练完成")

    # 7. 预测 (递归预测 Recursive Prediction)
    print("🔮 正在进行递归预测 (这可能比统计模型慢)...")
    model.eval()
    
    # 这里的逻辑是：只用训练集的数据，一步步往后推算出未来5天
    # 初始输入：训练集最后 SEQ_LENGTH 个真实值
    curr_seq = torch.from_numpy(train_data[-SEQ_LENGTH:]).float().to(device).unsqueeze(0) # Shape: (1, seq_len, 1)
    
    predictions_scaled = []
    
    with torch.no_grad():
        for _ in range(test_size):
            # 预测下一步
            next_val_scaled = model(curr_seq)
            predictions_scaled.append(next_val_scaled.item())
            
            # 更新输入序列：去掉最老的一个，加上新预测的一个
            # next_val_scaled shape is (1, 1). Need to reshape/view correctly
            next_val_seq = next_val_scaled.unsqueeze(1) # Shape: (1, 1, 1)
            curr_seq = torch.cat((curr_seq[:, 1:, :], next_val_seq), dim=1)

    # 8. 反归一化 (Inverse Scaling)
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    
    # 整理结果
    # 构造完整的 DataFrame
    # 训练集部分的预测我们这里为了省事暂不回测（因为递归回测太慢），我们主要关注测试集
    # 我们用 Nan 填充训练集部分的预测列，只放测试集结果
    
    full_yhat = np.full(len(df), np.nan)
    full_yhat[-test_size:] = predictions.flatten()
    
    result_df = df.copy()
    result_df['yhat'] = full_yhat
    # 修正负值
    result_df['yhat_clean'] = np.nan_to_num(result_df['yhat']).clip(min=0)
    # 注意：训练集部分 yhat_clean 变成了0，这在画图时要小心，只画后半部分

    # 9. 评估 (只评估测试集)
    test_res = result_df.iloc[-test_size:].copy()
    y_true = test_res['y'].values
    y_pred = test_res['yhat_clean'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📊 === 方案 C (LSTM) 评估结果 ===")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R2 Score: {r2:.4f}")

    # 10. 保存结果
    full_res_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_full_forecast.csv")
    result_df.to_csv(full_res_path, index=False)
    
    comp_df = pd.DataFrame({
        'ds': test_res['ds'],
        'Actual': y_true,
        'Predicted_Clean': y_pred,
        'Error': y_true - y_pred
    })
    comp_res_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_test_comparison.csv")
    comp_df.to_csv(comp_res_path, index=False)
    print(f"   💾 数据已保存至: {RES_DIR}")

    # 11. 生成可视化
    print("🎨 正在生成可视化图表...")
    def get_img_path(name):
        return os.path.join(IMG_DIR, f"{FILE_PREFIX}_{name}")

    # 图 1: 全局预览 (只画最后7天，因为全量预测没做)
    plt.figure(figsize=(14, 6))
    plot_start_idx = len(df) - (7 * 24 * 12)
    subset = result_df.iloc[plot_start_idx:]
    plt.plot(subset['ds'], subset['y'], label='Actual', color='gray', alpha=0.5)
    plt.plot(subset['ds'], subset['yhat_clean'], label='LSTM Prediction (Recursive)', color='green', linewidth=1.5)
    plt.axvline(x=subset['ds'].iloc[-(test_size)], color='orange', linestyle='--', label='Start of Recursive Forecast')
    plt.title(f'LSTM Recursive Forecast (Last 7 Days) - RMSE: {rmse:.2f}')
    plt.legend()
    plt.savefig(get_img_path("1_forecast_overview.png"), dpi=300)
    plt.close()

    # 图 2: 放大测试集
    plt.figure(figsize=(14, 7))
    plt.plot(test_res['ds'], test_res['y'], label='Actual', color='gray', alpha=0.6, linewidth=1.5)
    plt.plot(test_res['ds'], test_res['yhat_clean'], label='LSTM Prediction', color='green', linewidth=2)
    plt.title('Validation Zoom-in (Model C: LSTM)', fontsize=14)
    plt.legend()
    plt.savefig(get_img_path("2_test_set_zoom_in.png"), dpi=300)
    plt.close()

    # 图 3: 拟合散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(test_res['y'], test_res['yhat_clean'], alpha=0.3, color='green')
    max_val = max(test_res['y'].max(), test_res['yhat_clean'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.title('Actual vs Predicted (LSTM)', fontsize=14)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(get_img_path("3_fit_scatter.png"), dpi=300)
    plt.close()
    
    # 图 4: 残差分布
    residuals = test_res['y'] - test_res['yhat_clean']
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Error Distribution (Model C: LSTM)', fontsize=14)
    plt.savefig(get_img_path("4_error_distribution.png"), dpi=300)
    plt.close()

    # 图 5: 残差时间序列
    plt.figure(figsize=(14, 6))
    plt.plot(test_res['ds'], residuals, color='green', alpha=0.8)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residuals over Time (Model C: LSTM)', fontsize=14)
    plt.savefig(get_img_path("5_residuals_over_time.png"), dpi=300)
    plt.close()

    print(f"   🖼️ 所有图表已保存至: {IMG_DIR}")
    print("\n✅ 任务一 (方案C - LSTM) 运行完毕！")

if __name__ == "__main__":
    run_lstm_model()
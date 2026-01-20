import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 输入：清洗后的数据所在目录
INPUT_DIR = "./mcm26Train-B-Data_clean"

# 输出：Task 2 特征文件保存目录 (修改为你要求的路径)
OUTPUT_DIR = "./mcm26Train-B-Data_clean"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ===========================================

def generate_task2_features():
    print(f"🚀 开始构建 Task 2 (分类模式) 特征数据集...")
    print(f"   📂 目标保存路径: {OUTPUT_DIR}")
    
    # 1. 读取清洗后的数据
    # 必须确保你已经运行过 preprocess_data_v3.py 生成了这些文件
    hall_path = os.path.join(INPUT_DIR, 'clean_hall_calls.csv')
    load_path = os.path.join(INPUT_DIR, 'clean_load_changes.csv')
    
    if not os.path.exists(hall_path) or not os.path.exists(load_path):
        print("❌ 错误: 找不到输入文件。请先运行 'preprocess_data.py'。")
        return

    df_hall = pd.read_csv(hall_path)
    df_load = pd.read_csv(load_path)
    
    # 转换时间格式
    df_hall['Time'] = pd.to_datetime(df_hall['Time'])
    df_load['Time'] = pd.to_datetime(df_load['Time'])
    
    # ==========================================
    # 2. 特征工程 (Feature Engineering)
    # ==========================================
    print("   🛠️ 正在提取特征 (流量、方向、楼层分布)...")
    
    # --- 特征 A: 流量强度 (Total Load) ---
    # 按 5 分钟聚合总载重
    feat_load = df_load.set_index('Time').resample('5T')['Load In (kg)'].sum().reset_index()
    feat_load.columns = ['Time', 'Total_Load_kg']
    
    # --- 特征 B: 呼叫方向与位置 (From Hall Calls) ---
    # 预处理：将方向转换为数值
    df_hall['is_Up'] = (df_hall['Direction'] == 'Up').astype(int)
    df_hall['is_Down'] = (df_hall['Direction'] == 'Down').astype(int)
    
    # 预处理：判断是否是大厅 (假设 1 楼是大厅)
    df_hall['is_Lobby'] = (df_hall['Floor'] == 1).astype(int)
    
    # 聚合 Hall Calls 数据 (按 5 分钟)
    feat_calls = df_hall.set_index('Time').resample('5T').agg({
        'Floor': 'count',       # 总呼叫次数 (Total Demand)
        'is_Up': 'sum',         # 上行次数
        'is_Down': 'sum',       # 下行次数
        'is_Lobby': 'sum'       # 大厅出发次数
    }).reset_index()
    
    feat_calls.columns = ['Time', 'Total_Calls', 'Up_Count', 'Down_Count', 'Lobby_Count']
    
    # ==========================================
    # 3. 合并特征表
    # ==========================================
    # 使用 outer join 保证即使某时刻只有载重没有呼叫(或反之)也能保留时间点
    df_features = pd.merge(feat_load, feat_calls, on='Time', how='outer').fillna(0)
    
    # ==========================================
    # 4. 计算关键比例 (Key Ratios)
    # ==========================================
    # 这些比例是 K-Means 聚类区分 "上行高峰" vs "下行高峰" 的核心依据
    # 使用 replace(0, 1) 防止除以零错误
    
    # 上行比例: 接近 1 说明全是上行 (早高峰特征)
    df_features['Up_Ratio'] = df_features['Up_Count'] / df_features['Total_Calls'].replace(0, 1)
    
    # 下行比例: 接近 1 说明全是下行 (晚高峰特征)
    df_features['Down_Ratio'] = df_features['Down_Count'] / df_features['Total_Calls'].replace(0, 1)
    
    # 大厅出发比例: 接近 1 说明所有人都在大厅等车 (早高峰特征)
    df_features['Lobby_Ratio'] = df_features['Lobby_Count'] / df_features['Total_Calls'].replace(0, 1)
    
    # 添加时间辅助特征 (Hour)
    df_features['Hour'] = df_features['Time'].dt.hour
    
    # ==========================================
    # 5. 保存结果
    # ==========================================
    save_filename = "task2_classification_features.csv"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    
    df_features.to_csv(save_path, index=False)
    
    print(f"   ✅ Task 2 特征文件已生成: {save_filename}")
    print(f"      -> 保存路径: {save_path}")
    print(f"      -> 数据量: {len(df_features)} 行")
    print("   👉 准备工作完成！现在可以开始运行 K-Means 聚类模型了。")

if __name__ == "__main__":
    generate_task2_features()
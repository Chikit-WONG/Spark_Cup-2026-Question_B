import pandas as pd
import numpy as np
import os

# ================= 配置区域 (Configuration) =================
# 原始数据所在的文件夹路径
INPUT_DIR = "./mcm26Train-B-Data"

# 清洗后数据要保存的文件夹路径
OUTPUT_DIR = "./mcm26Train-B-Data_clean"
# ==========================================================

def clean_and_save():
    print(f"🚀 初始化数据预处理流水线...")
    print(f"   📂 输入路径: {INPUT_DIR}")
    print(f"   📂 输出路径: {OUTPUT_DIR}")
    print("-" * 30)

    # 1. 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 找不到输入文件夹 '{INPUT_DIR}'")
        print("   请确认你的脚本文件和 'mcm26Train-B-Data' 文件夹在同一个目录下。")
        return

    # 2. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✅ 已创建输出目录: {OUTPUT_DIR}")
    else:
        print(f"✅ 输出目录已存在: {OUTPUT_DIR}")
    
    print("-" * 30)

    # ==========================================
    # 定义通用清洗函数
    # ==========================================
    def standard_clean(df, file_name, time_col='Time'):
        # 转换时间格式
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # 删除无效时间行
        original_len = len(df)
        df = df.dropna(subset=[time_col])
        
        # 按时间排序
        df = df.sort_values(by=time_col).reset_index(drop=True)
        
        print(f"   ✨ {file_name}: 清洗完毕 (行数: {original_len} -> {len(df)})")
        return df

    # 辅助函数：获取完整的文件读取路径
    def get_input_path(filename):
        return os.path.join(INPUT_DIR, filename)

    # 辅助函数：获取完整的保存路径
    def get_save_path(filename):
        return os.path.join(OUTPUT_DIR, filename)

    # ==========================================
    # 3. 逐个处理文件
    # ==========================================
    
    # --- 1. hall_calls.csv ---
    try:
        input_path = get_input_path('hall_calls.csv')
        df_hall = pd.read_csv(input_path)
        df_hall = df_hall.dropna(subset=['Floor']) # 特殊清洗
        df_hall = standard_clean(df_hall, 'hall_calls.csv')
        
        save_path = get_save_path('clean_hall_calls.csv')
        df_hall.to_csv(save_path, index=False)
    except FileNotFoundError:
        print(f"   ❌ 未找到 {input_path}")

    # --- 2. load_changes.csv ---
    try:
        input_path = get_input_path('load_changes.csv')
        df_load = pd.read_csv(input_path)
        df_load = standard_clean(df_load, 'load_changes.csv')
        df_load['Load In (kg)'] = pd.to_numeric(df_load['Load In (kg)'], errors='coerce').fillna(0)
        
        save_path = get_save_path('clean_load_changes.csv')
        df_load.to_csv(save_path, index=False)
    except FileNotFoundError:
        print(f"   ❌ 未找到 {input_path}")

    # --- 3. car_calls.csv ---
    try:
        input_path = get_input_path('car_calls.csv')
        df_car = pd.read_csv(input_path)
        df_car = standard_clean(df_car, 'car_calls.csv')
        
        save_path = get_save_path('clean_car_calls.csv')
        df_car.to_csv(save_path, index=False)
    except FileNotFoundError:
        print(f"   ❌ 未找到 {input_path}")

    # --- 4. car_stops.csv ---
    try:
        input_path = get_input_path('car_stops.csv')
        df_stop = pd.read_csv(input_path)
        df_stop = standard_clean(df_stop, 'car_stops.csv')
        
        save_path = get_save_path('clean_car_stops.csv')
        df_stop.to_csv(save_path, index=False)
    except FileNotFoundError:
        print(f"   ❌ 未找到 {input_path}")

    # --- 5. car_departures.csv ---
    try:
        input_path = get_input_path('car_departures.csv')
        df_dept = pd.read_csv(input_path)
        df_dept = standard_clean(df_dept, 'car_departures.csv')
        
        save_path = get_save_path('clean_car_departures.csv')
        df_dept.to_csv(save_path, index=False)
    except FileNotFoundError:
        print(f"   ❌ 未找到 {input_path}")

    # --- 6. maintenance_mode.csv ---
    try:
        input_path = get_input_path('maintenance_mode.csv')
        df_maint = pd.read_csv(input_path)
        time_cols = [c for c in df_maint.columns if 'Time' in c or 'Start' in c or 'End' in c]
        for col in time_cols:
            df_maint[col] = pd.to_datetime(df_maint[col], errors='coerce')
        
        if time_cols:
            df_maint = df_maint.sort_values(by=time_cols[0]).reset_index(drop=True)
            
        print(f"   ✨ maintenance_mode.csv: 清洗完毕")
        save_path = get_save_path('clean_maintenance_mode.csv')
        df_maint.to_csv(save_path, index=False)
    except FileNotFoundError:
        print(f"   ❌ 未找到 {input_path}")

    # ==========================================
    # 4. 生成 Task 1 专用数据集
    # ==========================================
    print("-" * 30)
    print("📦 正在生成 Task 1 专用的 5分钟聚合流量表...")
    
    if 'df_load' in locals():
        # 核心逻辑：按5分钟聚合流量
        df_task1 = df_load.set_index('Time').resample('5T')['Load In (kg)'].sum().reset_index()
        
        # 尝试合并 Hall Calls 计数作为辅助特征
        if 'df_hall' in locals():
            df_hall_count = df_hall.set_index('Time').resample('5T')['Floor'].count().reset_index()
            df_hall_count.columns = ['Time', 'Hall_Call_Count']
            df_task1 = pd.merge(df_task1, df_hall_count, on='Time', how='left').fillna(0)

        # 重命名适配 Prophet 模型
        df_task1.rename(columns={'Time': 'ds', 'Load In (kg)': 'y'}, inplace=True)
        
        # 保存到指定文件夹
        task1_path = get_save_path('task1_traffic_flow_5min.csv')
        df_task1.to_csv(task1_path, index=False)
        print(f"   🎉 成功生成: task1_traffic_flow_5min.csv")
        print(f"      -> 保存路径: {task1_path}")
    else:
        print("   ⚠️ 缺少 load_changes 数据，跳过生成 Task 1 数据。")

    print(f"\n🏁 脚本运行结束。")

if __name__ == "__main__":
    clean_and_save()
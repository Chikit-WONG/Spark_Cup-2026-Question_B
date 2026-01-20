import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 (Configuration) =================
# 输入数据路径
INPUT_FILE = "./mcm26Train-B-Data_clean/task2_classification_features.csv"

# === 输出路径配置 (方案 A 独立文件夹) ===
BASE_OUT_DIR = "./task2-The_Pulse_of_the_Building/output/model_A_kmeans"
IMG_DIR = os.path.join(BASE_OUT_DIR, "images")
RES_DIR = os.path.join(BASE_OUT_DIR, "results")

# 文件名前缀
FILE_PREFIX = "model_A_kmeans"

# === 修改点: 扩大寻优范围 (测试 K=2 到 K=20) ===
K_RANGE = range(2, 21)

# 绘图风格
plt.style.use('bmh')
# ==========================================================

def run_model_A_kmeans():
    print(f"🚀 启动任务二 [方案 A: K-Means] 自适应聚类模型 (Range: 2-20)...")
    print(f"   📂 目标输出目录: {BASE_OUT_DIR}")
    
    # 1. 创建目录
    for d in [IMG_DIR, RES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"   ✅ 已创建目录: {d}")

    # 2. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 3. 数据准备 (标准化)
    features = ['Total_Load_kg', 'Total_Calls', 'Up_Ratio', 'Down_Ratio', 'Hour']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ==========================================
    # 4. 超参数寻优 (Hyperparameter Tuning)
    # ==========================================
    print(f"📊 正在寻找最佳 K 值 (测试 2 到 20 类)...")
    
    inertia_list = []
    silhouette_list = []
    
    # 进度条提示
    total_k = len(K_RANGE)
    for idx, k in enumerate(K_RANGE):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertia = kmeans.inertia_
        score = silhouette_score(X_scaled, labels)
        
        inertia_list.append(inertia)
        silhouette_list.append(score)
        
        # 每计算3个打印一次，避免刷屏
        if (idx + 1) % 3 == 0 or (idx + 1) == total_k:
            print(f"   [{(idx+1)/total_k:.0%}] Checked K={k}: Silhouette Score = {score:.4f}")

    # --- 绘制优化曲线 ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Error Sum)', color=color)
    ax1.plot(K_RANGE, inertia_list, marker='o', color=color, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    # 强制显示整数刻度
    ax1.set_xticks(K_RANGE)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score (Higher is Better)', color=color)
    ax2.plot(K_RANGE, silhouette_list, marker='s', linestyle='--', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Model A Optimization: K from 2 to 20', fontsize=14)
    plt.tight_layout()
    
    opt_plot_path = os.path.join(IMG_DIR, f"{FILE_PREFIX}_0_optimization_curve.png")
    plt.savefig(opt_plot_path, dpi=300)
    plt.close()
    print(f"   📈 优化曲线已保存: {opt_plot_path}")

    # ==========================================
    # 5. 使用最佳 K 运行最终模型
    # ==========================================
    best_k_idx = np.argmax(silhouette_list)
    best_k = K_RANGE[best_k_idx]
    best_score = silhouette_list[best_k_idx]
    
    print(f"\n🏆 最佳方案选中: K = {best_k} (Score: {best_score:.4f})")
    print(f"🚀 正在生成最终聚类结果...")
    
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = final_kmeans.fit_predict(X_scaled)
    df['Cluster_Label'] = clusters
    
    # ==========================================
    # 6. 动态命名逻辑 (Dynamic Naming for large K)
    # ==========================================
    cluster_profile = df.groupby('Cluster_Label')[features].mean()
    cluster_profile['Count'] = df['Cluster_Label'].value_counts().sort_index()
    
    label_map = {}
    print("\n🏷️ Assigning Dynamic Names:")
    
    for i, row in cluster_profile.iterrows():
        # 获取特征
        h = row['Hour']
        load = row['Total_Load_kg']
        up = row['Up_Ratio']
        down = row['Down_Ratio']
        
        # A. [时段 Period]
        if h < 6: period = "Night"
        elif 6 <= h < 9: period = "Early-Morn"
        elif 9 <= h < 11: period = "Late-Morn"
        elif 11 <= h < 14: period = "Lunch"
        elif 14 <= h < 17: period = "Afternoon"
        elif 17 <= h < 20: period = "Evening"
        else: period = "Late-Night"

        # B. [强度 Intensity]
        if load < 300: intensity = "Idle"
        elif load < 1500: intensity = "Light"
        elif load < 4000: intensity = "Moderate"
        elif load < 7000: intensity = "Heavy"
        else: intensity = "Extreme"

        # C. [方向 Direction]
        if intensity == "Idle":
            direction_str = "" 
        elif up > 0.60:
            direction_str = "Up-Flow"
        elif down > 0.60:
            direction_str = "Down-Flow"
        elif abs(up - down) < 0.2:
            direction_str = "Balanced"
        else:
            direction_str = "Mixed"

        # D. 组合名称
        if intensity == "Idle":
            final_name = f"{period} {intensity}"
        else:
            final_name = f"{period} {intensity} {direction_str}"
        
        # 去除多余空格
        final_name = " ".join(final_name.split())
        label_map[i] = final_name
        print(f"   Cluster {i} -> {final_name}")

    df['Cluster_Name'] = df['Cluster_Label'].map(label_map)

    # ==========================================
    # 7. 保存结果文件 (CSV)
    # ==========================================
    res_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_clustered_results.csv")
    df.to_csv(res_path, index=False)
    
    prof_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_cluster_profiles.csv")
    cluster_profile.to_csv(prof_path)
    
    # 保存指标
    metrics_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_metrics.csv")
    metrics_df = pd.DataFrame([{
        'Model': 'K-Means (Model A)',
        'Best_K': best_k,
        'Silhouette_Score': best_score,
        'Inertia': final_kmeans.inertia_
    }])
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"   💾 结果数据已保存至: {RES_DIR}")

    # ==========================================
    # 8. 生成可视化图表 (Images)
    # ==========================================
    print("🎨 正在生成最终图表...")
    
    def get_img_path(name):
        return os.path.join(IMG_DIR, f"{FILE_PREFIX}_{name}")

    # [图1] 热力图 (Pattern Heatmap)
    # 注意：K很大时热力图会变长，调整figsize
    heatmap_data = pd.crosstab(df['Hour'], df['Cluster_Name'], normalize='index')
    plt.figure(figsize=(12, max(8, best_k * 0.5))) # 动态调整高度
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Probability'})
    plt.title(f'Model A: Traffic Pattern Probability (K={best_k})', fontsize=14)
    plt.tight_layout()
    plt.savefig(get_img_path("1_pattern_heatmap.png"), dpi=300)
    plt.close()

    # [图2] 时间轴散点图 (Timeline)
    plt.figure(figsize=(15, 7))
    subset = df[df['Time'] < df['Time'].min() + pd.Timedelta(days=3)]
    
    # 如果类别很多，使用 tab20 调色板
    palette = 'tab20' if best_k > 10 else 'bright'
    
    sns.scatterplot(data=subset, x='Time', y='Total_Load_kg', 
                    hue='Cluster_Name', palette=palette, s=40, alpha=0.9)
    plt.title(f'Model A: Identified Patterns Timeline (K={best_k})', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1) # 图例放右边
    plt.tight_layout()
    plt.savefig(get_img_path("2_pattern_timeline.png"), dpi=300)
    plt.close()

    print(f"   🖼️ 所有图表已保存至: {IMG_DIR}")
    print("\n✅ 任务二 [方案 A: K-Means (K=2~20)] 运行完毕！")

if __name__ == "__main__":
    run_model_A_kmeans()
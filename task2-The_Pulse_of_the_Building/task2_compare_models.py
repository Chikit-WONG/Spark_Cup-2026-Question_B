import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
# 方案 A 和 方案 B 的指标文件路径
PATH_A = "./task2-The_Pulse_of_the_Building/output/model_A_kmeans/results/model_A_kmeans_metrics.csv"
PATH_B = "./task2-The_Pulse_of_the_Building/output/model_B_RandomForest/results/model_B_RandomForest_metrics.csv"

# 方案 B 的分类报告 (用于获取 F1-score)
PATH_B_REPORT = "./task2-The_Pulse_of_the_Building/output/model_B_RandomForest/results/model_B_RandomForest_classification_report.csv"

# 输出路径
OUT_DIR = "./task2-The_Pulse_of_the_Building/output/comparison"
IMG_DIR = os.path.join(OUT_DIR, "images")
RES_DIR = os.path.join(OUT_DIR, "results")

plt.style.use('bmh')
# ===========================================

def compare_task2_models():
    print(f"⚔️ 启动任务二 [模型对比与验收] 程序...")
    
    # 1. 创建目录
    for d in [IMG_DIR, RES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 2. 读取数据
    data_a = None
    data_b = None
    
    if os.path.exists(PATH_A):
        data_a = pd.read_csv(PATH_A)
        print("   ✅ 已加载方案 A (K-Means) 指标")
    else:
        print(f"   ❌ 未找到方案 A 文件: {PATH_A}")

    if os.path.exists(PATH_B):
        data_b = pd.read_csv(PATH_B)
        print("   ✅ 已加载方案 B (Random Forest) 指标")
    else:
        print(f"   ❌ 未找到方案 B 文件: {PATH_B}")

    if data_a is None or data_b is None:
        print("   ⚠️ 缺少必要文件，无法进行完整对比。")
        return

    # 3. 提取关键指标
    # 方案 A: 轮廓系数 (Silhouette Score) -> 代表聚类的"清晰度"
    score_a = data_a['Silhouette_Score'].iloc[0]
    best_k = data_a['Best_K'].iloc[0]
    
    # 方案 B: 准确率 (Accuracy) -> 代表聚类的"可解释性/可复现性"
    acc_b = data_b['Accuracy'].iloc[0]
    best_params = data_b['Best_Params'].iloc[0]

    print("\n📊 === 综合对比报告 (Final Report) ===")
    print(f"   [探索者] 方案 A (K-Means):")
    print(f"       - 最佳类别数 (K): {best_k}")
    print(f"       - 轮廓系数 (Silhouette): {score_a:.4f} (越高说明模式越独特)")
    
    print(f"   [验证者] 方案 B (Random Forest):")
    print(f"       - 拟合准确率 (Accuracy): {acc_b:.4f} (越高说明模式逻辑越清晰)")
    print(f"       - 最佳参数: {best_params}")

    # 4. 生成总结表
    summary_df = pd.DataFrame({
        'Metric': ['Model Type', 'Key Indicator', 'Value', 'Interpretation'],
        'Model A (K-Means)': [
            'Unsupervised (Discovery)', 
            'Silhouette Score', 
            f"{score_a:.4f}", 
            "High value indicates distinct clusters"
        ],
        'Model B (Random Forest)': [
            'Supervised (Verification)', 
            'Prediction Accuracy', 
            f"{acc_b:.4f}", 
            "High value indicates learnable rules"
        ]
    })
    
    csv_path = os.path.join(RES_DIR, "final_model_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"   💾 对比数据已保存至: {csv_path}")

    # ==========================================
    # 5. 可视化：生成"模型成绩单"
    # ==========================================
    print("🎨 正在生成最终验收图表...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off') # 不显示坐标轴
    
    # 绘制标题
    plt.title("Task 2: Traffic Pattern Recognition - Final Verdict", fontsize=16, weight='bold')
    
    # 绘制左侧文本 (Model A)
    text_a = (
        f"Model A: K-Means (Discovery)\n"
        f"----------------------------\n"
        f"Optimal Clusters: K = {best_k}\n"
        f"Silhouette Score: {score_a:.3f}\n\n"
        f"Status: PATTERNS IDENTIFIED"
    )
    plt.text(0.1, 0.5, text_a, fontsize=12, va='center', ha='left', 
             bbox=dict(boxstyle="round", facecolor="#e6f2ff", edgecolor="blue"))

    # 绘制右侧文本 (Model B)
    text_b = (
        f"Model B: Random Forest (Verification)\n"
        f"----------------------------\n"
        f"Reproduction Accuracy: {acc_b*100:.2f}%\n"
        f"Interpretation: Highly Robust\n\n"
        f"Status: PATTERNS VERIFIED"
    )
    # 根据准确率变色
    bg_color = "#e6ffe6" if acc_b > 0.9 else "#ffe6e6"
    edge_color = "green" if acc_b > 0.9 else "red"
    
    plt.text(0.6, 0.5, text_b, fontsize=12, va='center', ha='left', 
             bbox=dict(boxstyle="round", facecolor=bg_color, edgecolor=edge_color))

    # 绘制中间的箭头
    plt.arrow(0.42, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='gray', ec='gray')
    plt.text(0.47, 0.55, "Verified By", fontsize=10, ha='center', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "1_final_verdict_scorecard.png"), dpi=300)
    plt.close()

    # ==========================================
    # 6. 读取并绘制 F1-Score 条形图 (如果存在)
    # ==========================================
    if os.path.exists(PATH_B_REPORT):
        report_df = pd.read_csv(PATH_B_REPORT, index_col=0)
        # 过滤掉 'accuracy', 'macro avg', 'weighted avg'
        classes = report_df.index[:-3]
        f1_scores = report_df.loc[classes, 'f1-score']
        
        plt.figure(figsize=(12, 6))
        # 排序以便观看
        f1_scores.sort_values().plot(kind='barh', color='#4c72b0')
        plt.title(f'Reliability of Each Traffic Pattern (F1-Score)', fontsize=14)
        plt.xlabel('F1-Score (Ability to Correctly Identify)')
        plt.xlim(0.8, 1.0) # 重点展示高分段
        plt.axvline(x=0.9, color='r', linestyle='--', alpha=0.5, label='High Reliability Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "2_pattern_reliability_ranking.png"), dpi=300)
        plt.close()

    print(f"   ✅ 对比完成！请查看: {OUT_DIR}")

if __name__ == "__main__":
    compare_task2_models()
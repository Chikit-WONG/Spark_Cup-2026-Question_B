import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 (Configuration) =================
# 输入：方案 A (K-Means) 生成的带标签数据
# 确保这个路径指向你 Task 2 方案 A 生成的最新结果文件
INPUT_FILE = "./task2-The_Pulse_of_the_Building/output/model_A_kmeans/results/model_A_kmeans_clustered_results.csv"

# 输出路径
BASE_OUT_DIR = "./task2-The_Pulse_of_the_Building/output/model_B_RandomForest"
IMG_DIR = os.path.join(BASE_OUT_DIR, "images")
RES_DIR = os.path.join(BASE_OUT_DIR, "results")
FILE_PREFIX = "model_B_RandomForest"

# 超参数搜索范围 (Grid Search)
# 注意：如果内存不足，可以减少 n_estimators 的选项或将 n_jobs 设为 1
# PARAM_GRID = {
#     'n_estimators': [50, 100, 150, 200, 250, 300],      # 树的数量
#     'max_depth': [None, 10, 20, 30, 100],          # 树的最大深度
#     'min_samples_split': [3, 4, 5, 6, 7]               # 节点分裂最小样本数
# }
PARAM_GRID = {
    'n_estimators': [245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255],      # 树的数量
    'max_depth': [None],          # 树的最大深度
    'min_samples_split': [5]               # 节点分裂最小样本数
}


# 绘图风格
plt.style.use('bmh')
# ==========================================================

def run_model_B_rf_optimized():
    print(f"🚀 启动任务二 [方案 B: Random Forest] 超参数寻优增强版...")
    print(f"   📂 目标输出目录: {BASE_OUT_DIR}")
    
    # 1. 目录准备
    for d in [IMG_DIR, RES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"   ✅ 已创建目录: {d}")

    # 2. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        print("   请先运行方案 A (task2_model_A_kmeans_v2.py) 生成标签。")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"   已加载数据: {len(df)} 行")
    
    # 3. 准备特征与标签
    # K-Means 聚类使用的核心特征，加上 'Cluster_Label' 作为目标
    features = ['Total_Load_kg', 'Total_Calls', 'Up_Ratio', 'Down_Ratio', 'Hour']
    X = df[features]
    y = df['Cluster_Label']
    
    # 获取 "Label ID" 到 "Label Name" 的映射字典 (用于画图显示真实名字)
    label_map = df[['Cluster_Label', 'Cluster_Name']].drop_duplicates().set_index('Cluster_Label')['Cluster_Name'].to_dict()
    # 排序后的标签名称列表
    unique_labels = sorted(list(set(y)))
    label_names = [label_map[i] for i in unique_labels]
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 4. 超参数网格搜索 (Grid Search)
    # ==========================================
    print(f"🔍 开始超参数寻优 (Grid Search)...")
    print(f"   参数空间: {PARAM_GRID}")
    
    rf = RandomForestClassifier(random_state=42)
    
    # n_jobs=-1 使用所有CPU核心。如果内存不足报错，请改为 n_jobs=2 或 n_jobs=1
    grid_search = GridSearchCV(estimator=rf, param_grid=PARAM_GRID, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n🏆 最佳参数组合 found:")
    print(f"   {best_params}")
    print(f"   训练集验证得分: {best_score:.4f}")

    # ==========================================
    # 5. 模型评估与结果保存 (增强版)
    # ==========================================
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 === 最终测试集评估 ===")
    print(f"   Test Accuracy: {test_acc:.4f}")

    # [文件1] 基础指标 (Metrics)
    metrics_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_metrics.csv")
    pd.DataFrame([{
        'Model': 'Random Forest (Optimized)',
        'Accuracy': test_acc,
        'Best_Params': str(best_params),
        'Training_Set_Size': len(X_train),
        'Test_Set_Size': len(X_test)
    }]).to_csv(metrics_path, index=False)
    
    # [文件2] 详细分类报告 (Classification Report) - 新增！
    # 这会保存每个类别的 Precision, Recall, F1-Score
    report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=label_names)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_classification_report.csv")
    report_df.to_csv(report_path)
    print(f"   💾 [新增] 详细分类报告已保存: {report_path}")
    
    # [文件3] 保存模型对象 (.pkl) - 新增！
    # 方便 Task 3 直接加载使用
    model_path = os.path.join(RES_DIR, f"{FILE_PREFIX}_best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"   💾 [新增] 模型文件已保存: {model_path}")

    # ==========================================
    # 6. 生成高级可视化图表 (5张图)
    # ==========================================
    print("🎨 正在生成高级可视化图表...")
    
    def get_img_path(name):
        return os.path.join(IMG_DIR, f"{FILE_PREFIX}_{name}")

    # --- 图 1: 超参数性能热力图 (Heatmap) ---
    try:
        results_df = pd.DataFrame(grid_search.cv_results_)
        # 聚合数据: 这里展示 n_estimators vs max_depth 对分数的影响
        pivot_table = results_df.pivot_table(index='param_max_depth', 
                                             columns='param_n_estimators', 
                                             values='mean_test_score')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis")
        plt.title('Hyperparameter Performance (Accuracy)', fontsize=14)
        plt.xlabel('Number of Trees (n_estimators)')
        plt.ylabel('Max Depth')
        plt.tight_layout()
        plt.savefig(get_img_path("1_hyperparameter_heatmap.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"   ⚠️ 无法生成热力图 (可能是参数维度不足): {e}")

    # --- 图 2: 特征重要性 (Feature Importance) ---
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
    plt.title(f'Feature Importance (Best Model)', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(get_img_path("2_feature_importance.png"), dpi=300)
    plt.close()

    # --- 图 3: 混淆矩阵 (Confusion Matrix) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix (Optimized RF)', fontsize=14)
    plt.ylabel('True Label (K-Means)')
    plt.xlabel('Predicted Label (RF)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(get_img_path("3_confusion_matrix.png"), dpi=300)
    plt.close()

    # --- 图 4: 分类报告可视化 (Report Heatmap) ---
    # 去掉最后几行 aggregate (accuracy, macro avg 等)，只看具体类别的得分
    heatmap_df = report_df.iloc[:-3, :3] 
    plt.figure(figsize=(10, len(label_names)*0.5 + 3))
    sns.heatmap(heatmap_df, annot=True, cmap="RdYlGn", fmt=".2f", vmin=0.8, vmax=1.0)
    plt.title('Class-wise Performance Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(get_img_path("4_classification_report.png"), dpi=300)
    plt.close()

    # --- 图 5: 学习曲线 (Learning Curve) ---
    # 检查是否过拟合
    print("   绘制学习曲线中 (可能需要几秒)...")
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation Score")
    plt.title('Learning Curve (Detect Overfitting)', fontsize=14)
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_img_path("5_learning_curve.png"), dpi=300)
    plt.close()

    print(f"   🖼️ 所有 5 张图表已保存至: {IMG_DIR}")
    print("\n✅ 任务二 [方案 B: 随机森林增强版] 运行完毕！")

if __name__ == "__main__":
    run_model_B_rf_optimized()
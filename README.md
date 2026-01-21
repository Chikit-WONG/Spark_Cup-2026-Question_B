- # Spark Cup 2026 — Question B

  与AI的聊天记录：

  数据预处理、任务一和任务二：

  https://gemini.google.com/share/6e6742bbc8a3

  

  电梯运行数据分析与建模，包含三大任务：

  - **Task 1 (The Crystal Ball)**: 预测未来乘梯流量（5 分钟粒度），对比 Prophet / SARIMA / LSTM 三种时序模型。
  - **Task 2 (The Pulse of the Building)**: 识别并验证楼宇交通模式，先用 K-Means 发现簇，再用 Random Forest 进行可复现性验证。
  - **Task 3 (The Strategic Wait)**: 动态停靠策略优化，使用 MDP+Q-Learning / 多目标优化 / 仿真验证 三种方法。

  ## 目录结构

  - preprocess_data.py：清洗原始 CSV 并生成任务数据集。
  - mcm26Train-B-Data/：原始数据（hall_calls、load_changes 等）。
  - mcm26Train-B-Data_clean/：清洗后的数据及 Task1 聚合表。
  - task1-The_Crystal_Ball/：任务一模型与对比脚本（Prophet、SARIMA、LSTM）。
  - task2-The_Pulse_of_the_Building/：任务二模型与对比脚本（K-Means、Random Forest）。

  ## 代码结构

  ```text
  Spark_Cup-2026-Question_B/
  ├─ preprocess_data.py                 # 原始数据清洗与 Task1 聚合生成
  ├─ mcm26Train-B-Data/                 # 原始数据（输入）
  ├─ mcm26Train-B-Data_clean/           # 清洗后数据与 task1_traffic_flow_5min.csv, task2_classification_features.csv
  ├─ task1-The_Crystal_Ball/
  │  ├─ task1_model_A_prophet.py        # 方案A：Prophet 预测
  │  ├─ task1_model_B_SARIMA.py         # 方案B：SARIMA 预测
  │  ├─ task1_model_C_LSTM.py           # 方案C：LSTM 预测
  │  ├─ task1_compare_models.py         # 任务一模型指标与图表对比
  │  └─ output/
  │     ├─ model_A_prophet/
  │     │  ├─ images/
  │     │  └─ results/
  │     ├─ model_B_SARIMA/
  │     │  ├─ images/
  │     │  └─ results/
  │     ├─ model_C_LSTM/
  │     │  ├─ images/
  │     │  └─ results/
  │     └─ comparison/
  │        ├─ images/
  │        └─ results/
  ├─ task2-The_Pulse_of_the_Building/
     ├─ preprocess_task2_features.py    # 任务二特征工程（如需要）
     ├─ task2_model_A_kmeans.py         # 模式发现：K-Means
     ├─ task2_model_B_RandomForest.py   # 模式验证：Random Forest
     ├─ task2_compare_models.py         # 任务二对比与可视化
     └─ output/
        ├─ model_A_kmeans/
        │  ├─ images/
        │  └─ results/
        ├─ model_B_RandomForest/
        │  ├─ images/
        │  └─ results/
        └─ comparison/
           ├─ images/
           └─ results/
  ├─ task3-The_Strategic_Wait/
     ├─ task3_preprocess.py             # 任务三数据预处理
     ├─ task3_model_A_MDP_QLearning.py  # 方案A：MDP + Q-Learning
     ├─ task3_model_B_MultiObjective.py # 方案B：多目标优化 NSGA-II
     ├─ task3_model_C_Simulation.py     # 方案C：离散事件仿真
     ├─ task3_compare_models.py         # 任务三对比与综合
     ├─ data/                           # 预处理输出数据
     └─ output/
        ├─ model_A_MDP_QLearning/
        │  ├─ images/
        │  └─ results/
        ├─ model_B_MultiObjective/
        │  ├─ images/
        │  └─ results/
        ├─ model_C_Simulation/
        │  ├─ images/
        │  └─ results/
        └─ comparison/
           ├─ images/
           └─ results/
  ```

  ## 环境依赖

  Python 3.10+，建议使用虚拟环境。主要依赖：

  - pandas, numpy, matplotlib, scikit-learn
  - prophet, statsmodels (SARIMA)
  - torch (LSTM 需要；可用 GPU)

  安装示例：

  ```bash
  pip install pandas numpy matplotlib scikit-learn prophet statsmodels torch
  ```

  ## 快速开始

  0. **进入文件夹**：

     ```bash
     cd Spark_Cup-2026-Question_B
     ```

     以下命令均在路径`path/to/Spark_Cup-2026-Question_B`运行

  1) **准备数据**：确保原始数据位于 `./mcm26Train-B-Data/`。若路径不同，修改 preprocess_data.py 中的 `INPUT_DIR`。

  2) **数据清洗与聚合**：

  ```bash
  python ./preprocess_data.py
  ```

  输出存放在 `mcm26Train-B-Data_clean/`，并生成 Task1 所需的 `task1_traffic_flow_5min.csv`。

  3) **任务一：流量预测（单模型运行示例）**

  ```bash
  python ./task1-The_Crystal_Ball/task1_model_A_prophet.py
  python ./task1-The_Crystal_Ball/task1_model_B_SARIMA.py
  python ./task1-The_Crystal_Ball/task1_model_C_LSTM.py
  ```

  模型输出：`task1-The_Crystal_Ball/output/model_*/`，包含结果 CSV 与可视化图片。若在集群运行，可参考 `scripts/run_task1-The_Crystal_Ball-Prophet.sh`。

  4) **任务一：模型对比**

  ```bash
  python ./task1-The_Crystal_Ball/task1_compare_models.py
  ```

  汇总指标与对比图保存在 `task1-The_Crystal_Ball/output/comparison/`。

  5) **任务二：模式发现与验证**

  ```bash
  python ./task2-The_Pulse_of_the_Building/preprocess_task2_features.py # 先对任务二需要的数据进行预处理
  python ./task2-The_Pulse_of_the_Building/task2_model_A_kmeans.py
  python ./task2-The_Pulse_of_the_Building/task2_model_B_RandomForest.py
  python ./task2-The_Pulse_of_the_Building/task2_compare_models.py
  ```

  输出位于 `task2-The_Pulse_of_the_Building/output/`，含聚类结果、分类报告与最终对比图。

  6) **任务三：策略性等待（动态停靠策略优化）**

  ```bash
  # 步骤1: 数据预处理
  python ./task3-The_Strategic_Wait/task3_preprocess.py
  
  # 步骤2: 运行三种优化方法
  python ./task3-The_Strategic_Wait/task3_model_A_MDP_QLearning.py    # 方案A: MDP + Q-Learning
  python ./task3-The_Strategic_Wait/task3_model_B_MultiObjective.py   # 方案B: 多目标优化 NSGA-II
  python ./task3-The_Strategic_Wait/task3_model_C_Simulation.py       # 方案C: 离散事件仿真
  
  # 步骤3: 模型对比
  python ./task3-The_Strategic_Wait/task3_compare_models.py
  ```

  输出位于 `task3-The_Strategic_Wait/output/`，包含：
  - 最优停靠策略表
  - 帕累托最优解集
  - 仿真验证结果
  - 模型对比报告

  ## 结果产物

  - 清洗数据：`mcm26Train-B-Data_clean/clean_*.csv`，`task1_traffic_flow_5min.csv`，`task2_classification_features.csv`。
  - 任务一模型结果：`output/model_A_prophet`、`model_B_SARIMA`、`model_C_LSTM` 内的 CSV 与图表。
  - 任务一对比：`output/comparison/final_metrics_comparison.csv` 及图片。
  - 任务二结果：`output/model_A_kmeans`、`model_B_RandomForest` 生成的指标与报告。
  - 任务二对比：`output/comparison/final_model_comparison.csv` 与评分卡图片。
  - 任务三结果：`task3-The_Strategic_Wait/output/` 下各模型输出：
    - MDP 学习策略表、训练曲线
    - 多目标帕累托解集与收敛图
    - 仿真验证的策略排名与推荐
    - 综合对比分析与最终推荐

  ## 备注

  - LSTM 训练可使用 GPU；未检测到 GPU 时自动退回 CPU。
  - 若修改数据路径，请同步调整脚本顶部的配置常量。
  - 输出目录不存在时脚本会自动创建。


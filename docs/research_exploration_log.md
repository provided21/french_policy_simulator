# Research Exploration Log

项目：French Policy Simulator 数据科学期末项目  
主题：基于合成法国居民画像与 LLM 的政策态度模拟  
当前主数据：`data/processed/modeling_samples/main_1w_faiss_p2_clean.parquet`

## 0. 记录目的

这份文档记录我们的研究思路如何从数据出发逐步生长出来。

主线不是“做了哪些技术”，而是：

1. 找到可用数据；
2. 思考这些数据能支持什么问题；
3. 提出可检验设想；
4. 想清楚 ground truth 从哪里来，并检索真实社会调查作为研究案例；
5. 构建实验框架；
6. 通过实验结果不断修正方向；
7. 最终形成可用于数据科学分析的主数据集与建模问题。

其中旧数据问题会作为方法审计的一部分记录，但不作为研究主线。

## 1. 数据起点：我们实际拥有什么？

项目的核心数据不是传统问卷微观数据，而是一批大规模合成法国居民画像。

可用数据包括：

- 约 600 万条法国居民合成画像，存储在 `data/processed/df_full.parquet`。
- 每个画像包含人口统计、职业、家庭、地区、性格、文化背景、生活方式、技能和职业目标等字段。
- 已有 LLM 调用结果存储在 `data/results.db` 的 `responses` 表中。

单个画像大致包含以下维度：

- 基础人口信息：年龄、性别；
- 社会经济信息：职业、学历、婚姻、家庭结构；
- 地理信息：省份、城镇；
- 文本画像：`persona`、`cultural_background`、`professional_persona`；
- 生活方式：体育、艺术、旅行、美食、兴趣；
- 职业发展：技能、职业目标；
- 拼接后的完整画像：`persona_text`。

这批数据的特点：

- 规模大，适合做抽样、检索和机器学习实验；
- 信息维度丰富，适合比较结构化变量和文本画像变量；
- 但它不是实际调查数据，因此态度标签需要由 LLM 生成；
- 真实民调只能作为宏观参照，而不能直接提供个体级 ground truth。

## 2. 从数据出发：可以做什么研究？

看到这批数据后，我们先形成了几个可能方向：

### 方向 A：做一个政策态度模拟系统

给定一个政策问题，从画像库中选择一批“居民”，让 LLM 扮演这些人并回答政策态度。

这个方向更偏系统实现，关注：

- 如何构建画像；
- 如何检索相关人群；
- 如何批量调用 LLM；
- 如何展示结果。

### 方向 B：做 LLM 民调模拟的方法评估

不只展示系统，而是问：

> LLM 生成的政策态度分布，能否在宏观上接近真实民调？

这个方向需要真实调查作为基准。我们选择法国退休改革议题，因为它有明确真实民调参照，也与年龄、职业、劳动强度、社会阶层高度相关。

### 方向 C：做个体层面的数据科学建模

如果 LLM 为每个画像生成了态度，那么可以进一步问：

> 画像中的哪些特征能够预测 LLM 生成的态度？

这使项目进入数据科学/机器学习范式：

- X：画像特征；
- y：LLM 生成的 `support_score` 和 `stance`；
- 任务：回归与分类；
- 分析目标：特征工程、模型比较、解释性分析。

我们最终把三个方向串起来：

1. 先构建 LLM 政策态度模拟框架；
2. 用真实民调做宏观校准；
3. 再把生成结果转化为监督学习数据，分析 LLM 态度生成机制。

## 3. 初始研究设想

围绕法国退休年龄政策，我们形成了几个可检验设想。

### 设想 1：抽样策略会显著影响宏观模拟结果

传统调查强调随机抽样，但 LLM 角色扮演不一定适合完全随机画像。

原因：

- 退休改革是一个有强利益相关性的议题；
- 与退休、劳动、年龄弱相关的画像可能让 LLM 更倾向中立或泛泛回答；
- 语义检索可以先筛出与议题更相关的人群。

因此提出对比：

- Random：从画像库均匀随机抽样；
- FAISS：用 Sentence-BERT + FAISS 检索与政策问题语义相关的画像。

### 设想 2：Prompt 设计会显著影响 LLM 民调分布

LLM 不是稳定测量仪器。问题表述、角色约束、输出格式和上下文要求都可能改变结果。

因此不能只跑一个 prompt，而要把 prompt 当作实验变量。

### 设想 3：完整画像可能优于只给人口统计信息

如果 LLM 能有效使用 persona，那么完整文本画像应该帮助它生成更细致的态度。

但也可能相反：

- 文本字段太多会引入噪声；
- 生活方式字段可能与政策态度弱相关；
- 完整画像可能稀释年龄/职业等关键信号。

因此需要比较：

- 只给人口统计；
- 加职业经济字段；
- 给完整画像；
- 给全字段精简摘要。

### 设想 4：LLM 生成的态度应当存在社会分层结构

如果模拟不是随机的，那么态度应与以下变量有关：

- 年龄；
- 职业；
- 学历；
- 是否退休；
- 家庭/经济压力。

后续建模的核心问题就是：

> 这些结构化变量和文本变量能否解释 LLM 生成的态度差异？

## 4. Ground Truth 从哪里来：真实社会调查作为宏观校准

在构建实验框架之前，我们必须先回答一个基础问题：

> 如果 LLM 是在给合成画像生成态度，那么“真实答案”到底在哪里？

这个问题决定了整个项目不能简单套用传统监督学习逻辑。

### 4.1 没有个体级 ground truth

我们的画像是合成数据，不是真实受访者。

因此不存在如下意义上的个体真值：

- 某个具体 persona 在真实世界中是否支持恢复到 62 岁；
- 某个 persona 的真实 `support_score` 应该是多少；
- 某个 persona 的真实政治态度能否被直接验证。

这意味着：

- LLM 生成的 `support_score` 不是现实个体标签；
- 后续 ML 建模不能被解释为“预测真实法国人的态度”；
- 它更准确的含义是：预测 LLM 在给定画像条件下生成的模拟态度。

因此，个体级建模的研究问题应表述为：

> 哪些画像特征能够解释 LLM 生成态度的差异？

而不是：

> 哪些画像特征能够解释真实法国人的态度差异？

### 4.2 宏观 ground truth：真实社会调查

虽然没有个体级 ground truth，但我们可以使用真实社会调查作为宏观校准基准。

具体做法：

- 检索法国退休改革相关真实民调；
- 寻找问题表述与我们政策主张最接近的调查；
- 将真实民调中的总体分布作为宏观 benchmark；
- 比较 LLM 模拟分布与真实调查分布的差距。

这使项目评价从“个体标签准确率”转为“宏观分布拟合”：

- LLM 输出不是逐个个体验证；
- 而是看总体支持/中立/反对比例是否接近真实民调；
- 再进一步分析不同抽样策略、prompt 设计如何影响宏观拟合。

### 4.3 选择法国退休改革作为研究案例

我们选择法国退休年龄改革作为案例，是因为它同时满足三个条件。

第一，它有明确真实社会调查参照。

项目材料中整理了 IFOP 等法国民调。最终主基准采用：

- 支持回到 62 岁：61%；
- 维持 64 岁：34%；
- 进一步提高退休年龄：5%。

第二，它与画像字段高度相关。

退休改革天然关联：

- 年龄；
- 是否临近退休；
- 是否已经退休；
- 职业劳动强度；
- 学历与社会阶层；
- 家庭和经济压力。

这意味着它不仅能做宏观模拟，也能支持后续特征工程和机器学习解释。

第三，它适合检验 LLM 角色扮演是否真正使用 persona 信息。

如果 LLM 只是泛泛输出政治常识，那么不同年龄、职业、学历之间不应出现稳定差异。

但如果 LLM 确实利用 persona，那么我们应观察到：

- 年龄越大越倾向支持恢复 62 岁；
- 体力劳动者和低学历群体更支持；
- 高学历、高职业地位群体更可能中立或反对；
- reason 中应出现年龄、退休、工作压力、经济压力等线索。

因此，真实社会调查不仅提供宏观 benchmark，也帮助我们判断 LLM 模拟是否产生了合理的社会结构。

### 4.4 对后续实验设计的影响

明确 ground truth 来源后，实验框架也随之改变。

我们不再追求不存在的个体级真实标签，而是分成两层评价：

1. 宏观层面：
   - LLM 生成的支持/中立/反对分布是否接近 IFOP；
   - 抽样策略和 prompt 设计如何影响分布拟合。

2. 个体机制层面：
   - 在选定的 LLM 模拟流程下，哪些画像特征解释了 LLM 的态度生成；
   - 结构化变量和文本画像分别贡献多少；
   - 模型是否形成可解释的社会分层结构。

这也是为什么后续要先做 `sampling × prompt` 交叉实验，再扩展 1w 主数据并进行 ML 建模。

## 5. 实验框架构建

基于上述设想，我们构建了一个三阶段框架。

### 阶段 1：宏观模拟流程选择

目标：

找到一个相对稳定、可解释、并且宏观上接近真实 IFOP 基准的 LLM 模拟流程。

比较维度：

- 抽样策略：Random vs FAISS；
- Prompt 版本：P0 / P1 / P2 / P3；
- 模型温度：固定为 `temperature = 0.2`。

真实 IFOP 基准：

- 支持回到 62 岁：61%；
- 中立/维持 64 岁：34%；
- 反对/进一步提高：5%。

### 阶段 2：扩展主数据

用阶段 1 中表现最好的组合跑 1w 条样本，形成后续数据科学主数据。

主数据需要满足：

- 无重复画像；
- API 调用成功；
- stance 与 score 一致；
- 保留完整画像字段；
- 保留 LLM response、reason、support_score、stance。

### 阶段 3：数据科学建模

将主数据转化为监督学习问题：

- 回归：预测 `support_score`；
- 分类：预测 `stance`；
- 特征：人口统计、年龄工程、职业学历编码、文本嵌入、字段级文本特征；
- 模型：从简单线性模型到树模型和 boosting 模型；
- 解释：特征重要性、SHAP、reason 与模型解释的一致性。

## 6. 方法审计插曲：旧 FAISS 数据的问题

在重新跑实验之前，我们审计了已有 `.db` 中的旧结果。

发现：

- 旧 FAISS 主样本 `20260511_022424` 原始分布为支持 72.8%、中立 17.9%、反对 9.3%；
- 但该样本 10000 行中只有 4924 个唯一 `persona_id`；
- 某个 `persona_id` 重复出现 5077 次，且回答、分数和 stance 完全相同。

进一步定位：

- 原始 parquet 中该画像只出现 1 次，不是源数据重复；
- FAISS 返回无效索引 `-1` 时，代码没有过滤；
- pandas `df.iloc[-1]` 会取最后一行，于是大量无效结果被映射为同一个画像；
- 后续用 `persona_id -> response` 字典回填，又使这些重复行获得相同回答。

这个问题改变了我们的实验策略：

- 旧 FAISS 结果不能作为最终证据；
- 旧模型图、SHAP 图、学习曲线图需要隔离；
- 必须修复 FAISS 检索逻辑后重新做抽样策略实验。

旧图已移入：

- `output/figures/legacy_error_data/`

这部分在汇报中可以自然作为“数据管道审计发现”提到，但不作为主线开头。

## 7. Prompt × Sampling 交叉实验

修复 FAISS 后，我们设计了 `2 × 4 × 2000` 交叉实验。

抽样策略：

- `random`：随机抽样；
- `faiss`：修复后的语义检索抽样。

Prompt 版本：

- `P0_demo`：只给人口统计字段；
- `P1_career`：人口统计 + 职业画像、技能、职业目标；
- `P2_full_text`：完整 `persona_text`；
- `P3_compact_all`：全字段精简摘要。

共同设置：

- 模型：DeepSeek 官方 API `deepseek-v4-flash`；
- 温度：`0.2`；
- 输出：JSON，包含 `support_score`、`stance`、`reason`；
- 每组：2000 条。

输出记录：

- `output/sampling_prompt_experiment/summary.csv`
- `output/sampling_prompt_experiment/detailed_summary.csv`
- `output/sampling_prompt_experiment/experiment_record.md`

## 8. 交叉实验结果

使用 IFOP 三分类基准 `61 / 34 / 5` 后，综合排序为：

| rank | sampling | prompt | support | neutral | oppose | distribution_mae |
|---:|---|---|---:|---:|---:|---:|
| 1 | faiss | P2_full_text | 64.75 | 32.15 | 3.10 | 2.50 |
| 2 | random | P2_full_text | 68.25 | 29.70 | 2.05 | 4.83 |
| 3 | random | P3_compact_all | 70.55 | 27.80 | 1.65 | 6.37 |
| 4 | faiss | P3_compact_all | 71.65 | 27.35 | 1.00 | 7.10 |
| 5 | random | P1_career | 73.85 | 23.70 | 2.45 | 8.57 |

主要发现：

1. 完整画像 `P2_full_text` 明显优于其他 prompt。
2. 修复后的 FAISS + P2 最接近 IFOP 三分类分布。
3. 所有方案都低估反对类。
4. 全字段精简摘要 P3 并没有优于完整 persona_text，说明“字段越多、越结构化”不一定更好。
5. P0/P1 更容易推高支持率，说明只给人口统计或职业字段时，模型可能倾向把恢复 62 岁理解为普遍有利政策。

因此选择：

- `sampling = faiss`
- `prompt = P2_full_text`
- `temperature = 0.2`

作为 1w 主数据生成方案。

## 9. 1w 主数据构建

主实验配置：

- 修复后的 FAISS 抽样；
- P2_full_text prompt；
- temperature = 0.2；
- DeepSeek 官方 API；
- n = 10000。

结果：

| dataset | rows | support | neutral | oppose | note |
|---|---:|---:|---:|---:|---|
| full | 10000 | 66.79 | 31.01 | 2.20 | 原始完整结果 |
| clean | 9960 | 66.95 | 30.85 | 2.20 | 去掉 API 失败和 stance/score 不一致 |
| strict_json | 8990 | 71.71 | 26.02 | 2.27 | 只保留严格 JSON 成功 |

主分析选择 `clean` 数据。

原因：

- `clean` 保留 API 成功和标签一致样本；
- 不强制只保留严格 JSON；
- `strict_json` 会明显推高支持率，说明 JSON 遵循程度并非随机缺失。

主数据文件：

- `data/processed/modeling_samples/main_1w_faiss_p2_clean.parquet`

## 10. 主数据 EDA

新增脚本：

- `scripts/eda_main_dataset.py`

输出目录：

- `output/main_eda/`

主数据规模：

- 9960 行；
- 9960 个唯一 persona；
- 无重复画像。

整体分布：

| stance | n | pct | IFOP | gap |
|---|---:|---:|---:|---:|
| support | 6668 | 66.95 | 61.0 | +5.95 |
| neutral | 3073 | 30.85 | 34.0 | -3.15 |
| oppose | 219 | 2.20 | 5.0 | -2.80 |

年龄趋势：

| age_group | support_pct |
|---|---:|
| 18-24 | 54.32 |
| 25-34 | 61.38 |
| 35-44 | 63.14 |
| 45-54 | 65.66 |
| 55-61 | 71.38 |
| 62-64 | 73.80 |
| 65-74 | 76.68 |
| 75+ | 75.72 |

职业差异：

| occupation | support_pct |
|---|---:|
| Ouvriers | 92.78 |
| Retraités | 76.34 |
| Employés | 64.55 |
| Professions intermédiaires | 58.85 |
| Cadres et professions intellectuelles supérieures | 31.33 |

学历差异：

| education_level | support_pct |
|---|---:|
| Sans diplôme ou CEP | 83.63 |
| CAP ou BEP | 77.92 |
| Bac+5 ou plus | 32.13 |

reason 关键词：

| keyword_group | pct |
|---|---:|
| age_retirement | 61.18 |
| job_work | 47.29 |
| economy | 17.49 |
| family | 21.80 |
| uncertain | 14.19 |

相似度与态度：

- `similarity` 与 `support_score` 相关系数约为 -0.018。

解释：

- FAISS 负责筛出与政策议题语义相关的人群；
- 但检索相似度本身几乎不直接决定态度方向；
- 态度主要由画像内部的年龄、职业、学历等社会结构变量驱动。

## 11. 当前研究判断

目前最有研究前景的发现是：

> LLM 的政策态度模拟形成了清晰的社会分层结构，但仍系统性压缩反对类。

具体表现：

- 年龄越大，越支持恢复到 62 岁；
- 工人、低学历群体支持率极高；
- 高学历、高管/知识职业明显更中立或更不支持；
- reason 中大量提到年龄、退休、工作和职业压力；
- 但 oppose 类仅 2.2%，低于 IFOP 的 5%。

这说明 LLM 并不是随机输出，也不是只追随 FAISS similarity。它似乎根据社会身份线索生成态度，但对“进一步提高退休年龄”的少数派立场表达不足。

## 12. 下一步计划

## 12. 第一轮结构化特征建模

在主数据 EDA 之后，我们先不急着使用文本 embedding 或 PCA，而是做第一轮结构化特征基线。

目标：

> 仅靠年龄、性别、职业、学历、婚姻、家庭结构和地区等结构化变量，能否解释 LLM 生成的态度？

新增脚本：

- `scripts/model_structured_baselines.py`
- `scripts/summarize_structured_baselines.py`

输出目录：

- `output/structured_baselines/`

特征方案：

- `A0_label`：基础人口统计 + label/ordinal encoding；
- `A1_onehot`：基础人口统计 + one-hot encoding；
- `A2_age_engineered`：A1 + 年龄段、临近退休、退休年龄段等年龄工程；
- `A3_social_grouped`：A2 + 职业/学历社会分组与手工交互标记。

模型：

- 回归：Linear Regression、Ridge、Random Forest、XGBoost、LightGBM；
- 分类：Logistic Regression、Random Forest、XGBoost、LightGBM。

主要结果：

### 回归

最佳模型：

- `A3_social_grouped + XGBoost`
- Test R² = 0.1999
- MAE = 0.1001
- RMSE = 0.1259

回归前几名：

| approach | model | R² | MAE |
|---|---|---:|---:|
| A3_social_grouped | XGBoost | 0.1999 | 0.1001 |
| A1_onehot | XGBoost | 0.1978 | 0.1005 |
| A2_age_engineered | XGBoost | 0.1968 | 0.1006 |
| A0_label | XGBoost | 0.1891 | 0.1008 |
| A3_social_grouped | LightGBM | 0.1865 | 0.1003 |

解释：

- 结构化变量确实有解释力，R² 约 0.20；
- one-hot、年龄工程和社会分组相较 A0 有小幅提升；
- 但提升不大，说明 LLM 态度并非完全由几个结构化字段决定；
- 文本画像可能仍有增量空间，但需要实验证明。

### 分类

最佳 Macro F1：

- `A0_label + RandomForest`
- Accuracy = 0.6720
- Macro F1 = 0.4258
- support F1 = 0.7627
- neutral F1 = 0.5146
- oppose F1 = 0.0000

分类前几名：

| approach | model | accuracy | macro_f1 | oppose_f1 | neutral_f1 | support_f1 |
|---|---|---:|---:|---:|---:|---:|
| A0_label | RandomForest | 0.6720 | 0.4258 | 0.0000 | 0.5146 | 0.7627 |
| A3_social_grouped | XGBoost | 0.7135 | 0.4227 | 0.0000 | 0.4576 | 0.8105 |
| A3_social_grouped | RandomForest | 0.6332 | 0.4227 | 0.0274 | 0.5130 | 0.7276 |
| A2_age_engineered | RandomForest | 0.6365 | 0.4219 | 0.0294 | 0.5048 | 0.7316 |
| A1_onehot | XGBoost | 0.7142 | 0.4205 | 0.0000 | 0.4493 | 0.8121 |

解释：

- 分类模型能较好识别 support；
- neutral 有一定可学性；
- oppose 基本学不到，F1 接近 0；
- 这与 EDA 中发现的 oppose 类被压缩一致：主数据中 oppose 只有 2.2%，数量少且边界不清。

阶段性结论：

> 结构化变量能解释 LLM 态度生成的一部分社会分层，但无法解决 oppose 类稀缺与少数派立场压缩问题。

这给下一步实验提出两个方向：

1. 文本画像是否能补充结构化变量未解释的差异；
2. oppose 类困难究竟是样本不平衡问题，还是 LLM 本身很少表达“进一步提高退休年龄”的立场。

## 13. 下一步计划

下一步进入文本特征建模。

第一轮结构化基线已经说明：结构化变量有信号，但解释力有限。

接下来需要验证文本画像是否提供增量信息：

### 文本特征工程

- T0：结构化最佳方案 A3；
- T1：A3 + 完整 `persona_text` embedding；
- T2：A3 + 职业相关文本字段 embedding；
- T3：A3 + 生活方式文本字段 embedding；
- T4：A3 + 分字段 embedding + PCA。

### 下一轮要回答的问题

1. 文本 embedding 是否能显著提高 R²？
2. 职业相关文本是否比生活方式文本更有效？
3. PCA 是否保留了有用态度信号，还是压掉了弱政治信号？
4. 文本能否帮助识别 oppose，还是 oppose 困难主要来自标签生成机制？
5. reason 中提到的年龄/职业/经济线索，是否对应模型里的重要特征？

完成第一轮结构化建模后，再进入文本特征：

- 完整 `persona_text` embedding；
- 分字段 embedding；
- PCA / 非 PCA；
- 职业相关字段 vs 生活方式字段消融。

## 14. 第二轮文本特征消融建模

第一轮结构化建模说明：人口统计、职业、学历、家庭、地区和社会分组变量已经能解释一部分 LLM 态度生成逻辑，但解释力有限，且 oppose 类几乎无法识别。

因此第二轮进入文本特征消融，而不是简单把所有文本都塞进模型。

新增脚本：
- `scripts/model_text_feature_ablation.py`

输出目录：
- `output/text_feature_ablation/`

核心结果记录：
- `output/text_feature_ablation/text_feature_ablation_results.csv`
- `output/text_feature_ablation/text_feature_ablation_summary.md`
- `output/text_feature_ablation/classification_reports.json`

### 文本特征方案

| 方案 | 含义 | 特征维度 |
|---|---|---:|
| T0_structured_A3 | A3 结构化社会分组特征 | 155 |
| T1_full_persona_text | A3 + 完整 `persona_text` embedding | 539 |
| T2_career_text | A3 + 职业相关文本 embedding | 539 |
| T3_lifestyle_text | A3 + 生活方式文本 embedding | 539 |
| T4_field_pca | A3 + 分字段 embedding 后 PCA 压缩 | 315 |

文本向量模型：
- `all-MiniLM-L6-v2`

### 为什么这样设计

这轮消融对应五个问题：

1. T0：没有文本时，结构化变量能做到哪里？
2. T1：完整 persona 文本是否提供整体增量？
3. T2：职业、技能、职业目标等政策相关文本是否更有用？
4. T3：生活方式文本是否只是噪声，还是也携带社会身份信号？
5. T4：分字段 embedding + PCA 是否能保留有效信号并降低高维噪声？

### 回归结果

最佳回归模型：

- `T4_field_pca + XGBoost`
- Test R² = 0.2064
- MAE = 0.0994
- RMSE = 0.1254

回归前几名：

| feature_scheme | model | R² | MAE |
|---|---|---:|---:|
| T4_field_pca | XGBoost | 0.2064 | 0.0994 |
| T4_field_pca | Ridge | 0.2048 | 0.0993 |
| T2_career_text | XGBoost | 0.2029 | 0.1008 |
| T0_structured_A3 | XGBoost | 0.1999 | 0.1001 |
| T3_lifestyle_text | XGBoost | 0.1984 | 0.1007 |
| T1_full_persona_text | XGBoost | 0.1936 | 0.1012 |

解释：
- 文本特征确实有增量，但提升很小；
- 最佳方案 T4 将 R² 从结构化基线 0.1999 提升到 0.2064；
- 直接加入完整 `persona_text` 并不稳定，测试集效果反而低于结构化基线；
- 职业文本略有帮助，生活方式文本没有稳定增量；
- PCA 在这里不是机械降维，而是起到了降噪作用：分字段语义被保留，高维文本噪声被压缩。

### 分类结果

最佳 Macro F1：

- `T4_field_pca + XGBoost`
- Accuracy = 0.7149
- Macro F1 = 0.4277
- support F1 = 0.8100
- neutral F1 = 0.4732
- oppose F1 = 0.0000

分类前几名：

| feature_scheme | model | accuracy | macro_f1 | oppose_f1 | neutral_f1 | support_f1 |
|---|---|---:|---:|---:|---:|---:|
| T4_field_pca | XGBoost | 0.7149 | 0.4277 | 0.0000 | 0.4732 | 0.8100 |
| T1_full_persona_text | Logistic Regression | 0.5890 | 0.4258 | 0.0826 | 0.4874 | 0.7075 |
| T3_lifestyle_text | Logistic Regression | 0.5890 | 0.4257 | 0.0833 | 0.4859 | 0.7079 |
| T0_structured_A3 | XGBoost | 0.7135 | 0.4227 | 0.0000 | 0.4576 | 0.8105 |
| T2_career_text | XGBoost | 0.7095 | 0.4208 | 0.0000 | 0.4560 | 0.8063 |

解释：
- 文本特征没有从根本上解决 oppose 类识别问题；
- Logistic Regression 加入文本后能识别极少数 oppose，但牺牲了 accuracy 和整体稳定性；
- XGBoost 仍倾向于把 oppose 压到 support/neutral；
- 这说明 oppose 困难主要来自主数据标签生成机制和类别极端不平衡，而不是单纯特征工程不足。

阶段性结论：

> 结构化社会变量已经捕捉到 LLM 态度生成的大部分可学习信号；persona 文本有少量增量，最适合通过分字段 embedding + PCA 的方式压缩后进入模型。但 oppose 类困难主要不是特征工程问题，而是主数据标签生成机制与类别极端不平衡共同造成的。

这给 PPT 的表达提供了一个更完整的技术叙事：

1. 先用结构化变量建立可解释基线；
2. 再用文本 embedding 验证 persona 叙事是否有增量；
3. 通过 T0-T4 消融说明 PCA 的必要性；
4. 最后诚实指出模型发现：LLM 模拟能生成社会分层，但对“进一步提高退休年龄”的少数立场表达不足。

## 15. Oppose 少数类画像诊断

在文本特征消融后，最佳模型仍然无法稳定识别 oppose 类。因此先不急着继续换模型，而是回到数据本身，诊断 oppose 样本是否真的有清晰结构。

新增脚本：
- `scripts/analyze_oppose_profile.py`

输出目录：
- `output/oppose_profile/`

核心输出：
- `output/oppose_profile/oppose_profile_summary.md`
- `output/oppose_profile/oppose_cases.csv`
- `output/oppose_profile/profile_by_occupation.csv`
- `output/oppose_profile/profile_by_education_level.csv`
- `output/oppose_profile/profile_by_age_bin.csv`
- `output/oppose_profile/reason_theme_by_stance.csv`
- `output/oppose_profile/figures/`

### 基本规模

主数据 clean 版本共有 9960 条，其中：

| stance | n | pct | score_mean | score_median |
|---|---:|---:|---:|---:|
| support | 6668 | 66.95 | 0.7644 | 0.8000 |
| neutral | 3073 | 30.85 | 0.5153 | 0.5000 |
| oppose | 219 | 2.20 | 0.2594 | 0.3000 |

oppose 类只有 219 条，比例 2.20%。这意味着训练集里 oppose 数量非常有限，三分类模型很容易把它吞并到 support/neutral。

### 社会画像

oppose 并非完全随机，而是有一定社会分布。

职业上最富集的是：

| occupation | n | oppose_n | oppose_rate | oppose_lift |
|---|---:|---:|---:|---:|
| Cadres et professions intellectuelles supérieures | 600 | 47 | 7.83% | 3.56 |
| Artisans, commerçants, chefs d'entreprise | 246 | 7 | 2.85% | 1.29 |
| Professions intermédiaires | 1458 | 36 | 2.47% | 1.12 |
| Employés | 2522 | 60 | 2.38% | 1.08 |
| Autres sans activité professionnelle | 1181 | 27 | 2.29% | 1.04 |

学历上最富集的是：

| education_level | n | oppose_n | oppose_rate | oppose_lift |
|---|---:|---:|---:|---:|
| Bac+5 ou plus | 691 | 40 | 5.79% | 2.63 |
| Bac+2 | 1261 | 42 | 3.33% | 1.51 |
| Bac+3 ou Bac+4 | 868 | 24 | 2.76% | 1.26 |
| Baccalauréat | 2220 | 60 | 2.70% | 1.23 |

年龄上 45-61 岁略高，但不是压倒性信号：

| age_bin | n | oppose_n | oppose_rate | oppose_lift |
|---|---:|---:|---:|---:|
| 45-54 | 1657 | 45 | 2.72% | 1.24 |
| 55-61 | 1230 | 33 | 2.68% | 1.22 |
| 18-24 | 1053 | 26 | 2.47% | 1.12 |

因此，oppose 更像是高学历、干部/管理/专业阶层、偏审慎或财政稳定叙事下的小众立场，而不是一个单纯由年龄决定的群体。

### Reason 语义诊断

reason 主题分布显示 oppose 的语义与 neutral 有明显贴近，但并非完全无意义。

| theme | support | neutral | oppose |
|---|---:|---:|---:|
| stability/status quo | 28.07% | 18.06% | 69.41% |
| change aversion | 0.64% | 8.53% | 40.64% |
| fiscal sustainability | 4.63% | 3.32% | 16.89% |
| reform/policy | 0.87% | 5.82% | 23.29% |
| unknown | 3.31% | 25.25% | 6.85% |

oppose reason 高频片段包括：
- “倾向”
- “稳定”
- “维持”
- “现状”
- “谨慎”
- “财政”
- “工作”
- “纪律”

这说明当前 oppose 很多时候并不是清楚表达“进一步提高退休年龄”，而是在表达：

- 反对降低退休年龄；
- 倾向维持现状；
- 偏好政策稳定；
- 担心财政可持续；
- 对改革倒退或突变保持谨慎。

### 诊断结论

> oppose 不是一个只要换更强分类器就能自然恢复的隐藏类别。在当前 LLM 主数据中，它数量极少，集中于部分高学历/干部/审慎型群体，并且语义上经常贴近“维持现状”而不是明确的“进一步提高退休年龄”。

这解释了为什么：

- 直接三分类时 oppose F1 接近 0；
- 文本 embedding 也不能根本解决问题；
- 复杂模型更容易选择把 oppose 合并进 neutral 或 support，以获得更高总体 accuracy。

下一步优化不应只堆模型复杂度，而应讨论：

1. 是否需要重新定义标签边界，区分“维持 64 岁”和“进一步提高”；
2. 是否采用两阶段分类：support vs non-support，再 neutral vs oppose；
3. 是否构建 oppose-enriched diagnostic set，用于研究少数立场机制，而不是替代主数据；
4. 是否在 prompt 中更明确要求模型区分“维持 64 岁”和“进一步提高退休年龄”。

## 16. 三类 reason 词云对比

为了更直观展示三类立场背后的理由差异，新增 reason 词云分析。

新增脚本：
- `scripts/generate_reason_wordclouds.py`

新增依赖：
- `wordcloud`
- `jieba`

输出目录：
- `output/reason_wordclouds/`

核心图表：
- `output/reason_wordclouds/figures/reason_wordclouds_panel.png`
- `output/reason_wordclouds/figures/support_reason_wordcloud.png`
- `output/reason_wordclouds/figures/neutral_reason_wordcloud.png`
- `output/reason_wordclouds/figures/oppose_reason_wordcloud.png`

词频表：
- `output/reason_wordclouds/support_reason_word_frequency.csv`
- `output/reason_wordclouds/neutral_reason_word_frequency.csv`
- `output/reason_wordclouds/oppose_reason_word_frequency.csv`

### 词云观察

support 类高频词集中在：
- 退休
- 提前退休
- 工人
- 家庭
- 体力劳动
- 退休年龄
- 恢复
- 早退

neutral 类高频词集中在：
- 退休
- 退休年龄
- 谨慎
- 家庭
- 政治
- 平衡
- 年轻
- 信息不足

oppose 类高频词集中在：
- 稳定
- 谨慎
- 维持现状
- 纪律
- 保守
- 担忧
- 财政
- 务实
- 改变
- 现行

这进一步支持上一节的诊断：

> oppose 类的语义核心不是“强烈主张进一步提高退休年龄”，而是“维持现状、政策稳定、财政审慎、反对过快变化”。因此它和 neutral 的边界天然接近，直接三分类难度较高。

PPT 使用建议：
- 用三联图 `reason_wordclouds_panel.png` 展示三类 reason 的差异；
- 如需解释 oppose 识别困难，可单独放大 `oppose_reason_wordcloud.png`，突出“维持现状、稳定、谨慎、财政”。

## 17. Oppose 识别优化探索

在确认 oppose 类具有“少数类 + 语义贴近 neutral”的问题后，进一步探索模型层面的优化空间。

本轮不改主数据、不重跑 LLM，只在现有 1w clean 数据上尝试：

1. 加权多分类；
2. 概率阈值调优；
3. 两阶段分类；
4. 回归分数阈值分类。

新增脚本：
- `scripts/explore_oppose_optimization.py`

输出目录：
- `output/oppose_optimization/`

核心输出：
- `output/oppose_optimization/oppose_optimization_results.csv`
- `output/oppose_optimization/oppose_optimization_summary.md`
- `output/oppose_optimization/top_by_macro_f1.csv`
- `output/oppose_optimization/top_by_oppose_f1.csv`
- `output/oppose_optimization/reasonable_oppose_recovery.csv`
- `output/oppose_optimization/figures/oppose_recall_accuracy_tradeoff.png`
- `output/oppose_optimization/figures/best_macro_confusion_matrix.png`
- `output/oppose_optimization/figures/best_oppose_f1_confusion_matrix.png`

### 对照基线

此前最佳直接三分类模型：

- `T4_field_pca + XGBoost`
- Accuracy = 0.7149
- Macro F1 = 0.4277
- Oppose F1 = 0.0000
- Oppose Recall = 0.0000
- Neutral F1 = 0.4732
- Support F1 = 0.8100

它总体准确率较高，但完全无法识别 oppose。

### 最佳 Macro F1 方案

最佳方案来自两阶段分类：

1. 第一阶段：`LogisticRegression` 判断 support vs non-support；
2. 第二阶段：`XGBoost` 在 non-support 内部判断 neutral vs oppose；
3. 使用 T4 特征：结构化 A3 + 分字段文本 embedding PCA。

参数：
- support_threshold = 0.45
- oppose_threshold = 0.16
- stage2_oppose_multiplier = 1

测试集结果：

| model | accuracy | macro_f1 | oppose_precision | oppose_recall | oppose_f1 | neutral_f1 | support_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct T4 + XGBoost baseline | 0.7149 | 0.4277 | 0.0000 | 0.0000 | 0.0000 | 0.4732 | 0.8100 |
| two-stage best macro | 0.6727 | 0.4869 | 0.1923 | 0.1515 | 0.1695 | 0.5302 | 0.7610 |

解释：
- Macro F1 从 0.4277 提升到 0.4869；
- oppose F1 从 0 提升到 0.1695；
- 测试集 33 个 oppose 中识别出 5 个；
- 代价是 accuracy 从 0.7149 降到 0.6727，support F1 下降。

混淆矩阵显示：

| true \ pred | oppose | neutral | support |
|---|---:|---:|---:|
| oppose | 5 | 16 | 12 |
| neutral | 10 | 263 | 188 |
| support | 11 | 252 | 737 |

### 最佳 Oppose F1 方案

仍然是两阶段分类：

- `LogisticRegression + XGBoost`
- support_threshold = 0.55
- oppose_threshold = 0.16

测试集结果：

| accuracy | macro_f1 | oppose_precision | oppose_recall | oppose_f1 | neutral_f1 | support_f1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6265 | 0.4725 | 0.1667 | 0.1818 | 0.1739 | 0.5450 | 0.6986 |

解释：
- 识别出 6/33 个 oppose；
- oppose F1 略高，但 accuracy 和 support F1 下降更明显；
- 说明提高 oppose 识别会把更多 support/neutral 推向 non-support。

### 高召回折中方案

如果目标是“尽量找回 oppose”，可以提高第二阶段 oppose 权重：

- `LogisticRegression + XGBoost`
- support_threshold = 0.45
- oppose_threshold = 0.30
- stage2_oppose_multiplier = 20

测试集结果：

| accuracy | macro_f1 | oppose_precision | oppose_recall | oppose_f1 | neutral_f1 | support_f1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6392 | 0.4609 | 0.0903 | 0.4242 | 0.1489 | 0.4728 | 0.7610 |

解释：
- 能识别约 42.4% 的 oppose；
- 但 precision 只有 9.0%，误报明显；
- 适合做“少数类筛查”，不适合做最终三分类预测器。

### 阶段性判断

模型优化确实有效，但不是根本解决：

> 两阶段分类能让 oppose 从完全不可识别变成“可识别一小部分”，Macro F1 也有明显提升；但由于 oppose 样本少、语义边界贴近 neutral，提高 oppose recall 必然牺牲 accuracy 和 precision。

因此下一步的研究判断是：

1. 如果目标是宏观分布拟合，直接 T4 + XGBoost 仍然有较好 accuracy，但不能解释 oppose；
2. 如果目标是少数立场机制研究，两阶段分类更合适；
3. 如果目标是找出潜在 oppose 人群，高召回方案可以作为筛查器，但需要人工或二次模型复核；
4. 后续真正提升 oppose，可能需要改标签边界或构建 oppose-enriched diagnostic set，而不只是继续堆模型。

## 18. 模型解释与典型案例

在完成 oppose 优化探索后，进一步解释模型到底学到了什么。

本轮复现两个模型：

1. 直接三分类基线：`T4_field_pca + XGBoost`；
2. oppose-aware 两阶段模型：第一阶段 `LogisticRegression` 判断 support vs non-support，第二阶段 `XGBoost` 判断 neutral vs oppose。

新增脚本：
- `scripts/explain_oppose_models.py`

输出目录：
- `output/model_explanations/`

核心输出：
- `output/model_explanations/model_explanation_summary.md`
- `output/model_explanations/model_explanation_metrics.csv`
- `output/model_explanations/direct_multiclass_feature_importance.csv`
- `output/model_explanations/direct_multiclass_grouped_importance.csv`
- `output/model_explanations/stage2_neutral_vs_oppose_feature_importance.csv`
- `output/model_explanations/stage2_neutral_vs_oppose_grouped_importance.csv`
- `output/model_explanations/two_stage_representative_cases.csv`
- `output/model_explanations/cases_correct_oppose.csv`
- `output/model_explanations/cases_oppose_to_neutral.csv`
- `output/model_explanations/cases_oppose_to_support.csv`
- `output/model_explanations/cases_neutral_to_oppose.csv`

图表：
- `output/model_explanations/figures/direct_multiclass_top_features.png`
- `output/model_explanations/figures/direct_multiclass_grouped_importance.png`
- `output/model_explanations/figures/stage2_neutral_vs_oppose_top_features.png`
- `output/model_explanations/figures/stage2_neutral_vs_oppose_grouped_importance.png`
- `output/model_explanations/figures/direct_multiclass_confusion_matrix.png`
- `output/model_explanations/figures/two_stage_confusion_matrix.png`

### 指标复核

| model | accuracy | macro_f1 | oppose_precision | oppose_recall | oppose_f1 | neutral_f1 | support_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct T4 + XGBoost | 0.7149 | 0.4277 | 0.0000 | 0.0000 | 0.0000 | 0.4732 | 0.8100 |
| two-stage T4 | 0.6727 | 0.4869 | 0.1923 | 0.1515 | 0.1695 | 0.5302 | 0.7610 |

两阶段模型降低 accuracy，但提高 macro F1，并让 oppose 从完全无法识别变成能识别少量样本。

### 直接三分类模型学到什么

直接三分类模型的重要特征主要包括：

| feature | group | importance |
|---|---|---:|
| manual_or_lowedu | social_manual_high | 0.0694 |
| occupation_group_upper_cadre | occupation | 0.0208 |
| occupation_Retraités | occupation | 0.0171 |
| occupation_group_retired | occupation | 0.0168 |
| education_group_higher_high | education | 0.0168 |
| upper_or_highedu | social_manual_high | 0.0163 |
| education_level_Bac+5 ou plus | education | 0.0122 |
| pca_skills_and_expertise_01 | text_pca:skills_and_expertise | 0.0121 |
| occupation_Ouvriers | occupation | 0.0106 |
| occupation_Cadres et professions intellectuelles supérieures | occupation | 0.0105 |
| age_sq | age | 0.0100 |

解释：

- 直接模型主要学到的是宏观社会分层；
- 手工劳动/低学历、高学历/干部、退休者、年龄是核心变量；
- 这与 EDA 中发现的年龄、职业、学历差异一致；
- 因此模型并非完全黑箱乱猜，而是在复现 LLM 生成态度中的社会结构。

### 第二阶段 neutral-vs-oppose 学到什么

第二阶段中，分组重要性更集中于文本 PCA：

| group | share |
|---|---:|
| text_pca:cultural_background | 11.10% |
| text_pca:hobbies_and_interests | 10.25% |
| text_pca:persona | 10.18% |
| text_pca:professional_persona | 10.14% |
| text_pca:arts_persona | 9.80% |
| text_pca:travel_persona | 9.38% |
| text_pca:skills_and_expertise | 9.24% |
| text_pca:sports_persona | 9.14% |
| text_pca:career_goals_and_ambitions | 8.25% |
| text_pca:culinary_persona | 7.84% |
| occupation | 1.22% |
| education | 1.11% |

解释：

- neutral 和 oppose 的边界不像 support vs non-support 那样主要由职业、学历、年龄决定；
- 它更依赖 persona 文本中的叙事、文化背景、价值倾向、生活方式和职业语义；
- 这与 reason 词云一致：oppose 更多是“稳定、维持现状、谨慎、财政审慎”等语义，而非一个单纯 demographics 类别。

### 典型案例观察

正确识别的 oppose 案例中，reason 常见表达包括：

- “谨慎性格，反对仓促变革”
- “倾向维持现行退休制度”
- “尊重现有规则，反对随意更改”
- “倾向稳定和纪律，反对改变现状”
- “尊重现行制度，倾向保守”

被误判为 neutral 的 oppose 案例也常见：

- “重视责任与稳定，担忧财政风险”
- “务实倾向，担忧财政负担”
- “谨慎怀疑轻易承诺，倾向保守”
- “谨慎于改变，倾向维持现状”
- “职业稳定，倾向维持现状”
- “偏好秩序与严谨，倾向维持现状”

这些案例说明：

> 模型不是完全看不到 oppose 信号，而是当前 oppose 标签本身经常表达“维持现状/反对仓促变化/财政审慎”，这与 neutral 的语义边界非常接近。

因此，后续如果继续优化，需要把重点从“换更强模型”转向：

1. 标签语义边界重新设计；
2. prompt 中更明确地区分“维持 64 岁”和“进一步提高到 65/66+”；
3. 构建 oppose-enriched diagnostic set；
4. 对少数立场使用单独的二阶段解释模型。

## 19. 残差分析与分组评估

为了回答“模型评估结果到底算不算差”，进一步对最佳回归模型做残差分析。

模型：
- `T4_field_pca + XGBoostRegressor`

新增脚本：
- `scripts/analyze_residuals_and_groups.py`

输出目录：
- `output/residual_group_analysis/`

核心输出：
- `output/residual_group_analysis/residual_group_analysis_summary.md`
- `output/residual_group_analysis/overall_regression_metrics.csv`
- `output/residual_group_analysis/test_predictions_with_residuals.csv`
- `output/residual_group_analysis/group_metrics_by_stance.csv`
- `output/residual_group_analysis/group_metrics_by_occupation.csv`
- `output/residual_group_analysis/group_metrics_by_education_level.csv`
- `output/residual_group_analysis/score_bin_metrics.csv`
- `output/residual_group_analysis/reason_theme_residual_metrics.csv`
- `output/residual_group_analysis/largest_absolute_residuals.csv`

图表：
- `output/residual_group_analysis/figures/predicted_vs_true_score.png`
- `output/residual_group_analysis/figures/absolute_residual_by_stance.png`
- `output/residual_group_analysis/figures/mae_by_score_bin.png`
- `output/residual_group_analysis/figures/mae_by_occupation.png`
- `output/residual_group_analysis/figures/mae_by_education.png`

### 整体指标解释

| metric | value |
|---|---:|
| R² | 0.2064 |
| MAE, 0-1 scale | 0.0994 |
| MAE, 0-10 scale | 0.99 |
| RMSE, 0-10 scale | 1.25 |
| Bias, 0-10 scale | -0.01 |

解释：

- R² 只有约 0.20，说明模型只能解释一部分个体层面的分数方差；
- 但 MAE 约等于 0-10 分制上的 1 分，说明模型对大致分数水平并非完全无效；
- Bias 约 -0.01 分，说明整体上没有明显系统性高估或低估；
- 因此模型不是“整体预测崩坏”，而是“能抓住大方向，但解释不了细粒度个体差异”。

### 误差按 stance 拆解

| stance | n | MAE, 0-10 | Bias, 0-10 |
|---|---:|---:|---:|
| support | 1000 | 0.80 | -0.69 |
| neutral | 461 | 1.25 | +1.21 |
| oppose | 33 | 3.49 | +3.49 |

解释：

- support 误差最低，平均约 0.8 分；
- neutral 误差更高，平均约 1.25 分，并且被系统性高估；
- oppose 误差最高，平均约 3.49 分，几乎全部被高估；
- 这说明整体 R² 不高很大程度来自低分/少数类/边界区，而不是所有样本都难预测。

### 分数区间误差

| true score bin | n | MAE, 0-10 | Bias, 0-10 |
|---|---:|---:|---:|
| 0.25-0.35 | 27 | 3.36 | +3.36 |
| 0.35-0.45 | 18 | 2.29 | +2.29 |
| 0.85-1.00 | 68 | 1.63 | -1.63 |
| 0.45-0.55 | 357 | 1.34 | +1.34 |
| 0.75-0.85 | 483 | 0.92 | -0.91 |
| 0.55-0.65 | 86 | 0.64 | +0.47 |
| 0.65-0.75 | 449 | 0.54 | -0.32 |

解释：

- 模型预测明显向中间收缩；
- 低分样本被高估，高分样本被低估；
- 中高分区间 0.65-0.85 误差较低；
- 低分区间和中立边界区误差较高。

这解释了为什么 R² 不高：

> 模型能学习排序趋势，但不擅长预测极端分数，尤其是低分 oppose。

### 分组误差

职业维度中，误差较高的群体包括：

| occupation | n | MAE, 0-10 | Bias, 0-10 |
|---|---:|---:|---:|
| Artisans, commerçants, chefs d'entreprise | 27 | 1.28 | +0.24 |
| Autres sans activité professionnelle | 180 | 1.06 | -0.04 |
| Retraités | 414 | 1.06 | +0.07 |
| Professions intermédiaires | 235 | 1.03 | -0.03 |
| Employés | 403 | 1.01 | -0.09 |
| Cadres et professions intellectuelles supérieures | 89 | 0.95 | +0.13 |
| Ouvriers | 145 | 0.60 | -0.08 |

学历维度中，误差最高的是中高学历区间：

| education_level | n | MAE, 0-10 | Bias, 0-10 |
|---|---:|---:|---:|
| Baccalauréat | 341 | 1.13 | -0.01 |
| Bac+3 ou Bac+4 | 114 | 1.12 | -0.03 |
| Bac+2 | 211 | 1.02 | -0.06 |
| Bac+5 ou plus | 118 | 1.00 | +0.04 |
| Brevet | 93 | 0.97 | -0.03 |
| Sans diplôme ou CEP | 271 | 0.93 | -0.08 |
| CAP ou BEP | 346 | 0.86 | +0.07 |

解释：

- Ouvriers 误差最低，说明体力劳动/低学历群体的 attitude pattern 更稳定、更容易预测；
- 自雇/企业主、退休者、中间职业、无职业群体误差更高，说明这些群体内部异质性更强；
- 学历上并非学历越高越难，而是中高学历区间误差略高，可能因为其 reason 更复杂、更容易落在 neutral/oppose 边界。

### reason 主题误差

| theme | n | MAE, 0-10 | Bias, 0-10 |
|---|---:|---:|---:|
| uncertain | 252 | 1.26 | +1.06 |
| fiscal_caution | 39 | 1.23 | +0.57 |
| stability_status_quo | 380 | 1.01 | -0.06 |
| age_retirement | 975 | 0.85 | -0.45 |
| family_life | 474 | 0.85 | -0.57 |
| manual_work | 609 | 0.75 | -0.37 |

解释：

- “不确定/信息不足”主题误差最高；
- “财政审慎”主题误差也高，和 oppose/neutral 边界问题一致；
- “体力劳动”“年龄退休”“家庭生活”主题误差较低，说明这些理由更稳定、更容易被模型学习。

### 阶段性结论

模型效果不能简单评价为“差”。

更准确的判断是：

> 当前模型能够学习 LLM 态度生成中的宏观社会分层，对 support 和手工劳动/退休/家庭等稳定主题预测较好；但它会把预测压向中间，难以处理低分 oppose、中立边界和不确定/财政审慎类 reason。

因此：

1. 作为个体级精确预测模型，效果偏弱；
2. 作为社会分层解释模型，效果可用；
3. R² 低主要来自极端分数、少数类和边界样本；
4. MAE 约 1 分说明模型仍有实际解释价值；
5. 后续优化重点应是标签边界、少数类诊断和不确定样本处理，而不是单纯换更复杂模型。

## 20. 模型效果受限原因验证

为了避免只凭主观解释说明“为什么模型效果不高”，进一步做三组验证实验：

1. reason 上限实验：检验原始 persona 特征是否缺少直接态度线索；
2. 标签边界重标实验：检验 neutral / oppose 是否语义混叠；
3. LLM 标签稳定性复测：检验同一 persona 重复调用时标签是否稳定。

### 20.1 Reason 上限实验

新增脚本：
- `scripts/validate_model_limitations.py`

输出目录：
- `output/model_limitation_validation/`

实验口径：

- `persona_T4`：正式建模可用特征，即结构化 A3 + 分字段文本 PCA；
- `reason_embedding_only`：只用 LLM 输出后的 reason embedding；
- `persona_T4_plus_reason`：persona 特征 + reason embedding。

注意：

> reason 是 LLM 生成标签之后给出的解释，不能作为正式预测特征。这里使用 reason embedding 只是做上限/泄漏诊断：如果 reason 能显著预测标签，说明原始 persona 里缺少直接态度线索。

回归结果：

| feature_set | model | R² | MAE, 0-10 |
|---|---|---:|---:|
| persona_T4_plus_reason | XGBoostRegressor | 0.6245 | 0.62 |
| reason_embedding_only | XGBoostRegressor | 0.6183 | 0.63 |
| persona_T4_plus_reason | Ridge | 0.6064 | 0.66 |
| reason_embedding_only | Ridge | 0.5975 | 0.66 |
| persona_T4 | XGBoostRegressor | 0.2064 | 0.99 |
| persona_T4 | Ridge | 0.2048 | 0.99 |

分类结果：

| feature_set | model | accuracy | macro_f1 | oppose_f1 | neutral_f1 | support_f1 |
|---|---|---:|---:|---:|---:|---:|
| reason_embedding_only | XGBoostClassifier | 0.9143 | 0.7926 | 0.5600 | 0.8781 | 0.9398 |
| persona_T4_plus_reason | XGBoostClassifier | 0.9110 | 0.7765 | 0.5217 | 0.8686 | 0.9391 |
| reason_embedding_only | LogisticRegression | 0.8748 | 0.7230 | 0.3893 | 0.8609 | 0.9188 |
| persona_T4 | XGBoostClassifier | 0.7149 | 0.4277 | 0.0000 | 0.4732 | 0.8100 |

解释：

- reason embedding 单独就能把回归 R² 提高到约 0.62；
- reason embedding 单独能把分类 Macro F1 提高到约 0.79；
- oppose F1 也从 0 提高到 0.56；
- 这说明 LLM 输出的 reason 中包含大量直接态度线索，而这些线索在原始 persona 特征中并不充分。

阶段判断：

> 原始 persona 特征不是没有信号，但它缺少 LLM 最后打分时使用的直接理由信息。因此正式模型 R² 不高，并不完全是算法问题，而是输入信息上限问题。

### 20.2 标签边界重标实验

同一脚本中进一步做标签边界审计。

规则：

- 如果 oppose reason 中出现“维持、现状、现行、稳定、纪律、保守、谨慎、财政、可持续、反对降低、反对恢复”等语义；
- 且没有明确“进一步提高、提高退休年龄、延迟退休、65/66”等语义；
- 则将该 oppose 临时重标为 neutral。

审计结果：

| metric | value |
|---|---:|
| rows | 9960 |
| original_oppose | 219 |
| relabel_oppose_to_neutral | 186 |
| remaining_oppose | 33 |

主题重叠：

| stance | n | status_quo_or_fiscal_pct | explicit_raise_pct |
|---|---:|---:|---:|
| neutral | 3073 | 14.71% | 0.00% |
| oppose | 219 | 84.93% | 0.91% |
| support | 6668 | 2.88% | 0.15% |

解释：

- 219 个 oppose 中，186 个更像“维持现状/财政审慎/反对回到 62 岁”；
- 只有 33 个保留下来作为更接近“明确反对恢复 62 岁、可能支持更高退休年龄”的少数类；
- oppose 中明确“进一步提高”的表达只有 0.91%。

分类重标后：

| target | model | accuracy | macro_f1 | oppose_f1 | neutral_f1 | support_f1 |
|---|---|---:|---:|---:|---:|---:|
| original | XGBoostClassifier | 0.7149 | 0.4277 | 0.0000 | 0.4732 | 0.8100 |
| boundary_relabel | XGBoostClassifier | 0.7149 | 0.4287 | 0.0000 | 0.4823 | 0.8039 |

解释：

- 重标后 Macro F1 只小幅提升，因为 remaining oppose 只剩 33 条，分类仍极难；
- 但这个实验的核心价值不是提升指标，而是证明标签语义确实混叠；
- 当前 oppose 标签中，大多数并非明确“进一步提高退休年龄”，而是“反对降低/维持现状/财政审慎”。

阶段判断：

> neutral / oppose 的困难来自标签定义本身。当前三分类标签并不完全干净，尤其 oppose 与 neutral 在“维持 64 岁/反对回到 62 岁”的语义上高度重叠。

### 20.3 LLM 标签稳定性复测

新增脚本：
- `scripts/run_label_stability_retest.py`

辅助脚本：
- `scripts/smoke_test_deepseek.py`

输出目录：
- `output/label_stability_retest/`

实验口径：

- 从主数据中按原始 stance 分层抽样；
- support / neutral / oppose 各 100 个 persona，共 300 个；
- 每个 persona 重复调用 2 次；
- 共 600 次 DeepSeek API 调用；
- 模型：`deepseek-v4-flash`；
- temperature = 0.2；
- prompt 与主数据 P2_full_text 口径保持一致。

运行说明：

- 初次运行时 API 返回 `402 Insufficient Balance`，未产生有效结果；
- 充值后先做 1 条 smoke test，确认 `{"ok": true}`；
- 随后重跑 600 次，并使用逐条 checkpoint 断点续跑；
- 最终 600 次 API 请求成功，严格 JSON 解析成功 492 次。

稳定性指标：

| metric | value |
|---|---:|
| sample_personas | 300 |
| calls | 600 |
| api_success_rate | 1.0000 |
| parse_success_rate_all | 0.8200 |
| valid_parsed_calls | 492 |
| repeat_score_corr | 0.6654 |
| repeat_score_mean_abs_diff_0_10 | 1.03 |
| repeat_score_median_abs_diff_0_10 | 1.00 |
| repeat_stance_agreement | 0.7451 |
| repeat_stance_cohen_kappa | 0.5436 |
| agreement_with_original_main_stance | 0.6037 |
| score_mae_vs_original_main_0_10 | 1.39 |

按原始 stance 看，与主数据标签的一致性：

| original_stance | n valid calls | stance_agreement | score_mae_0_10 |
|---|---:|---:|---:|
| support | 177 | 0.9040 | 0.71 |
| neutral | 165 | 0.4364 | 1.54 |
| oppose | 150 | 0.4333 | 2.02 |

解释：

- 同一 persona 重复调用时，分数相关系数只有 0.665；
- 两次 stance 一致率约 74.5%，Cohen’s kappa 约 0.544，属于中等一致；
- 与原主数据标签相比，整体一致率约 60.4%；
- support 最稳定，一致率约 90.4%；
- neutral 和 oppose 都很不稳定，一致率只有约 43%；
- oppose 与原始分数平均相差约 2.02 分。

阶段判断：

> LLM 标签本身存在稳定性上限，尤其是 neutral 和 oppose。下游监督模型不可能稳定预测一个自身重复生成都不完全稳定的标签。

### 20.4 综合结论

三组验证实验共同说明：

1. **输入信息上限**：persona 特征不足以完全解释 LLM 最终判断；reason 中包含大量后验态度线索；
2. **标签边界问题**：多数 oppose 实际表达“维持现状/财政审慎/反对降低”，与 neutral 高度重叠；
3. **标签稳定性问题**：同一 persona 重复调用时，LLM 对 neutral/oppose 的判断并不稳定；
4. **算法不是主因**：当引入 reason 这种后验解释特征时，模型效果大幅提升，说明 XGBoost/Ridge/Logistic 并非完全学不动，而是正式输入与标签之间存在信息缺口和噪声。

因此，模型效果偏弱的系统原因可以更严谨地表述为：

> 当前任务的可预测性受限于 LLM 标签生成机制、输入特征信息上限、neutral/oppose 标签边界混叠和少数类不稳定性，而不是单纯由模型复杂度不足造成。

PPT 表达建议：

- 不要说“模型效果差”；
- 应说“模型揭示了 LLM 社会模拟的可学习部分与不可学习边界”；
- 用 reason 上限实验、标签边界审计、稳定性复测三组证据支撑这个判断。

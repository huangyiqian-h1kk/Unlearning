# Selective unlearning 完整实验计划（讨论稿，尚未实施）

目标：在相同forget强度下，比较对同患者非目标事实的保留，而不是仅增大R-any−F-any。
旧rich SFT注入975患者×3属性=2925事实；旧forget100的每位患者仅指定一个属性，
但原retain900与forget患者不重叠。因此旧结果没有直接回答同患者其余两个属性是否被误伤。

## Target矩阵

每个target是明确的 `(patient_id, attribute, canonical_value)` 事实集合；同一事实的
QA、cloze、background都是评测表达，不算新的target或三个独立样本。
所有方法、Gemma和Llama使用相同target文件及哈希；先锁定，再训练。

| ID | 要删除的事实 | Forget事实数 | 涉及患者 | 同患者非目标事实数 | 优先级 |
| --- | --- | ---: | ---: | ---: | --- |
| T0 | 原始forget100，保留原名单/属性分布 | 100 | 100 | 200 | 首批桥接 |
| T1 | 每患者删1属性，age/gender/diagnosis各100 | 300 | 300 | 600 | 核心主任务 |
| T2 | 每患者删2属性，三种属性对各50患者 | 300 | 150 | 150 | 主表 |
| T3 | 每患者删全部3属性 | 300 | 100 | 0 | 主表整实体对照 |
| T4 | 只删age | 300 | 300 | 600 | 属性分层 |
| T5 | 只删gender | 300 | 300 | 600 | 属性分层、二元答案特别报告 |
| T6 | 只删diagnosis | 300 | 300 | 600 | 属性分层 |
| T7 | 新的平衡k=1小规模请求：34/33/33 | 100 | 100 | 200 | 请求规模曲线 |
| T8 | k=1大规模请求：每属性200 | 600 | 600 | 1200 | 请求规模曲线 |

T7⊂T1⊂T8，在同一预先抽取的患者排列及每患者固定目标属性上嵌套。
T4/T5/T6使用T1的相同300位患者。T2/T3使用该患者池的平衡子集，固定F300比较k=1/2/3。
固定F时患者数随k变化，这是主实验设计的一部分；另做固定100位患者的k=1/2/3敏感性检查
（F100/200/300）以区分请求粒度和实体数量，但不把它放进第一批。

新划分初筛request_seed=101；关键任务补102/103。training_seed使用42/43/44，
两类seed分别记录，不把同一请求的三个training seed当成三个独立请求。
T0历史名单不重抽；T7是控制属性比例与嵌套规模的新名单，不冒充T0复现。

“同属性/同答案”“语义邻近diagnosis”“远离target”首先作为retain评测分层，
不立即再加一套未验证可构造的target分区。由训练前固定事实/模型定义邻近分组，
不得按哪个方法受损最少来挑“难例”。

## Retain保护信息及固定评测集合

所有训练retain总数固定900。R-same是同患者未删除事实；R-other是其他患者事实。
所有可用R-same均进入评测，即使未进入retain训练。每个target固定同一R-other评测集，
不同retain coverage不换评测题。嵌套的新请求共享600患者池外的900个R-other事实。

| Target | 主设定R-train组成 | R-same评测 |
| --- | --- | --- |
| T0/T7 | 200 same + 700 other | 全部200 |
| T1/T4/T5/T6 | 600 same + 300 other | 全部600 |
| T2 | 150 same + 750 other | 全部150 |
| T3 | 0 same + 900 other | N/A，不填0分 |
| T8 | 分层抽600 same + 300 other | 全部1200，分训练覆盖/未覆盖报告 |

T0另做覆盖消融：C0=0same+900other（旧训练数据，优先复用权重）、
C50=100same+800other、C100=200same+700other。用嵌套抽样和属性分层控制组成。
这里改变的是可用保护信息，不应称为“越来越selective的forget target”。
既有R900只用于T0桥接；新target的R-other需重新验证不包含任何forget事实。

用于所有任务的评测包括：F-any/F-gen、R-same-any/R-other-any、对应SFT下降、MMLU，
并分age/gender/diagnosis及共享value分层。性别二元答案的chance较高，不靠总均值掩盖。
主要比较matched-forgetting的retain曲线/Pareto；同时保留原始R-any−F-any排名。
T1等局部任务增加R-same-any−F-any，但不以单一差值替代双轴结果。
报告SFT原先答对的事实上的paired survival，避免“本来不知道”被算成遗忘。

增加同患者联合成功率：目标无法提取且其余目标外属性仍可提取；只在SFT本来全部答对的
患者子集上补充报告。背景题可能通过年龄/出生信息等直接推出目标时须标注可推导性，
不能把逻辑上不可能同时满足的要求当作训练失败。

训练表达、validation表达、最终test表达固定分离。bootstrap按患者聚类；最终结论同时
报告training-seed与request-seed波动。保留生成原文、invalid/review和解析器版本。

## 首批ConRep配置：按step125的R-any−F-any排名

排名来自现有完整原始任务结果，不意味着新任务的排名已知。
新噪声配置多为单seed，0.33pp这样的差别不能称为稳定显著优势。

| 模型 | 配置（共同前缀G256B32W2） | 旧F-any% | 旧R-any% | 旧gap pp | 理由 |
| --- | --- | ---: | ---: | ---: | --- |
| Gemma | P1C70V4N10 | 70.00 | 84.00 | 14.00 | 本模型any-gap第1 |
| Gemma | P4C70V4IB2N20 | 71.00 | 84.67 | 13.67 | 第2；跨模型共同强配置 |
| Gemma | P2C70V4I1 | 71.00 | 83.67 | 12.67 | 第3；较少positive的插入方案 |
| Gemma | P4（C70V4，无token noise） | 73.67 | 84.78 | 11.11 | 三seed均值；相同P4的必要机制对照 |
| Llama | P1C50V8N00 | 75.00 | 85.44 | 10.44 | 本模型any-gap第1 |
| Llama | P1C70V8N00 | 74.00 | 84.11 | 10.11 | 第2；forget强度端点 |
| Llama | P4C70V4IB2N20 | 75.00 | 85.00 | 10.00 | 第3；跨模型共同强配置 |
| Llama | P4（C70V4，无token noise） | 75.67 | 84.37 | 8.70 | 三seed均值；同P4机制对照 |

固定LR1e-5、LoRA rank256/alpha512/dropout.05、F/R/general batch8/32/32、W2。
IB2N20表示每个positive的插入数服从Binomial(2,.2)，平均0.4个token，
不是对20%的原token做插入。噪声仅用于retain positive分支，anchor/retain LM保持clean。
P是每个retain anchor的positive views数，不是新事实数。

这些配置第一批各跑seed42的T0-C100，共8个ConRep训练。
P4 clean在改变target/retain数据后是新的匹配对照；T0-C0的旧控制不重复训练。
Gemma旧Delta4优胜的P4I1先列二线验证，当前排序轴已改为any-gap，不能混用排名。

T0-C100同时跑NPO、RMU、FALCON、LUNAR、SAGO、ReLearn六种baseline，最多12个训练。
每个方法使用同一F/R数据与SFT，并沿用各自合理参数；不把ConRep LR/架构强套给其他方法。
ReLearn须先解决原来的approved-target覆盖缺口；失败不能填0、不能捏造replacement。
两模型首批共最多20个训练，而不是把旧top10全搬过来。

随后T1先跑每模型P4IB2N20、P4clean和该模型首批最好的专用配置（最多6个ConRep），
加同六baseline（12个），总计最多18个。T2–T8使用冻结后的两种ConRep候选和全部可运行baseline，
不在每个test任务重新选一套参数。

所有方法新增任务的训练预算按F样本暴露对齐，同时报告walltime/峰值显存/positive总token量。
ConRep F8、10 forget-equivalent sampled passes：F100→125步、F200→250步、
F300→375步、F600→750步。这里不是严格不放回epoch。
主表不能把F300仍截在125步，再与F100的10pass结果当作同等训练。
baseline若采样/批量机制不同，要按实际forget exposure核对，不能只对齐step数字。

## 后续grid顺序

先在T1 validation做，原任务的旧LR1e-5成绩不能替代新任务的LR1e-5控制。

1. LR：每模型P4IB2N20及模型专用优胜者，各测5e-6/1e-5/2e-5。
   已有新任务1e-5单元复用，因此新增最多8个训练。回答retain插入在Llama是否需要更合适LR。
2. Forget CL权重W：每模型保留一个候选及其已选LR，W=1/2/3/5。
   复用W2，新增6个；按相同F-any区间比较R-same，而不是只看步数终点。
3. Retain保护：每模型一个候选，specified LM weight=.5/1/2，新增4个。
   若仍出现局部误伤，再单独B32→B64，每模型增加1个；不与上述维度做全笛卡尔积。
4. 条件性机制消融：P4clean vs P4IB、retain CL关闭但retain LM不变、T0的C0/C50/C100。
   最后才决定是否在新任务扩C/V与插入强度；已有广泛positive/noise搜索不从头重复。

锁定方案后，在T1/T2/T3为两种ConRep候选及主要竞争baseline补training seeds43/44。
关键T1再补request_seed102/103；正式统计区分两种变异。其他baseline得到同样的
validation选择机会和可说明的调参预算，不只优化ConRep。

SFT原模型/SFT参考都保留。可增加fact-level retain-only再训练reference，但必须重新构建
语料：仅删除目标句不够，病例/note/dialogue/profile也可能泄露目标值；保留同患者其他事实。
旧8750行的“排除全部forget患者”retain-only模型不是这个局部任务的理想reference。
Qwen只在本轮rich SFT注入与utility通过后，作为第三模型复核已冻结的关键任务，不先铺满grid。

本文件为讨论方案：此次代码只实现Qwen rich SFT/reference evaluation，以上selective训练尚未提交。

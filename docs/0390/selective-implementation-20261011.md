# Selective unlearning extensive experiment：实施规格

版本：2026-10-11，替代上一版讨论稿。本次交付是完整实施规格、CPU target构造脚本和实际审计结果。
**训练实现已发布；启动命令见 [selective-training.md](selective-training.md)。** 尚未在开发环境提交 ABCI job；实际 8 GPU 检查由启动流程执行。本文件保留实验设计和分析目标，具体已实现行为及验证边界以运行说明为准。

## 1. 范围、证据与命名

研究问题：控制遗忘事实数与训练暴露量后，ConRep能否更好地保留同患者未请求删除的事实？
历史R900与F100的患者完全分离，历史R-any优势不能直接证明同实体选择性。

按上一版候选表理解“不要P4”：移除两模型的**纯`G256B32W2P4`**对照及其新任务重复；
保留`G256B32W2P4C70V4IB2N20`插入候选。后者的retain positives仍是4，
不能把此方案描述为“所有四-positive配置已排除”。

已审计历史145条ConRep run（Gemma71、Llama74，包括Llama positive-completion）、实际rich-SFT
注入的975患者×3属性=2925事实、旧F100/R900，以及frozen v5评测和baseline代码。
代码起点为公开提交`88b73509ab9d041c45a6c8248946e714f0418a86`。
本地审计记录保存输入SHA256、6候选完整配置、旧LR控制及实际构造结果；该原始记录不发布到公开仓库。
CPU审计不证明ABCI权重仍存在、tokenizer噪声检查通过或GPU恢复正确；这些仍是执行前关口。

## 2. 请求和保护信息

一个target是`(patient_id, attribute, canonical_value)`事实；QA/cloze等表达不是独立事实。
两模型及所有方法使用同一事实清单和hash。

| ID | 请求结构 | F事实 | F患者 | R-same评测 | R-train：same+other | 步数 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| T0-C100 | 历史F100，每患者1属性 | 100 | 100 | 200 | 200+700 | 125 |
| T1 | k=1，三属性各100 | 300 | 300 | 600 | 600+300 | 375 |
| T2 | k=2，三种属性对各50患者 | 300 | 150 | 150 | 150+750 | 375 |
| T3 | k=3，删除整患者 | 300 | 100 | N/A | 0+900 | 375 |
| T4 | 只删age | 300 | 300 | 600 | 600+300 | 375 |
| T5 | 只删gender | 300 | 300 | 600 | 600+300 | 375 |
| T6 | 只删diagnosis | 300 | 300 | 600 | 600+300 | 375 |
| T7 | k=1，三属性34/33/33 | 100 | 100 | 200 | 200+700 | 125 |
| T8 | k=1，三属性各200 | 600 | 600 | 1200 | 600+300 | 750 |
| E2 | 固定T7的100患者，k=2 | 200 | 100 | 100 | 100+800 | 250 |

固定算法：按SHA256(schema, request_seed, patient_id)排序975患者，前600位循环分配
age/gender/diagnosis目标属性。T7/T1/T8取前100/300/600位，保证`T7⊂T1⊂T8`。
T2取前150位，删除所分配属性及循环的下一个属性，恰好三种属性对各50位；
T3取前100位删除全部；T4/T5/T6使用T1同300患者；E2取前100位的两个属性。
具体哈希序列化/namespace以CPU脚本为准，不能另写一个不同排序器。

T1/T2/T3固定F300比较粒度，但患者数不同；T7/E2/T3固定同100患者比较k=1/2/3，
补足实体数混杂。E2是唯一新增辅助target，k=1/k=3复用T7/T3。

新任务共同R-other评测池为排列第601–900位的300患者×3属性=900事实，全部在600位请求池外。
训练other从此池取固定分层前缀。每属性内部哈希排序，属性轮转，截取前缀保证嵌套。
R-same评测全部；T8分别报告600个训练覆盖及600个未覆盖事实，R-other也报告covered/unseen。
训练覆盖事实的保留不能冒充未覆盖事实的保留。

T0沿用旧R900作为固定R-other评测池，另有同患者200事实。覆盖实验C0=0+900、C50=100+800、
C100=200+700，F及评测集合不变，same和other训练子集分别嵌套。
C0引用原训练文件字节和原行序，不重写ID后宣称相同训练。优先复用C0权重补评测；
C50各模型仅共同插入配置一条。覆盖实验改变保护信息，不是改变forget粒度。

主request seed101；102/103仅额外运行T1请求鲁棒性。training seed42/43/44独立记录。
CPU脚本能为三个request seed构造全部任务，但调度器不能据此把整个任务矩阵乘三。

## 3. 首批配置

以下历史数值是旧任务seed42、step125的百分比，不预言selective任务排名。

| 模型/短名 | 配置 | 旧F-any | 旧R-any | 旧gap pp |
| --- | --- | ---: | ---: | ---: |
| Gemma G1 | G256B32W2P1C70V4N10 | 70.00 | 84.00 | 14.00 |
| Gemma G2 | G256B32W2P4C70V4IB2N20 | 71.00 | 84.67 | 13.67 |
| Gemma G3 | G256B32W2P2C70V4I1 | 71.00 | 83.67 | 12.67 |
| Llama L1 | G256B32W2P1C50V8N00 | 75.00 | 85.44 | 10.44 |
| Llama L2 | G256B32W2P1C70V8N00 | 74.00 | 84.11 | 10.11 |
| Llama L3 | G256B32W2P4C70V4IB2N20 | 75.00 | 85.00 | 10.00 |

共同设置：LoRA r256/alpha512/dropout.05/bias none，七个attention/MLP projection；
LR1e-5，global F/R/general batch8/32/32，accum1，world_size8；AdamW、linear schedule、
warmup.05、weight decay0、clip1、bf16、gradient checkpointing、max_length512。
forget CL W2，specified/general CL与LM均1；temperature .08/.1/.1；forget negative views4、
retain negative views1。沿用历史inter-instance negatives、retain negative weight2、margin.1；
不新增stop-gradient。起点分别是现有Gemma/Llama rich full-SFT final，各请求独立从SFT开始。

- G1：forget70%/4 views；retain1 view，保护事实span后的eligible token按.1概率replacement。
- G2/L3：forget70%/4；retain4 views，独立插入数Binomial(2,.2)，0/1/2概率64%/32%/4%，均值.4。
- G3：forget70%/4；retain2 views，每view固定插入1 token。
- L1/L2：forget50%或70%/8；retain1 view、无token noise、保留dropout。

P是每anchor的positive数，不是新增事实数；IB不是按原token数20%插入。retain noise仅用于
positive，anchor/negative来源/retain LM沿用历史clean分支。克隆审计中的完整resolved config，
只覆盖task数据、预算、seed、输出和明确grid字段，不能凭名称重建隐藏默认值。

## 4. 去重与历史桥接

旧LR5e-6搭配G128/B16/W5，旧2e-5搭配G64/B16/W5，不是当前G256/B32/W2候选的匹配控制。
T1还改变了F/R数据，不能用旧任务1e-5结果替代T1控制。旧任务C/V/P/noise/insertion sweep不重跑。

新增ledger，不修改现有night identity/COMPLETE格式，保持旧campaign兼容：

| 身份 | 必须纳入 |
| --- | --- |
| training identity v2 | 起始权重/模型、实际有序F/R/general行、tokenizer/chat模板、method有效参数、LoRA/optimizer/总schedule、batch/accum/world size、seed、训练依赖代码及环境 |
| evaluation identity | checkpoint内容身份、task/partition/逐题内容顺序、生成参数、tokenizer、parser/聚合版本、MMLU输入 |

输出路径与日志时间不应改变训练身份；仅评测题目变化触发复评。需审计训练依赖代码等价，
不能因为整个旧source包含评测代码而误判必须重训，也不能忽略真正的trainer修改。
legacy导入记录旧identity、config SHA、checkpoint完整性和映射证明。
动作包括reuse_training_and_metrics、reuse_checkpoint_reevaluate、resume_incomplete、train_new、excluded，
逐格输出原因。权重缺失/来源未知记unavailable；ZIP有metrics不代表有模型，不能静默重跑。

旧baseline常止于120步：历史桥接使用common step120；不把ConRep125与baseline120称作严格预算对齐。
旧120步结束的schedule不能直接延至125。新任务按新完整预算从头初始化所有方法。

## 5. 执行顺序：先完整固定配置selective，再grid

2026-10-11顺序修订：此前“先T0/T1→grid→调参后扩展T2–T8”的顺序取消。
**所有T0–T8及E2主实验均先使用第3节的既往三套配置/模型，超参数冻结；不得先在T1调参，
也不得根据T1结果淘汰其中某一候选后才扩展其他任务。** 新任务只改F/R数据、按F暴露对应的步数和输出，
仍独立从相同SFT起点开始。这样主结果直接回答既往最优配置能否迁移到更selective的任务。

六baseline为NPO、RMU、FALCON、LUNAR、SAGO、ReLearn，同样先冻结历史resolved配置完成新任务。
下表数字是新增训练上限，按ledger再扣重复；SFT参考/复评/校准/恢复尝试不计为新训练。

| 顺序 | 内容 | 六baseline全可用 | ReLearn全受阻，其余五个 |
| --- | --- | ---: | ---: |
| A1 | T0-C100、T1；每模型既往3 ConRep+全部baseline，seed42 | 36 | 32 |
| A2 | T2–T8；沿用相同3 ConRep+全部baseline，不先grid | 126 | 112 |
| A3 | E2固定实体；同A2配置 | 18 | 16 |
| B1 | T1/T2/T3补training seeds43/44；历史预定2 ConRep+全部baseline | 96 | 84 |
| B2 | T1 request102/103；seed42；历史预定2 ConRep+SAGO/LUNAR | 16 | 16 |
| B3 | C50及retain-CL-off机制补充，各模型各1条 | 4 | 4 |
| C1 | 完成上述selective后，T1 ConRep grid | 18 | 18 |
| C2 | T1 baseline LR grid | 24 | 20 |
| **合计** | 先固定配置主实验，再鲁棒性/机制，最后grid | **338** | **302** |

固定配置主实验A共180条（五baseline时160条）；补齐B后296条（五baseline264条），
最后grid新增42条（五baseline38条）。相较旧322上限多16条，是T2–T8/E2保留历史第三候选，
不再通过selective结果先筛成两条。不是增加新的超参数网格。

A1→A2→A3→B→C为释放顺序；B内可按资源调度，但任何C类grid都不得抢在必需的A/B训练及评测
完成之前。pending/failed/blocked不当作完成，grid保持waiting_for_fixed_setup_selective。
ReLearn未通过augmentation的任务显式列出，不拖住其它A/B任务；五baseline列只是缺席情况下的
预算，不代表调度器可默默跳过它并宣布全部完成。

### B：配置由历史预定，不依赖T1晋级

seed重复固定Gemma G1+G2、Llama L1+L3，即历史各模型gap第一与共同插入配置；
它们直接继承原LR/W/retain LM等设置，不用新selective结果选择或更新参数。
request重复baseline固定SAGO与LUNAR，也不按新请求结果换方法。
C50用初始共同插入配置，匹配A的C100与历史C0；retain-CL-off同样使用历史共同插入配置，
仅指定CL置0，retain LM和general分支不变。E2和T7/T3均已在A用相同历史配置完成。
training seed与request seed并非3×3全交叉，不宣称9个独立重复。

### C1：主实验完成后的ConRep grid，新增18条

1. LR：固定Gemma G1/G2、Llama L1/L3，各测5e-6/1e-5/2e-5；复用A的1e-5，新增8。
   直接检验Llama插入是否受LR影响，不需要先从新任务三候选中筛选结构。
2. W：每模型从C1已有LR结果中选一条候选及LR，测W=1/2/3/5，复用W2，新增6。
3. Retain LM：沿同一候选及选定W，指定LM权重=.5/1/2，复用1，新增4。

C/V/P/插入剂量不再扩grid。若仍有明显R-same误伤，才考虑B32→B64各模型1条；不计入338，
不预先加入队列。纯P4不重新加入机制对照。grid所得配置只形成单独的T1调参补充结果，
**不会覆盖A/B主结果，也不会自动触发调好参数后重跑整个selective矩阵。**

### C2：baseline grid，新增24条

各模型各方法在T1调自己的原生LR×.5/1/2，1倍复用A，新增2×6×2=24。
LUNAR调baseline.fit_learning_rate，其它方法调实际optimizer使用字段；继承server resolved原值，
不强套ConRep LR或rank。公开双方搜索预算，固定配置主结果与调参结果分表。

### 仅C阶段适用的grid晋级规则

只用request101/training42/T1 validation的10-pass终点：先gap_same=R-same-any−F-any，
同分按R-other-any、MMLU、稳定config ID；保存promotion.json及输入/评测hash。
此规则只用于C1后续W/retain-LM实验以及grid报告，不决定A/B配置，不选掉已有主结果。
保留全部checkpoint和曲线，不另设未经约定的MMLU硬阈值，也不读取A/B的test分数做grid晋级。

## 6. 暴露预算、权重保存与容量

global F8、accum1，max_steps=10×|F|/8，即125/250/375/750；是sampled equivalent passes，
不是严格不放回epoch。记录draws、unique fact覆盖、重复、丢弃与重放批次。
baseline按实际forget fact暴露审计；ReLearn先抽fact再抽表达，避免augmentation多的fact权重更大。
LUNAR每fit update也记录实际F特征样本数。所有方法R-train集合900，baseline保留其原生batch/损失，
另外公开retain/general实际token量、GPU时间和峰值显存，不能只用同step声称计算量相同。

13个验证点按旧10,20,…120,125同比放大：F100每10+125，F200每20+250，F300每30+375，
F600每60+750。SFT为step0参考。保存所有验证点的可重建权重和原始预测；resume继续原总schedule。

新checkpoint支持LoRA与native baseline：固定base fingerprint+所有变化参数及必要buffers，
保留dtype。LUNAR/RMU/FALCON可存精确updated tensors，不能只记layer名称或漏bias。
重建模型tensor严格相等、固定样例logits一致后才启用delta格式。
optimizer/scheduler/每rank RNG/采样游标/scaler（如用）单独保存最近两个完整resume状态。
COMPLETE只在临时目录fsync并原子提交后发布；不改不删旧campaign权重。

当前145-run摘要不足以推算可靠总时长，不承诺一晚完成。338×13最多4394个训练checkpoint验证点，
另有复评/SFT。首批测实际速度与checkpoint大小后更新ETA与容量forecast；task启动前验证剩余容量
可容纳该task并额外保留100GB。不足记blocked_storage，不自动删历史权重。

## 7. selective-v1评测适配

### 无答案提示的主协议

旧background题提供另外两项属性；k=2/3会提示另一forget事实，R-same题也可能泄漏F值。
新主协议固定三个family：直接QA、cloze、另一种措辞QA，**只提供patient ID和待问attribute**。
每partition每family一题；validation/test用不同模板，但都每事实3题。F-gen是三题平均，
F-any是任一题正确；R-same/R-other同样定义。test不能增至六题后当成同难度any。

额外安全背景题只含该患者已保留且非本题查询的事实，遮蔽该请求所有F属性。没有安全背景时N/A，
不编造内容；这一项单列，不纳入主any。每题保存provided_fact_ids和queried_fact_id并检查泄漏。
不能用全局答案字符串黑名单误删其它患者共享答案的合法retain事实。

typed-answer解析沿用审计过的frozen v5语义，新版本改变构题/聚合schema。T0另保留historical-v5
桥接结果；同一旧checkpoint双协议分列，不能把协议变化解释为训练收益。保存raw generations和parser SHA。

### MCQ与空集合

旧四选一attribute MCQ无法从二元gender中取三个不同错误值，旧validator还拒绝空文件。
需显式支持可用性，T5不能因此整任务失败：

- gender二选一单列，chance.5；age/diagnosis合法四选一单列，chance.25，不混为未分层MCQ总均值。
- identifier题从固定全集构造且保证唯一正确答案，近义诊断按同一typed matcher判等；
  不足合法distractor记ineligible及原因，不复制选项。候选池不随retain coverage改变。
- identifier题保存所有参与关系的IDs/F-R归属，作为关系关联诊断，不冒充单事实删除主指标。
- T3的R-same计数0，JSON为null+not_applicable、表中N/A，不能填0%。

### 指标、统计与解析复核

逐checkpoint报告F-gen/F-any、R-same-gen/any、R-other-gen/any、相对同协议SFT下降、MMLU，
分attribute、train-covered/unseen、同attribute/value共享。语义相近diagnosis分组如增加，
需先固定外部定义/版本，不能根据方法受损程度挑选近邻。
继续给gap_same、gap_other、gap_union三种轴，明示union组成随task变化，不能独自支撑跨task结论；
T3没有gap_same。

补充SFT-known条件化遗忘/retain survival并报告模型和属性分母，零分母N/A。
同患者联合成功定义为所有目标均不可提取且全部非目标仍可提取，只在SFT原先全部正确患者上报告；
k=3单列整实体遗忘。gender仍有二元chance问题。跨模型可辅报共同SFT-known集合，不能将不同分母
直接解释为模型能力差异。

主图给全13点的F-any vs R-same/R-other及Pareto。matched-forgetting摘要预定F-any阈值
0–100每5pp，只在validation实际checkpoint中选满足阈值且retain最高的点，然后报告同一权重的test。
test未达到相同阈值要标明；无可比点N/A，不外推/假定单调。它与10-pass终点的配置晋级用途不同。
A/B的历史三ConRep主配置、两条seed重复配置和baseline在揭示test前列出端点/曲线checkpoint清单；
C阶段grid结果单独报告，落选配置不参与test调参，不覆盖固定配置主结果。
MMLU validation用于调参诊断，disjoint test用于最终报告，输入和direct-choice parser固定。

按患者paired cluster bootstrap，固定seed、10000重采样，不把三种表达当独立样本。
T1/T2/T3三training seed报告均值、SD、配对差异；request101/102/103另列。
主检验预定T1同患者retain，与冻结的最强竞争方法作配对比较；多个主比较Holm校正，
其它task/分层标探索性。不凭单seed的0.33pp声称显著。

LUNAR及所有方法同样报告invalid/refusal/extra-text比例；按parsed正确/错误/无效分层，
用固定哈希抽样复核原文和target。parser若修改，所有方法按新eval identity一起复评。

## 8. Baseline适配

保留server frozen parent中的实际实现/config，不能用仓库较旧的同名模块覆盖。
prepare逐方法导出resolved_config和覆盖diff；受控变更只有task数据、budget/seed、LR grid及必要
selective适配。保留原生目标函数、参数化，不把ConRep LoRA设置继承给全部baseline。

| 方法 | 适配/验收 |
| --- | --- |
| NPO | 固定同一SFT reference，原序列归约/retain loss/LoRA/beta沿用resolved配置 |
| RMU | 原steering/layer/权重，校准或随机向量seed独立记录 |
| FALCON | 每model/task重新MI校准；按SFT/train F-R/calibration seed/code hash缓存；Sophia/梯度投影状态可恢复 |
| LUNAR | direction/features仅来自真实train F/R，不读额外R-eval；独立fit LR；实际暴露对齐；保存updated projections及fit optimizer/RNG |
| SAGO | 保持all-reduce与非线性projection的原顺序，核查global batch |
| ReLearn | 新task生成/审核augmentation，每F fact全覆盖；与eval表达隔离；fact-first采样；不足记blocked_augmentation |

ReLearn旧augmentation不自动适用于新请求。固定teacher/checkpoint/templates/seed，按原过滤规则最多三轮
补齐缺失fact；仍不足阻塞该task并输出缺口，不捏造replacement、跳过fact或填0分。
其他任务继续，baseline表保留失败状态。校准/augmentation耗时单报，不能藏在训练时间之外。

## 9. 工程落点与契约

新profile selective-extensive，冻结parent代码再叠加新文件；不修改运行中的Qwen/旧campaign snapshot。
实际入口为 `scripts/abci/0390_selective.py`；模块对应关系及调用方式见运行说明。

| 模块 | 责任和输出 |
| --- | --- |
| src/experiments/selective_data.py | 从审计membership生成train/probes；泄漏/唯一答案/空集合审计 |
| src/experiments/selective_eval.py | frozen typed parser、relation聚合、F/same/other与SFT配对、raw预测恢复 |
| src/experiments/selective_grid.py | 分阶段DAG、独立training/eval identity、reuse ledger、晋级冻结 |
| src/experiments/selective_train.py | native方法数据/校准/augmentation/resume适配及config diff |
| scripts/abci/0390_prepare_selective.py | 只读验证parent/model/data/env，生成manifest/token audit/smoke输入 |
| scripts/abci/0390_selective.py | prepare/status/start/arm/resume/stop，共享限额与阶段释放 |
| scripts/abci/0390_selective_worker.sh | 单8GPU节点，校准/训练后分配GPU评测，信号与退出码 |
| scripts/abci/0390_collect_selective.py | hash核对、跨campaign去重、部分导出、分离prediction与summary |

task.json记录request seed、ordered fact IDs、输入hash、F/same/other/coverage、模板/partition、可用指标。
训练行包含relation_id、patient_id、attribute、typed target、canonical text与合法positive素材；
LUNAR/ReLearn仅消费显式train requests，不从评测题逆推训练数据。F与train-R、same与other零交集。

plan.json记录stage/task/config/training identity/dependencies/expected outputs；source.json冻结代码，
inputs.json冻结数据，model-assets.json记录起点，reuse-ledger逐格给理由，promotion.json冻结选择，
state.json保存可变执行状态。experiment ID例如llama8b-T1-rq101-conrep-L1-lr5e-6-w2-rlm1-ts42，
完整config hash仍是身份真值，不能只解析短ID。

## 10. 调度、恢复、导出

- 账户gcg51557、queue R9920261000、rt_HF、单节点8GPU、每PBS walltime12h。
  使用既有共享logs/0390/jobs/submit.lock和账号qstat检查；整个账号同时最多3个job/节点，计入旧job。
- Qwen运行占1个或已武装待提交时预留1个槽，selective最多2个；Qwen完成后最多3个。
  qstat失败不推断空闲，不中断已有job。worker串行领取任务，训练和评测在allocation内分时使用GPU。
- 提前prepare并arm，supervisor自动等槽位接续，幂等避免重复job；处理qsub成功但qstat短时不可见。
  全campaign运行到任务结束，不把总工作误限成一个10h/12h窗口。
- DAG：prepare→CPU审计→GPU smoke→SFT参考/校准→A1/A2/A3固定配置全部任务→B鲁棒性/机制→C grid。
  grid是独立后续队列，需确认必需的A/B训练和评测全部完成；blocked/failed/pending不能解除门控。
  ReLearn失败不拖住其它A/B任务，但其缺席必须显式报告，不能伪造完成状态。
- 证实的管理侧中断/节点故障/walltime在20分钟后自动续跑；手工qdel不自动复活，未知失败blocked。
  每rank恢复自己的RNG，optimizer边界响应暂停，继续同一schedule/采样游标；恢复失败不静默从零重训。
  训练完但eval未完只补eval。COMPLETE与identity检查先于加载。

status同时显示逻辑实验数、验证点数、复用/阻塞原因、PBS IDs、阶段/依赖、Qwen和其他job占槽、
磁盘forecast、已测GPU小时/ETA。不是仅看qstat数量猜还有多少实验。

默认summary.zip包含manifest/config/hashes/target统计/各checkpoint分层指标/seed统计/训练诊断/
reuse ledger/校准审计/日志摘要。显式另导predictions.zip，按checkpoint+eval identity去重，SFT参考一份。
模型权重、optimizer、code checkout、HF cache均不放结果包。部分完成仍可导出并保留pending/blocked；
输出尺寸分类与SHA256，避免此前携带全部预测而不清楚包大小。

## 11. 验收和实施次序

1. **已完成CPU审计：** T0三coverage+三个request seed×九新任务，共30个设计cell；真实2925事实
   的规模、T2属性对、嵌套、固定实体、R-other和F/R互斥全部通过；历史6候选与LR混杂已核对。
2. 实现selective-v1数据与题目；两模型实际tokenizer检查事实span/插入gap/合法token/512截断，
   不允许监督答案被截掉；gender二选一与T3空集通过，train/validation/test表达不重叠。
3. 实现复用与eval：旧权重historical-v5复现原分数，新协议分列；只变eval仅复评，变训练行序、
   seed/noise/schedule产生新training identity；缺权重/marker和重复指标拒绝误复用。
4. 各种native参数化做tiny-model两步中断恢复对照，再做实际8GPU两步smoke，验证global draws、
   DDP顺序及delta重建；不能用ConRep通过替代LUNAR/Sophia通过，ReLearn先有已审核素材。
5. 调度测试覆盖Qwen占槽、qstat失败、重复arm、qsub暂不可见、管理终止/手工取消、仅eval待恢复；
   部分导出按hash可重建报表，预测独立增量。
6. 全部关口通过后发布真实可复制的prepare/start/status/resume/export命令，释放A1，然后按既往固定配置接续A2/A3；不先释放grid。
   CPU设计器成功不等于GPU campaign已完成实施。

CPU审计器现已可运行，只需标准库、不提交job：

```bash
python scripts/abci/0390_plan_selective_extensive.py \
  --prepared /path/to/pmc-rich-all-ia \
  --request-seeds 101 102 103 \
  --output /path/to/new-selective-design-directory
```

prepared须含injection_facts.jsonl、forget.jsonl、retain.jsonl；输出目录必须不存在。
可选--history接本次汇总格式历史run JSON。输出仅为审计和membership manifest，明确trainer_ready=false，
不是可直接交给trainer的数据目录。

Qwen本轮仍先完成独立rich SFT/reference evaluation，通过注入和utility复核后再加入冻结关键任务，
不乘进338。fact-level retain-only重训reference也未计入：必须清除case/note/dialogue/profile中的
目标事实，不能把旧“排除整患者”的retain-only SFT冒充局部删除oracle。

# Selective unlearning target plan（已被新版实施规格替代）

完整方案见 [selective-implementation-20261011.md](selective-implementation-20261011.md)。

新版移除两模型的纯 P4 对照，保留 P4C70V4IB2N20 插入候选，明确 T0–T8、固定实体 E2、
历史完整配置去重、先完成全部既往固定配置 selective 再 grid 的顺序、评测防泄漏和调度恢复契约。

CPU target 构造已在实际 2925 事实、request seeds 101/102/103 上审计；
原始审计记录保留本地；公开仓库仅发布实施规格与可复现的 CPU 设计脚本。
GPU selective campaign 仍待按规格实施，本次没有提交 ABCI job。

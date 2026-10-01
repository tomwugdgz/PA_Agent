# Laya 分析接入指南（LAYA_INTEGRATION）

面向想把 PA_Agent 的 Laya 分析能力接入自己程序/系统的开发者。
读完这篇你能：在 10 分钟内跑通第一次推理，并知道怎么把「方向 + 价格计划」
接到你自己的交易或信号系统上。

---

## 一、Laya 是什么（30 秒版本）

[Convai Innovations 的 Laya](https://huggingface.co/convaiinnovations/laya) 是
**非自回归 System-1 决策模型**：不产出文字，只输出三种结构化原语——

| 原语 | 含义 | PA_Agent 用它问什么 |
|---|---|---|
| `choice` | 多选项 + 概率 + 置信度 | 周期结构（8 类）、交易方向（多/空/观望） |
| `noul` | 校准后的 P(true) | 信号是否有效、是否噪音区、多周期是否冲突 |
| `score` | 刻度打分 | （预留） |

PA_Agent 在其上的增量：**确定性定价层**——Laya 给方向和概率，
入场/止损/止盈由结构位 + ATR 规则算出（见第四节），模型不胡报价格。

## 二、环境准备

```bash
# Python 3.12（Windows 下 MT5 相关功能也依赖 3.12）
pip install laya torch    # torch 有 CUDA 就装 GPU 版，推理快 30 倍以上

# 下载 multilingual 权重（约 615 MB，走 hf-mirror，支持断点续传）
python tools/download_laya.py
# 默认落到 ~/laya-models/laya/multilingual/（可用 LayaSettings.model_dir 覆盖）
```

权重齐了就是**纯离线推理**，不再访问网络。

## 三、接入方式 A：一行管线（推荐）

```python
from pa_agent.config.settings import load_settings
from pa_agent.report.laya_pipeline import generate_report

settings = load_settings("config/settings.json")   # 你的配置路径
frame = ...   # 一个 KlineFrame（见下）

report = generate_report(frame, settings)

print(report.symbol, report.timeframe)
print("方向置信度:", report.prediction.answers["方向"].confidence)
plan = report.long_plan if report.prediction.answers["方向"].value == "long" else report.short_plan
if plan.actionable:
    print(f"入场 {plan.entry}  止损 {plan.stop}  止盈 {plan.target}  盈亏比 {plan.rr_ratio}")
```

### 你需要提供什么：KlineFrame

```python
from pa_agent.data.base import KlineBar, KlineFrame, IndicatorBundle
from pa_agent.data.snapshot import build_analysis_frame

# 方式一：只有原始 OHLC 列表（newest-first，最新收盘在前）
frame = build_analysis_frame(bars_raw, n=250, symbol="XAUUSD", timeframe="H1")
# 内部会自动算 EMA20/ATR14、剔除未收盘 K 线

# 方式二：自己已有指标
frame = KlineFrame(
    symbol="XAUUSD", timeframe="H1",
    bars=(...),                                  # newest-first，bars[0] 最新收盘
    indicators=IndicatorBundle(ema20=(...), atr14=(...)),
    snapshot_ts_local_ms=0,
)
```

**数据新鲜度铁律**：每次分析都应基于最新收盘 K 线重建 frame——
PA_Agent GUI 的做法是传入 `frame_provider` 回调（见 `gui/laya_report_dialog.py`），
每按一次「刷新」就用此刻数据重推理。不要缓存旧 frame 反复出报告。

## 四、接入方式 B：低层 API（自己解释答案）

```python
from pa_agent.ai.laya_engine import LayaEngine
from pa_agent.ai.laya_schema import build_state, build_questions
from pa_agent.ai.market_features import compute_simple_market_features

engine = LayaEngine.get(model_dir="~/laya-models/laya", subfolder="multilingual",
                        device="auto")
agent = engine.ensure_loaded()          # 幂等加载，进程内缓存

features = compute_simple_market_features(frame)   # 结构位/突破质量/铁丝网等
state = build_state(symbol="XAUUSD", timeframe="H1",
                    features=features, atr=1.2, close=3850.0)
raw = engine.predict(state, build_questions(), lang="zh")
# raw = {"answers": {"结构": {"type": "choice", "choice": "trending_tr",
#                              "probabilities": {...}, "confidence": 0.42, ...},
#                     "方向": {...}, "信号有效": {"type": "noul", "noul": 0.78, ...}}}
```

**三个硬约束**（不遵守结果不可用）：

1. **state ≤ 700 字符**：模型上下文 1024 token，中文 ≈ 1 字 1 token；
   超长会从尾部的术语表开始截断。
2. **choice 选项标签 ≤ 8 个汉字**：所有问题共享 256 token 的选项区。
3. **零样本置信度偏低**（结构题 ~0.17）：用 `min_confidence`（默认 0.35）
   过滤，低于门槛的答案只做参考；要可靠输出请走第六节微调。

## 五、报告对象结构（接入方消费什么）

```
LayaReport
├── symbol / timeframe / generated_at
├── close, atr
├── supports / resistances          # 前 3 个结构位
├── prediction
│   ├── answers: dict[qid, LayaAnswer]   # kind/value/confidence/probabilities/reliable
│   ├── device / latency_ms / load_ms
│   └── usage
├── long_plan / short_plan          # PricePlan（frozen dataclass）
│   ├── direction, actionable, reason
│   ├── entry / stop / target / risk_per_unit / rr_ratio
│   ├── entry_fallback / stop_fallback / target_fallback   # 哪些值是 ATR 兜底
│   └── notes                       # 风险提示（R 过小 / MM 过远等）
├── state_text                      # 喂给模型的完整状态串（审计用）
└── errors                          # 解析失败的问题列表
```

定价口径（`report/laya_pricing.py`，确定性、可复算）：

- 入场 = 结构位 + 0.1×ATR（无结构位则市价 − 1×ATR 兜底）
- 止损 = 结构失效位 − max(0.25, 1.0)×ATR
- 目标 = Measured Move 投影，否则 entry + 2×R
- R < 0.2×ATR 或目标 > 10×ATR 时自动在 `notes` 里写风险提示

## 六、微调数据闭环（让置信度可用）

### 自动收集（零成本）

每次推理自动落一份标注（`laya.collect_annotations: true`）：

```
experience/laya_annotations/YYYY-MM.jsonl
  每行 = {ts_ms, state, questions, answers, context, label: null}
```

### 半自动标注（GUI）

Laya 报告窗口点击「进入标注模式」→ 为每个问题选择你认为正确的答案 → 「保存标注」。
标签自动回填到 JSONL 的 `label` 字段。

### 一键微调脚本

```bash
# 先装依赖
pip install transformers peft accelerate datasets

# 干跑：只统计样本数
python tools/finetune_laya.py --dry-run

# 真正训练（至少 500 条已标注样本）
python tools/finetune_laya.py --epochs 3 --lr 2e-5 --batch-size 4

# 输出权重到 ~/laya-models/laya-finetuned/
# 在 config/settings.json 中把 laya.model_dir 指向该目录即可启用新模型
```

### 经验库自动裁决

`experience_writer.py` 会同时用后续行情自动裁决每笔建议的盈亏，
success/failure 案例会注入后续分析提示词，形成**无需人工干预的弱监督信号**。

当积累了足够多自动裁决的样本后，可批量回填 label：

```python
from pa_agent.ai.laya_annotation import relabel_line
relabel_line(Path("experience/laya_annotations/2026-10.jsonl"), line_no=42,
             label={"方向": "long", "结构": "trending_tr"})
```

## 七、运行记录（审计）

所有 Laya 报告自动写入五层流水账（append-only，按月归档）：

```
experience/journal/YYYY-MM/data.jsonl      # 输入快照
experience/journal/YYYY-MM/decision.jsonl  # 答题摘要 + 双向计划
experience/journal/YYYY-MM/risk.jsonl      # 风险告警
```

跨层用 `symbol + timeframe + ts_ms` 对齐。

## 八、GUI 集成点

| 入口 | 位置 | 说明 |
|---|---|---|
| 「Laya 报告」按钮 | 主窗口工具栏 | 弹出报告窗，支持 MD/HTML 导出 |
| 「刷新（最新数据）」 | 报告窗 | 用此刻最新收盘 K 线重推理 |
| `laya_latest.json` | `logs/` | 最近一次报告的价格计划快照，供下单面板「导入分析」 |

## 九、已知边界（诚实声明）

- 回测/定价是**确定性规则**，不是 Laya 本身——Laya 出概率，规则出价格
- 零样本置信度未校准，微调前只当排序参考
- 模型权重与推理遵循 Laya 上游的 Apache 2.0 许可

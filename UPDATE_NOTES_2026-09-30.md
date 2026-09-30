# 更新说明 · 2026-09-30

> 本次更新三大块：**关键 Bug 修复**、**Laya 本地报告**、**MT5 交易/回测/MQL5 板块**。
> 全部功能已在 Windows + Python 3.12.10 实机验证。

---

## 一、关键修复

### 1. Stage1 报 400 错误（max_tokens 超上限）
- **现象**：`Error code: 400 - Range of max_tokens should be [1, 131072]`，1h 周期分析大面积失败
- **根因**：`deepseek_client.py` 的 `_DEEPSEEK_MAX_OUTPUT_TOKENS` 原为 393216，取小后仍发出 384000，超 DeepSeek 实盘上限
- **修复**：改为 **131072**，并在未知网关兜底分支增加 DeepSeek 家族模型限流
- **影响**：1h 周期成功率从 25%（1/4）恢复

## 二、Laya 本地报告（新功能）

工具栏新增 **「Laya 报告」** 按钮：本地判别模型 [Laya](https://huggingface.co/convaiinnovations/laya)（System 1 决策模型，非生成式 LLM）分析当前图表，**完全离线、不消耗 API token**。

- 输出：方向/周期结构/信号有效性等 **6 项概率判断**（choice/score/noul 三种原语）
- 输出：**买入/卖出参考价**——全部由确定性代码计算（结构位优先 + ATR 兜底），Laya 不产出文本与价格
- GPU 加速：torch 2.14.0+cu130 实测单次推理 **p50 56ms**（CPU 为 2039ms，快 36 倍）
- 微调数据自动收集：每次调用追加样本到 `experience/laya_annotations/YYYY-MM.jsonl`（零样本结构题置信度仅 ~0.17，必须微调后才能作实盘依据；报告内已用 ⚠️ 标注低置信项）
- 新文件：`ai/laya_schema.py`（问题集+状态构造，选项 ≤8 字适配 head 256 token 预算）、`ai/laya_engine.py`（单例懒加载+离线守卫）、`ai/laya_annotation.py`、`report/`（定价+报告渲染）、`gui/laya_report_dialog.py`、`tools/download_laya.py`（hf-mirror 断点续传下载器）

详见 **LAYA_README.md**。

## 三、MT5 交易 · 回测 · MQL5 板块（新功能）

工具栏新增 **「MT5 交易」** 按钮，三个标签页：

1. **下单（当前决策）**：Python 直连 MT5（`order_send`），把最近一次 AI 决策（方向/限价/突破/市价、入场/止损/止盈）一键发单。三道安全门：`mt5trading.enabled` 默认关闭 → GUI 确认弹窗 → 点差/手数/止损方向机器校验。无任何自动循环下单路径。
2. **生成 MQL5**：一键生成两份 `.mq5`（输出到 `logs/mql5/`）：
   - **决策执行 EA**：把当前 AI 决策写成 input 参数默认值，编译挂图即自动执行这一单（挂单带有效期）
   - **策略回测 EA**：完整确定性策略（回踩限价+突破市价+失效位止损+R 倍数目标+超时平仓+铁丝网过滤），供 MT5 策略测试器与实盘
   - ⚠️ MQL5 未经 MetaEditor 编译验证，首次使用请先编译（F7）+ 策略测试器跑通
3. **回测**：程序内确定性回测引擎，直接用当前图表数据逐根走查与 Laya 报告/AI 决策同源的定价规则；输出胜率/盈利因子/总收益 R/最大回撤/逐笔清单（涨红跌绿）。740 根 0.2 秒。
   - 口径声明：回测回放的是确定性规则，不是 DeepSeek LLM 本身（LLM 无法逐根调用）

新文件：`mt5trading/`（mt5_bridge / backtest / mq5_generator）、`gui/mt5_panel_dialog.py`。详见 **MT5_README.md**。

## 四、经验库学习闭环（补齐写侧）

- 新增 `records/experience_writer.py`：分析成功后自动落 pending 案例 → 用后续 K 线按「TP 先到还是 SL 先到」自动判定 success/failure 并晋升入库 → `ExperienceReader` 注入后续分析提示词
- 判定规则确定性可复核：同根双触发保守记亏；50 根超时按 MFE≥1R 裁决；无单案例永不晋升
- `prompt.experience_max_entries` 0→**5**（经验注入正式开启）

## 五、其他

- `运行智能体.bat`：GBK 启动脚本（显式 Python 3.12 路径）
- `PA_Agent_分析准确度优化方案.md/.html`：准确度评估报告（实测基线+根因矩阵+量化验收指标）
- `.gitignore`：排除含明文密钥的 `config/settings.json.bak*`（上游遗留文件已移出跟踪）与本地大文件
- tvDatafeed 2.1.0 websocket headers bug 修复说明见部署文档（site-packages 层面）

## 已知限制

- Laya 零样本置信度不可信（尤其结构题），微调前仅供决策辅助
- MQL5 模板未经 MetaEditor 编译验证
- 回测成本为固定点数估算，未建模浮动点差与滑点
- 经验库闭环刚启用，案例库从零积累

## 运行环境

Windows 10/11 · Python 3.12（MetaTrader5 wheel 仅支持到 3.12）· torch 2.14.0+cu130（可选，纯 CPU 亦可运行）· Laya 权重 615MB（`tools/download_laya.py` 下载，国内走 hf-mirror）

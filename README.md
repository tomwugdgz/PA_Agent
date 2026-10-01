# PA Agent — AI K线分析辅助工具（桌面端）

**交流 QQ 群：1063897401**

---

面向主观交易者的 **价格行为（Price Action）** AI 辅助决策工具。从 **MT5 / TradingView / yfinance / AkShare** 读取 K 线，将结构化 K 线数据与预计算特征送入大模型做**两阶段分析**（市场诊断 → 交易决策），**不是**截图识图，**不连接券商、不执行下单**。

---

## 主要功能

- 📈 **多数据源**：MT5（Windows）、TradingView（全平台）、yfinance（期货/加密货币）、AkShare（A 股）
- 🧠 **两阶段 AI 分析**：市场诊断 → 策略路由 → 交易决策（限价/突破/市价或不下单）
- 🔄 **增量分析与持续跟踪**：新增 K 线时复用上次结论；开启 `keep_analysis` 后新 K 线收盘自动触发新一轮分析
- 🌳 **决策树可视化**：赛博科幻风格可交互流程图，自动播放闸门→策略路径动画
- 🔮 **未来走势预期**：AI 预测下一根 K 线方向和下一个市场周期位置
- 💬 **分析后自由追问**：完整对话会话管理器，实时推理流 + Token 进度条，对话历史持久化
- 📚 **经验库**：按周期位置检索历史案例供分析参考
- 📝 **完整落盘**：Prompt、原始响应、诊断/决策 JSON、Token 用量、追问记录
- 🛡️ **可配置校验体系**：JSON 校验、一致性检查、语义校验、截断修复、失败自动重试
- 🔒 **API Key** 本地加密存储

---

## 环境要求

| 项目     | 要求                                                                    |
| -------- | ----------------------------------------------------------------------- |
| 操作系统 | Windows 10 / 11（主支持）、macOS 12+（TradingView 数据源）              |
| Python   | 3.11+                                                                    |
| 数据源   | MT5 / TradingView / yfinance / AkShare **至少配置一种**                  |
| 网络     | 可访问所配置的 AI API（如 DeepSeek、PackyAPI 等）                        |

---

## 快速开始

直接在系统中安装（推荐部署在本机）：

```cmd
pip install -e .
python -m pa_agent.main
```

首次启动后在**设置**中填写 **Base URL**、**模型名** 与 **API Key**。

> 如需隔离环境也可创建虚拟环境：`python -m venv .venv` 后激活再 `pip install -e .`。

**安装内容**：PyQt6（GUI 框架）+ pyqtgraph（K 线图表绘图）+ numpy/pandas（数据处理）+ openai（AI API 客户端）+ json 校验、模型定义等全套依赖。

> 若需运行测试（pytest）或代码格式化（ruff/black），额外安装：`pip install -e ".[dev]"`。


```cmd
# 1. 安装 uv（仅需一次）
pip install uv
# 或官方脚本：curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 首次运行或依赖变更时，make 自动创建 .venv 并同步依赖
make uv-run

# 3. 之后每次启动
make uv-run
# 或手动：uv run python -m pa_agent.main
```

> 运行测试：`make uv-test`，代码检查：`make uv-lint`。

---

## 详细说明

完整操作界面说明见 [`PA_Agent使用文档.md`](PA_Agent使用文档.md)，配置字段说明见 [`config/README.md`](config/README.md)。

---

**免责声明**：本工具仅供学习与研究，不构成投资建议。交易有风险，决策后果自负。

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) 发布。

---

## 群友反馈榜单

感谢群友的使用反馈与鼓励，以下为群友评价截图（按时间从早到晚排列）：

<p align="center">
  <img src="qunyou/BD58CB2D6E4F45CC17CF832C506A982C.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/653EC872A0D6883A34B7B37B692C8B1D.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260619-205140.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260619-235505.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260620-150714.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260620-150833.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260620-220824.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260623-125929.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/91003065F07407E92B50964AE7F8A944.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260624-191001.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260628-014043.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260628-213700.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260629-163821.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ20260701-212522.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/BB4AE8110A7011426BD29D5CE8B5F73B.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/F383D366F2254692418DB18AAA617ACE.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/AD48DF6289CB6A9D51FE0B8EE2EC38C2.jpg" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/F61C8DCDB67924B64B33403D20047E0B.png" alt="群友反馈" width="480" />
</p>
<p align="center">
  <img src="qunyou/QQ_1783089951396.png" alt="群友反馈" width="480" />
</p>

---

## 联系与支持

分支主页：[duckwolf.cn](https://duckwolf.cn)

如果你觉得这个程序对你有帮助，欢迎访问上方网站了解更多项目或留言交流。

---

## 版权与归属

本项目基于上游开源仓库 **rosemarycox5334/PA_Agent**（Apache-2.0 许可）二次开发。
原始代码、架构设计、Prompt 工程及两阶段分析框架的版权归属于原作者。
本分支在保留原有版权声明的前提下，新增了 Laya 本地推理、MT5 交易集成、五层运行记录等功能模块。

使用本项目时请遵守以下条款：
- 商业使用需获得原作者授权
- 不得用于违法活动或高风险金融操作
- 作者不对因使用本软件造成的任何损失负责

---

## 近期更新（Tom / duckwolf 分支）

### v1.42 — 2026-10-01
- **Laya 报告链路打通**：每次打开报告自动从 1 秒级刷新循环取最新 K 线重建帧；报告窗新增「刷新（最新数据）」按钮
- **下单表单化**：方向/类型下拉 + 入场/止损/止盈/手数可编辑；「① 导入分析」优先 AI 决策、否则读 Laya 价格计划；「② 复制下单信息」进剪贴板
- **10015 Invalid price 提前拦截**：价格规格化、限价/突破单挂错边中文提示、SL/TP 距离不足 stops_level 拦截
- **品种选择根修**：`symbol_select(visible=True)` 解决 MetaQuotes-Demo 不在市场报价的品种报 -1 问题
- **连接自愈**：`terminal_info()` 验活 + 失效重连，解决 -4: Terminal: Not found
- **默认手数 1 手**：`mt5trading.default_lot: 0.01 → 1.0`
- **数据根数 250**：回测/调参样本更足；回测/调参按钮加 K 线预检查弹窗

### v1.41 — 2026-09-30
- **五层运行记录（journal）**：data/decision/risk/backtest/execution/tuning 六层 JSONL 按月归档
- **贪婪调参器**：以回测净收益 R 为目标做坐标下降，GUI 一键应用最优参数
- **RefreshLoop 退避溢出修复**：指数封顶防止长时间挂机后线程崩溃

### v1.40 — 2026-09-30
- **Laya 本地报告引擎**：离线权重推理（GPU 36× 加速），结构化决策 + 确定性定价
- **MT5 交易板块**：下单 / 生成 MQL5 / 确定性回测四合一面板
- **经验库闭环**：pending 案例自动裁决 → success/failure 注入后续 prompt

---

## 未来优化路线图

### 短期（1–3 个月）
- [ ] **多周期共振分析**：同时看 H1/H4/D1 三个周期，提高信号可靠性
- [ ] **风险管理系统**：根据账户净值动态调整手数（凯利公式/固定分数法）
- [ ] **回测可视化**：权益曲线、逐笔盈亏分布、最大回撤区间标注
- [ ] **Laya 微调工具链**：半自动标注界面 + 一键启动微调脚本
- [ ] **更多经纪商适配**：除了 MetaQuotes-Demo，测试国内常见 MT5 经纪商

### 中期（3–6 个月）
- [ ] **多资产组合**：同时监控多个品种，自动选出最优交易机会
- [ ] **新闻情绪过滤**：接入财经日历，重大事件前后暂停交易
- [ ] **云端同步**：分析结果/对话历史跨设备同步（可选）
- [ ] **插件系统**：用户可自定义指标/策略并热加载

### 长期愿景
- 打造完全本地化、隐私优先、不依赖云 API 的专业级 AI 交易辅助平台
- 所有模型权重、推理逻辑、定价规则完全透明可审计
- 社区共建：欢迎提交 PR、分享策略、反馈 bug

---

## 如何贡献

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交改动（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 发起 Pull Request

对于重大改动，请先开 Issue 讨论后再提交 PR。

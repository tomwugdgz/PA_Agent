# tools/ 脚本说明

PA_Agent 的命令行工具集。每个脚本都可独立运行，多数支持 `--help`。

---

## ⚠️ 先看这里：用哪个 Python 解释器

**本项目必须用 `Python312/python.exe`，不能用 3.13。**

| 解释器 | laya | laya.calibrate | MetaTrader5 | PyQt6 | 能否用 |
|---|---|---|---|---|---|
| `C:\Users\wolf2\AppData\Local\Programs\Python\Python312\python.exe` | 0.3.22 | ✅ | ✅ | ✅ | ✅ **用这个** |
| `C:\Users\wolf2\AppData\Local\Programs\Python\Python313\python.exe` | 0.3.4 | ❌ | ✅ | ❌ | ❌ |
| `~\.workbuddy\binaries\python\...`（托管隔离环境） | 未装 | ❌ | ❌ | ❌ | ❌ |

**为什么**：`pyproject.toml` 写的是 `requires-python = ">=3.11"`，但 `laya` 这个包的
**校准模块 `laya.calibrate` 只存在于 0.3.x 较新版本，且实际只装在 3.12 环境里**。
在 3.13 上会静默降级成 0.3.4（PyPI 上的旧版），该版本**根本没有 `calibrate` 子模块**，
`fit_temperatures` / `records_from_labeled` 全部不可用——置信度校准会直接失效。

验证当前环境是否正确：

```bash
PY="C:/Users/wolf2/AppData/Local/Programs/Python/Python312/python.exe"
"$PY" -c "import laya, importlib.util as u; \
print(laya.__version__, 'calibrate:', bool(u.find_spec('laya.calibrate')))"
# 期望输出形如：0.3.22 calibrate: True
```

---

## Laya 决策板块（`laya_*` / `weak_label` / `calibrate_laya`）

Laya 是 ConvAI 开源的**非自回归 System 1 决策模型**：输入一段市场状态 + 一组类型化
问题（choice / score / noul），一次前向输出全部答案与概率，**不生成文本、无幻觉**。

### 官方能力边界（重要，别被误导）

- **官方 PyPI 包没有训练/微调模块**。`pip show -f laya` 只有推理 + 校准组件
  （`agent` / `calibrate` / `router` / `serve` / `evals` / `tl_kernels`），**无 `laya.train`**。
- 官方唯一的"学习"路径是**置信度温度校准**（Confidence Calibration）：
  用自己标注的数据拟合一个温度标量，把过度自信的原始分布压平，使报出的置信度
  接近真实命中率。**本机 CPU 秒级完成，不写模型权重，不需要 GPU。**
- 官方微调 notebook（`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`，
  Kaggle 双 T4 约 4–5 小时）存在于 GitHub 仓库但**未随 PyPI 包发布**，需要
  自己从 `github.com/NandhaKishorM/laya` 取。适合要真正改模型权重的场景，
  不是本项目的起步路径。

因此本项目的路线是：**零样本跑通链路 → 积累标注 → 温度校准 → 攒够再考虑微调**。

### 数据流

```
生成报告 ──► append_sample() 写 JSONL（state + questions + answers）
                            │
                ┌───────────┴───────────┐
         人工标注（报告页标注模式）   tools/weak_label.py（弱监督自动标注）
         source="manual_label"      source="weak_supervision"
                └───────────┬───────────┘
                            ▼
                   to_laya_pairs()  →  (state, questions, one-hot targets)
                            ▼
              tools/calibrate_laya.py  →  calibration.json
                            ▼
        推理时 laya.load(..., calibration="calibration.json")
                            ▼
               报告页「置信度可信度」卡片（分级 + 行动建议）
```

### `tools/weak_label.py` — 弱监督自动标注

用**决策时刻之后的真实走势**反推正确答案，零操作积累标注数据。

判定规则：

- **Kaufman 效率比** `ER = |净变动| / Σ|逐根变动|`（0=纯震荡，1=单边直线）
  - `ER ≥ 0.55` → `周期结构 = trending_tr`（趋势）
  - `ER ≤ 0.25` → `周期结构 = trading_range`（震荡）
  - `0.25 ~ 0.55` 灰区 → **不写结构标签**（宁缺勿滥）
- **震荡市一律给 `neutral` 方向** —— 否则会产出"看涨 + 区间震荡"这类自相矛盾标签
- 方向阈值：净涨跌 > `0.8 × ATR` 才算有效方向
- **缺口守卫** `max_gap_atr=1.5`：决策价必须与未来第一根 K 线连续，
  错位样本直接拒掉（见下方「已知阻断项」）

```bash
"$PY" tools/weak_label.py --status        # 看待标注数量与分布
"$PY" tools/weak_label.py --dry-run      # 干跑，只报告会标多少
"$PY" tools/weak_label.py                # 实际写入
"$PY" tools/weak_label.py --reset-weak   # 清掉所有弱监督标签（人工标签保留）
```

### `tools/calibrate_laya.py` — 置信度温度校准

```bash
"$PY" tools/calibrate_laya.py --status   # 不加载模型即可看状态
"$PY" tools/calibrate_laya.py            # 跑校准
```

产出 `calibration.json`（挂到推理链）+ `calibration_meta.json`（ECE 前后对比）。

官方门槛常量（`laya/calibrate.py`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `MIN_TYPE_N` | 10 | 类型级温度的最小样本数 |
| `MIN_BUCKET_N` | 2000 | 分桶温度的最小样本数 |
| `ECE_HOLDOUT_FRAC` | 0.2 | 留出评估比例 |

**样本量说明**：类型级温度几十条样本就能拟合；**分桶温度需要数千条**，
达不到时官方会把桶排除评估、`n_eval=0` → ECE 显示 `nan`，但**不影响校准本身生效**。

### `tools/laya_train_smoke.py` — 离线冒烟（12 项）

不连 MT5、不加载模型权重，纯离线校验判定逻辑与数据构造。改完 Laya 板块先跑它。

---

## 时间对齐：MT5 服务器时区 + 已收盘 K 线（关键坑）

弱监督要拿「决策时刻之后的走势」，必须先把样本 `ts_ms` 对齐到正确的 K 线。
这里有**两个独立的错位源**，任一没修都会让缺口守卫把好样本全判成「报价漂移」：

###坑 1：MT5 K 线时间是**服务器时区**的墙钟

`copy_rates_range` 返回的 `rates["time"]` 是交易所服务器时区的 epoch 墙钟，
而样本 `ts_ms` 是本机墙钟（UTC+8）。实测本机 MT5 服务器偏 **-3 小时**。

不校正就会整体错位 3 小时——1m 品种直接错 180 根。

**修法**（`tools/weak_label.py::_server_tz_offset_ms`）：用
`symbol_info().time`（最后一根已收盘 1m K线的开盘时间）减本机 UTC epoch，
再**四舍五入到整小时**以消掉「距该根开盘的时长」（实测稳定在 0.1 分钟内）。
拉取时减去该偏移，`ts_open` 再加回来，两边就同一坐标系了。

### 坑 2：样本 `close` 取自**已收盘的上一根**，不是决策时刻实时价

样本 `ts_ms` 是推理时刻，落在当根 K 线之内；而 `close` 来自**已收盘**的那根。
两者可差 1~5 根（实测 USDJPY 15m 样本精确命中在时间锚点**前 5 根**）。

**修法**（`pa_agent/ai/laya_weaklabel.py::_future_closes_for`）：

1. 先用 `ts_ms` 定位时间锚点；
2. 在锚点附近 ±8 根内找**精确匹配**（相对误差 1e-9）；
3. 精确匹配找不到，才退回 ATR 容差匹配。

⚠️ **顺序不能反**：ATR 容差（如 2×ATR）往往比相邻根价差还宽，
会先把时间锚点本身误判为命中，取到错误位置的未来序列。

⚠️ **不能用 `[-lookahead:]`**：必须从锚点**紧接**往后取 N 根。
`[-lookahead:]` 从序列末尾截断，锚点后若还有 100+ 根（样本老、窗口大），
会跳过紧邻走势——这正是之前 105 条里104 条被误判为漂移的**主因**。

### 坑 3：短周期噪音远大于长周期

1m 品种相邻根价差可达 4~5 ATR（15m+ 基本为 0）。统一用 1.5 ATR 的缺口守卫会误杀。
故按周期自适应：1m 用 6.0 ATR、5m 用 4.0 ATR、其余用 1.5 ATR。

### 修复效果（实测）

| | 修复前 | 修复后 |
|---|---|---|
| 可标注样本 | 1 / 110 | **91 / 111** |
| 被误判为漂移 | 104 | 14（真实漂移） |

---

## 已知阻断项：XAUUSD 等品种的报价源变更

**现象**：XAUUSD 样本 `close=2001.06`，当前 MT5 报价 `4187.95`，缺口 **486 倍 ATR**。

**原因**：经纪商更换了 XAUUSD 的报价源 / 合约规格，历史价格**在 MT5 里根本不存在**
（用±14 小时全偏移扫描 + 全历史精确匹配均找不到该价格）。这不是时间对齐问题，
无法用代码修复。

当前 110 条里剩 14 条属于此类，另有 6 条是「样本太新、行情还没走出来」。
**这两类都等一等就会自然好转**：XAUUSD 需人工补标，新样本等 20 根后自动可标。

---

## 校准现状

已用 91 条样本完成首次校准（`--force` 越过 120 建议下限）：

```
类型级温度 = [1.94, 1.0, 5.0]     # 顺序对应 [choice, ?, noul]
分桶温度   = 0 个（需每桶 ≥2000 条，当前 223 条）
ECE        = NaN（留出集为空，样本量低于官方门槛）
```

**方向题温度 1.94** 说明零样本输出确实过度自信（把分布压平 1.94 倍后
报出的置信度才接近真实命中率）——这印证了作者"near chance"的说法。
校准文件落在 `laya-models/calibration.json`，引擎会自动挂载。

想看到 ECE 数字需积累数千条样本，届时重跑本脚本即可自动评估。

---

## 其它脚本

| 脚本 | 用途 |
|---|---|
| `finetune_laya.py` | 导出微调所需的训练数据（Notebook 格式） |
| `opt_smoke.py` | 通用优化项离线冒烟（4 项） |

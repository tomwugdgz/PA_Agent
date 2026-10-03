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

## ⚠️ 已知阻断项：经纪商报价漂移

**现象**：历史样本记录的 `close` 与当前 MT5 行情不再同源。

实测最差样本：`XAUUSD 1h close=2001.06` 但当前行情首根 `=4153.15`，
缺口达 **478 倍 ATR**（另有 USDJPY 样本 158.314 vs 现价 157.427）。

**原因**：经纪商更换了报价源 / 合约规格，历史价格无法回溯对齐。
**这不是代码 bug**，弱监督拿不到样本当时的行情，任何时间对齐都会错位。

**当前状态**：110 条历史样本中 104 条因此无法自动标注（由缺口守卫拦下，
不写入脏标签），仅 1 条成功。

**可行路径**（按推荐顺序）：

1. **从现在开始持续积累** —— 每次生成报告自动写样本，等行情走出来后再回溯标注
   （这是唯一能同时绕开漂移又保证质量的路）
2. **人工标注** —— 报告页「进入标注模式」，对关键结论手工打标
3. 攒够 ≥120 条后跑 `tools/calibrate_laya.py`

---

## 其它脚本

| 脚本 | 用途 |
|---|---|
| `finetune_laya.py` | 导出微调所需的训练数据（Notebook 格式） |
| `opt_smoke.py` | 通用优化项离线冒烟（4 项） |

# Laya 本地报告功能说明

> 2026-09-30 新增。与「提交分析」（DeepSeek 两阶段）**完全独立**的另一条分析路径。

## 一、这是什么

工具栏新增 **「Laya 报告」** 按钮：用本地判别模型 [Laya](https://huggingface.co/convaiinnovations/laya)
（System 1 决策模型，非生成式 LLM）分析当前图表，输出：

1. **方向/结构/信号有效性** 的概率与校准置信度；
2. **买入/卖出参考价**——全部由本地确定性代码计算（结构位优先 + ATR 兜底），不依赖任何 AI 生成。

## 二、与「提交分析」的区别

| | 提交分析（DeepSeek） | Laya 报告 |
|---|---|---|
| 模型 | 生成式 LLM（云端 API） | 判别模型（本地离线推理） |
| 输出 | 自然语言诊断 + 结构化决策 | 概率（choice/score/noul）+ 确定性价格 |
| 依赖 | 需要 API Key、联网 | 完全离线，不消耗 token |
| 记录 | 写 records/pending | 不写 records；标注样本写 experience/laya_annotations/ |

## 三、首次使用前的准备（已由 AI 完成）

- ✅ Python 3.12 装 `laya 0.3.22` + `torch`（当前 CPU 版；CUDA 版安装中，装完自动生效）
- ✅ 权重下载到 `%USERPROFILE%\laya-models\laya\multilingual\`（615 MB，脚本 `tools/download_laya.py`）
- ✅ `config/settings.json` 写入 `laya` 配置段

若换机器重装，只需：

```bat
python tools\download_laya.py
```

## 四、配置项（config/settings.json → laya）

| 键 | 默认 | 说明 |
|---|---|---|
| enabled | true | 关闭后按钮提示已禁用 |
| model_dir | %USERPROFILE%\laya-models\laya | 权重根目录 |
| subfolder | multilingual | 中文档（"" 为英文档） |
| device | auto | auto=有 CUDA 用 GPU，否则 CPU |
| min_confidence | 0.35 | 低于此置信度标 ⚠️，相关价格仅供参考 |
| collect_annotations | true | 每次调用落一条微调标注样本 |
| entry_offset_atr | 0.10 | 挂单价相对结构位的 ATR 偏移 |
| stop_buffer_atr | 0.25 | 止损相对结构失效位的 ATR 缓冲 |
| fallback_target_r | 2.0 | 无 MM 投影时的兜底目标（R 倍数） |
| fallback_stop_atr | 1.0 | 无结构位时的 ATR 兜底止损 |

## 五、定价口径（混合：结构优先 + ATR 兜底）

做多（做空镜像）：

- **挂单价** = 最近支撑 + `entry_offset_atr` × ATR
- **止损** = 做多结构失效位（supports[0]）− `stop_buffer_atr` × ATR
- **目标** = measured_move 投影；无投影时 = 挂单价 + `fallback_target_r` × R
- 结构位缺失 → 该项自动回落到 ATR 倍数，报告里标「兜底」
- 支撑/失效位都缺失 → 该方向不出价，给原因
- R < 0.2×ATR 或目标 > 10×ATR → 报告自动附「盈亏比虚高」提示

## 六、微调闭环（解决「结构题置信度 0.166」的唯一途径）

1. **自动收集**：每生成一次报告，`experience/laya_annotations/YYYY-MM.jsonl` 追加一条
   `{state, questions, answers, context, label: null}`；
2. **人工/回测补标签**：`pa_agent.ai.laya_annotation.relabel_line(path, 行号, label={...})`
   把正确答案填进 `label`；
3. 凑够样本（建议 ≥500 条/类）后用 Laya 官方微调流程训练，替换权重目录即可。

## 七、已知限制

- **零样本置信度不可信**（尤其结构题），报告已用 ⚠️ 标注，微调前只当参考；
- Laya **不产出文本与价格**——报告正文与价格全部来自本地代码，这是设计而非缺陷；
- state 上限 1024 token（约 700 中文字），超出截断（报告第四节展示喂入原文，可复核）；
- 选项 >20 会崩（官方限制），当前最大 8 项，安全。

## 八、排障

| 现象 | 处理 |
|---|---|
| 「权重目录不存在/缺少文件」 | 重跑 `tools/download_laya.py` |
| 报告卡在加载 | 首次 CPU 加载约 15-20 s；之后再点只需推理 2-3 s |
| 推理失败：CUDA | settings.json 把 `laya.device` 改回 `"cpu"` |
| `HF_HUB_OFFLINE` 相关报错 | 权重未就绪即触发；补齐权重即可（程序强制离线，防代理卡死） |

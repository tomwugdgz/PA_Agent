# -*- coding: utf-8 -*-
"""Laya 微调标注数据集（JSONL）。

用途
----
Laya zero-shot 结构题置信度极低（实测 0.166），必须微调。
Laya 官方微调吃的是 `(state, questions, expected_answers)` 三元组，
因此本模块在**每次生成 Laya 报告时**顺手把三元组原样落盘成 JSONL：

    experience/laya_annotations/2026-09.jsonl
    {"ts_ms":..., "state":{...}, "questions":{...}, "answers":{...},
     "context":{symbol,timeframe,close,atr}, "label":null, "source":"auto"}

两条产线：
1. **自动**（本文件）：无需人工，先把输入输出原样留住。
2. **人工补标签**：`relabel_line()` 供后续脚本把 `label` 填上
   （如：结构题正确答案、方向题正确答案），形成监督样本。

设计取舍：按月分文件，避免单文件无限膨胀；append-only，绝不改写历史行。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def annotations_dir(experience_dir: Path | None = None) -> Path:
    """标注数据集目录（默认 `EXPERIENCE_DIR / laya_annotations`）。"""
    if experience_dir is None:
        from pa_agent.config.paths import EXPERIENCE_DIR

        experience_dir = EXPERIENCE_DIR
    return experience_dir / "laya_annotations"


def append_sample(
    *,
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    answers: dict[str, Any],
    context: dict[str, Any],
    labels: dict[str, str] | None = None,
    experience_dir: Path | None = None,
) -> Path | None:
    """把一次 Laya 推理的三元组追加到当月 JSONL。返回文件路径；失败仅记日志。

    Args:
        labels: 人工标注的正确标签（qid → 正确选项）；None = 自动收集未标注样本。
    """
    try:
        out_dir = annotations_dir(experience_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        month = time.strftime("%Y-%m")
        path = out_dir / f"{month}.jsonl"

        sample = {
            "ts_ms": int(time.time() * 1000),
            "source": "manual_label" if labels else "auto",
            "label": labels,          # 人工/回测补齐后回填
            "state": state,
            "questions": questions,
            "answers": answers,       # 模型实际输出（注意：不是正确答案）
            "context": context,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return path
    except Exception as exc:  # noqa: BLE001 - 标注收集绝不能影响报告主流程
        logger.warning("Laya 标注样本写入失败: %s", exc)
        return None


def relabel_line(
    path: Path,
    line_no: int,
    *,
    label: dict[str, Any],
) -> bool:
    """给第 line_no 行（1-based）补标签。JSONL 是 append-only，
    本函数是唯一的「改写」入口——只在明确补标签时使用。

    Returns:
        True 表示成功改写。
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not (1 <= line_no <= len(lines)):
            return False
        sample = json.loads(lines[line_no - 1])
        sample["label"] = label
        sample["labeled_at_ms"] = int(time.time() * 1000)
        lines[line_no - 1] = json.dumps(sample, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Laya 标注补标签失败 (%s#%d): %s", path, line_no, exc)
        return False


def stats(experience_dir: Path | None = None) -> dict[str, int]:
    """数据集概况：总行数 / 已标注数 / 未标注数。"""
    out = {"total": 0, "labeled": 0, "unlabeled": 0}
    try:
        out_dir = annotations_dir(experience_dir)
        for p in sorted(out_dir.glob("*.jsonl")):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out["total"] += 1
                    try:
                        if json.loads(line).get("label") is not None:
                            out["labeled"] += 1
                        else:
                            out["unlabeled"] += 1
                    except json.JSONDecodeError:
                        out["unlabeled"] += 1
    except OSError:
        pass
    return out

# -*- coding: utf-8 -*-
"""Laya 训练/校准数据集（JSONL）。

为什么不是微调
--------------
Laya 0.3.22 官方**没有**公开 `train` 模块（`pip show -f laya` 只有推理 + 校准）。
它提供的官方学习路径是**置信度校准**，而不是改权重：

    laya.calibrate.records_from_labeled(agent, pairs)
      → agent.fit_temperatures(records)      # 拟合温度标量+ 分桶温度
      → agent.save_calibration("cal.json")   # 纯JSON，不写权重
      → laya.load(..., calibration="cal.json")

温度标量把模型过度自信的原始分布压平，使报出的置信度接近真实命中率。
本机CPU 就能跑，秒级，不需要 Kaggle GPU。只有 6 个问题 × N 个样本，
根本不足以做梯度微调——**校准是这个模型规模下的正确解法**。

数据格式
--------
每次生成 Laya 报告时顺手把三元组原样落盘：

    experience/laya_annotations/2026-10.jsonl
    {"ts_ms":..., "state":{...}, "questions":{...}, "answers":{...},
     "label":null, "source":"auto", "context":{symbol,timeframe,close,atr}}

两条产线：
1. **弱监督自动**（``weak_label.py``）：用未来 N 根 K 线的真实走势当老师，
   自动判定当时的方向判断对错并回填 ``label``——用户零操作即可积累样本。
2. **人工补标签**（Laya 报告「进入标注模式」）：逐题下拉框选正确答案。
   人工标签优先级高于弱监督，见 ``to_laya_pairs`` 的去重规则。
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
    source: str | None = None,
    experience_dir: Path | None = None,
) -> Path | None:
    """把一次 Laya 推理的三元组追加到当月 JSONL。返回文件路径；失败仅记日志。

    Args:
        labels: 人工标注的正确标签（qid → criteria 的 key）；None = 未标注样本。
        source: 数据来源。默认按有无 labels 推断 ``manual_label`` / ``auto``。
    """
    try:
        out_dir = annotations_dir(experience_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        month = time.strftime("%Y-%m")
        path = out_dir / f"{month}.jsonl"

        sample = {
            "ts_ms": int(time.time() * 1000),
            "source": source or ("manual_label" if labels else "auto"),
            "label": labels,          # 人工/弱监督补齐后回填
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


# ── 训练/校准数据读取 ───────────────────────────────────────────────────────

def _iter_samples(data_dir: Path) -> list[dict[str, Any]]:
    """读全部 JSONL 行，坏行跳过。"""
    out: list[dict[str, Any]] = []
    if not data_dir.is_dir():
        return out
    for p in sorted(data_dir.glob("*.jsonl")):
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("读取标注文件失败 %s: %s", p, exc)
    return out


def _target_vector(qspec: dict[str, Any], answer: str) -> list[float] | None:
    """把正确答案转成 one-hot 向量，键序 = ``criteria`` 的插入序。

    Laya 的 ``records_from_labeled`` 要求 ``targets[qid]`` 是与 logits 同宽的
    one-hot（或 soft）分布，宽度必须等于该题的选项数——也就是 ``criteria`` 的长度。
    """
    qtype = qspec.get("type")
    if qtype == "noul":
        # noul 是二分类：1=有效(P=1)，0=无效(P=0)
        val = 1.0 if str(answer) in ("有效", "1", "True", "true") else 0.0
        return [1.0 - val, val]
    criteria = qspec.get("criteria") or {}
    if not criteria:
        return None
    keys = list(criteria.keys())
    if answer not in keys:
        # 也接受中文标签（人工标注文本框里可能出现）
        rev = {str(v): k for k, v in criteria.items()}
        if str(answer) in rev:
            answer = rev[str(answer)]
        else:
            return None
    vec = [0.0] * len(keys)
    vec[keys.index(answer)] = 1.0
    return vec


def to_laya_pairs(
    samples: list[dict[str, Any]],
    *,
    prefer_manual: bool = True,
    min_answer_confidence: float = 0.0,
    subset_questions: bool = True,
) -> tuple[list[tuple[Any, dict, dict]], dict[str, int]]:
    """把 JSONL 样本转成 Laya 官方 ``(state, questions, targets)`` 三元组。

    官方 ``laya.calibrate.records_from_labeled`` 会对 ``questions`` 的**每个**
    qid 去取 ``targets[qid]``，缺一个就 ``KeyError``。而弱监督常常只能判定
    部分题（ER 灰区不标结构、震荡市不标方向），硬丢样本会损失大量数据。

    Args:
        prefer_manual: ``manual_label`` 来源排在 ``weak_supervision`` 之前。
        min_answer_confidence: 模型自身输出置信度低于此值的样本直接丢弃——
            零样本下大量样本是「模型自己都不知道」，拿它们当标准答案会污染校准。
        subset_questions: True（默认）= 把 ``questions`` **裁剪成只有已标注的题**，
            这样部分标注的样本也能用上（每条记录独立采集，部分题缺失不影响其他题）。
            False = 要求全覆盖，否则整条样本丢弃。

    Returns:
        ``(pairs, info)``；``info`` 含 total / paired / dropped_* 便于诊断。
    """
    info = {"total": len(samples), "paired": 0, "dropped_no_label": 0,
            "dropped_bad_target": 0, "dropped_low_conf": 0,
            "dropped_incomplete": 0, "subset_q": 0}
    prio = {"manual_label": 0, "weak_supervision": 1}
    ordered = sorted(
        samples,
        key=lambda s: prio.get(str(s.get("source")), 2) if prefer_manual else 0,
    )

    pairs: list[tuple[Any, dict, dict]] = []
    for s in ordered:
        label = s.get("label")
        if not label:
            info["dropped_no_label"] += 1
            continue
        if min_answer_confidence > 0:
            ans = s.get("answers") or {}
            confs = [
                float(a.get("confidence", 1.0)) for a in ans.values()
                if isinstance(a, dict) and a.get("confidence") is not None
            ]
            if confs and max(confs) < min_answer_confidence:
                info["dropped_low_conf"] += 1
                continue
        qs = s.get("questions") or {}
        if not qs:
            info["dropped_bad_target"] += 1
            continue
        targets: dict[str, list[float]] = {}
        for qid, correct in label.items():
            qspec = qs.get(qid)
            if not qspec:
                continue
            vec = _target_vector(qspec, str(correct))
            if vec is not None:
                targets[qid] = vec
        if not targets:
            info["dropped_bad_target"] += 1
            continue
        if subset_questions and len(targets) < len(qs):
            # 裁剪成只有已标注的题，避免官方函数对缺失 qid 抛 KeyError
            qs = {qid: q for qid, q in qs.items() if qid in targets}
            info["subset_q"] += 1
        state = s.get("state")
        if state is None:
            info["dropped_bad_target"] += 1
            continue
        pairs.append((state, qs, targets))
        info["paired"] += 1
    return pairs, info


def calibration_out_path(model_dir: str | Path) -> Path:
    """校准文件默认落点：权重目录下的 ``calibration.json``。

    放权重目录旁边而不是 settings 里配路径，是为了让"换了模型目录"自动失效——
    旧模型的温度对新模型没有意义。
    """
    return Path(model_dir).parent / "calibration.json"


def load_calibration_if_any(model_dir: str | Path) -> str | None:
    """返回可用的校准文件路径；没有就返回 None。

    存在性 + 可读性双重检查，坏文件绝不阻断推理。
    """
    try:
        p = calibration_out_path(model_dir)
        if p.is_file() and p.stat().st_size > 0:
            return str(p)
    except OSError:
        pass
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
    """数据集概况：总行数 / 已标注数 / 未标注数 / 分来源统计。"""
    out = {"total": 0, "labeled": 0, "unlabeled": 0,
           "manual": 0, "weak": 0, "auto": 0}
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
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        out["unlabeled"] += 1
                        continue
                    src = str(s.get("source") or "auto")
                    if src == "manual_label":
                        out["manual"] += 1
                    elif src == "weak_supervision":
                        out["weak"] += 1
                    else:
                        out["auto"] += 1
                    if s.get("label") is not None:
                        out["labeled"] += 1
                    else:
                        out["unlabeled"] += 1
    except OSError:
        pass
    return out

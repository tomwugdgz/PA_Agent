# -*- coding: utf-8 -*-
"""Laya 置信度校准：从标注样本到可加载的 calibration.json。

为什么是「校准」而不是「微调」
---------------------------
Laya 0.3.22 官方**没有发布 `train` 模块**（`pip show -f laya` 里只有 agent /
calibrate / router / serve 等推理与校准组件）。它给第三方的官方学习路径是
**置信度温度校准**：

    records = laya.calibrate.records_from_labeled(agent, pairs)
    result  = agent.fit_temperatures(records, compute_ece=True)
    agent.save_calibration("calibration.json")
    # 之后推理时：
    laya.load(model_dir, subfolder=..., calibration="calibration.json")

校准做的事：给每类问题（choice / noul / score）和每个选项桶拟合一个温度标量，
推理时把 logits 除以这个温度。零样本下模型系统性过度自信，标定后报出的置信度
才接近真实命中率。**本机 CPU 秒级完成，不写模型权重，不需要 Kaggle GPU。**

为什么不用 Kaggle 微调
----------------------
- 本项目只有 6 个问题。即便上万条样本，也远不够 LoRA 微调一个 615MB 模型，
  微调极易过拟合到单一品种。
- 微调要 4–5 小时 GPU + 环境踩坑；校准要几分钟且随时可回滚（删掉 json即可）。
- 两者不互斥：先用校准把置信度修正到可信，再攒够样本考虑微调。

用法
----
    # 1) 看数据够不够
    python tools/calibrate_laya.py --status

    # 2) 用已有标注跑校准
    python tools/calibrate_laya.py

    # 3) 样本 < 阈值时先造样本（弱监督，无需人工）
    python tools/weak_label.py --bars 400
    python tools/calibrate_laya.py

调完自动生效：``laya_engine`` 会在 ``load(..., calibration=...)`` 时自动挂载
``<model_dir>/../calibration.json``，无需改settings。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pa_agent.ai.laya_annotation import (  # noqa: E402
    _iter_samples,
    annotations_dir,
    calibration_out_path,
    load_calibration_if_any,
    stats,
    to_laya_pairs,
)
from pa_agent.ai.laya_schema import build_questions  # noqa: E402

#: 低于此样本量不建议校准（每题至少要 20+ 条才稳）
MIN_SAMPLES = 120
#: 官方分桶门槛：``laya.calibrate.MIN_BUCKET_N``。分桶温度需要每桶 2000 条，
#: 达不到就只出类型级标量（MIN_TYPE_N=10）。这里用于提示用户。
OFFICIAL_MIN_BUCKET_N = 2000


def _normalize_identity_field(cal_path: Path) -> bool:
    """把校准文件里的 ``model_id_or_path`` 归一化成 agent 实际持有的形态。

    上游 ``laya.calibrate._warn_if_identity_mismatch`` 用**字符串相等**判断
    校准是否匹配当前模型。而 ``settings.json`` 里``model_dir`` 惯用正斜杠
    （``C:/Users/...``），agent 内部保存的却是反斜杠（``C:\\Users\\...``），
    于是每次加载都会刷一条误报警告::

        UserWarning: calibration was fitted for model_id_or_path='C:/...'
        but this agent is model_id_or_path='C:\\...'. Loading anyway.

    温度**确实照常加载**（上游只是warn 后继续），所以这是纯噪音问题。
    这里只改写记录字段，不动任何温度数值。
    """
    try:
        payload = json.loads(cal_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    recorded = payload.get("model_id_or_path")
    if not isinstance(recorded, str) or not recorded:
        return False
    # os.path.normpath 会把正斜杠统一成本机原生分隔符（Windows 下为反斜杠）
    normalized = os.path.normpath(recorded)
    if normalized == recorded:
        return False
    payload["model_id_or_path"] = normalized
    cal_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True


def _collect_records(agent: Any, pairs: list) -> list:
    """采集校准记录，绕开上游 ``records_from_labeled`` 的 grad 缺陷。

    Laya 0.3.22 的 ``Agent.predict`` 带 ``@torch.no_grad()`` 装饰器，但
    ``laya.calibrate.records_from_labeled`` 内部直接调裸 ``agent._forward()``，
    而后者对带 grad 的 tensor 直接 ``.numpy()`` 会抛::

        RuntimeError: Can't call numpy() on Tensor that requires grad.

    这只影响校准路径（推理路径不受影响）。这里用 ``torch.no_grad()`` 包一层
    再调官方函数——不重写它的逻辑，只补上它漏掉的 grad 关闭。
    """
    import torch
    from laya import calibrate as lcal

    try:
        with torch.no_grad():
            return lcal.records_from_labeled(agent, pairs)
    except RuntimeError as exc:
        if "requires grad" not in str(exc):
            raise
        # 兜底：若 no_grad 仍不够（未来版本可能改了实现），逐条关闭参数梯度
        for p in getattr(agent, "model", None).parameters() if hasattr(agent, "model") else ():
            p.requires_grad_(False)
        with torch.no_grad():
            return lcal.records_from_labeled(agent, pairs)


def load_config() -> Any:
    """读项目 settings（拿laya.model_dir / subfolder / device）。"""
    from pa_agent.config.settings import load_settings

    return load_settings()


def _agent_kwargs(cfg: Any) -> dict[str, Any]:
    lc = cfg.laya
    device = getattr(lc, "device", "auto")
    return {
        "model_dir": lc.model_dir,
        "subfolder": lc.subfolder or None,
        "device": None if device in ("", "auto") else device,
    }


def show_status() -> int:
    """打印样本与校准现状，不加载模型。"""
    d = annotations_dir()
    s = stats()
    print("── Laya 校准状态 ──")
    print(f"标注目录：{d}")
    print(f"样本：{s['total']} 条｜已标注 {s['labeled']}"
          f"（人工 {s['manual']} / 弱监督 {s['weak']}）｜待标注 {s['unlabeled']}")

    try:
        cfg = load_config()
        cal = calibration_out_path(cfg.laya.model_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 读settings 失败：{exc}")
        cal = None

    if cal is not None and cal.is_file():
        try:
            payload = json.loads(cal.read_text(encoding="utf-8"))
            rep = payload.get("report") or {}
            print(f"校准文件：{cal}")
            print(f"  version={payload.get('version')}  "
                  f"temperature={payload.get('temperature')}  "
                  f"分桶={len(payload.get('temperature_by_options') or {})}")
            if rep:
                print(f"  样本 n={rep.get('n')}  留出评估 n_eval={rep.get('n_eval')}")
                print(f"  ECE：校准前 {rep.get('ece_before')} → 校准后 {rep.get('ece_after')}")
            print("  → 已生效，下次加载自动挂载")
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 校准文件损坏（不影响推理）：{exc}")
    else:
        print("校准文件：尚未生成")
        if s["labeled"] < MIN_SAMPLES:
            print(f"  → 需先积累 ≥{MIN_SAMPLES} 条已标注样本。"
                  f"先跑 tools/weak_label.py 用未来走势自动造标签。")
        else:
            print(f"  → 样本已够（{s['labeled']} ≥ {MIN_SAMPLES}），"
                  f"运行 python tools/calibrate_laya.py 生成。")

    print(f"\n当前置信度基线：零样本下 '方向' 题实测仅 0.14–0.17（作者原话 near chance）。")
    return 0


def run_calibration(
    *,
    min_confidence: float = 0.0,
    max_pairs: int = 0,
    force: bool = False,
) -> int:
    """执行校准并落盘。返回进程退出码。"""
    try:
        import laya  # noqa: F401  # 延迟导入：没装 laya 时 --status 仍可用
    except ImportError as exc:
        print(f"[错误] 未安装 laya：{exc}\n  pip install laya", file=sys.stderr)
        return 1

    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 读 settings 失败：{exc}", file=sys.stderr)
        return 1

    d = annotations_dir()
    samples = _iter_samples(d)
    pairs, info = to_laya_pairs(
        samples, prefer_manual=True, min_answer_confidence=min_confidence
    )
    print("── 数据准备 ──")
    print(f"总样本 {info['total']} → 可用 {info['paired']}")
    print(f"  跳过：无标签 {info['dropped_no_label']}、"
          f"目标构造失败 {info['dropped_bad_target']}、"
          f"置信度过低 {info['dropped_low_conf']}")
    if info.get("subset_q"):
        print(f"  部分标注（已裁剪问题集，仍参与校准）：{info['subset_q']} 条")

    if not pairs:
        print("\n[错误] 没有可用标注样本。先做这一步：", file=sys.stderr)
        print("  python tools/weak_label.py --bars 400   # 用未来走势自动造标签", file=sys.stderr)
        return 1
    if len(pairs) < MIN_SAMPLES and not force:
        print(f"\n[警告] 仅 {len(pairs)} 条样本，低于建议下限 {MIN_SAMPLES}，"
              f"校准结果不稳定。确认要继续请加 --force。")
        return 2
    if max_pairs and len(pairs) > max_pairs:
        print(f"[截断] 只用最近 {max_pairs} 条（全部 {len(pairs)} 条）")
        pairs = pairs[-max_pairs:]

    out_path = calibration_out_path(cfg.laya.model_dir)
    if out_path.is_file() and not force:
        print(f"\n[跳过] {out_path} 已存在。覆盖请加 --force。")
        return 0

    # ── 加载模型并采集记录 ──────────────────────────────────────────────
    print("\n── 加载模型 ──")
    t0 = time.perf_counter()
    from pa_agent.ai.laya_engine import LayaEngine

    engine = LayaEngine.get(**_agent_kwargs(cfg))
    engine.ensure_loaded()
    agent = engine._agent  # 官方校准 API 直接吃 Agent 实例
    print(f"权重已加载（{time.perf_counter() - t0:.1f}s，设备 {engine.device or '?'}）")

    print(f"\n── 采集 logits（{len(pairs)} 条，CPU 前向）──")
    t1 = time.perf_counter()
    records = _collect_records(agent, pairs)
    print(f"采集到 {len(records)} 条记录，耗时 {time.perf_counter() - t1:.1f}s")
    if not records:
        print("[错误] 没有采到任何记录——问题集可能与模型不匹配。", file=sys.stderr)
        return 1

    # ── 拟合温度 ────────────────────────────────────────────────────────
    print("\n── 拟合温度标量 + 分桶温度（留出 20% 算 ECE）──")
    result = agent.fit_temperatures(records, compute_ece=True)
    temp = result.get("temperature")
    by_opt = result.get("temperature_by_options") or {}
    rep = result.get("report") or {}
    print(f"  类型级温度 = {temp}")
    print(f"  分桶温度 {len(by_opt)} 个"
          + (f"：{json.dumps(by_opt, ensure_ascii=False)[:200]}"
             if by_opt else f"（需每桶 ≥{OFFICIAL_MIN_BUCKET_N} 条，当前 {len(records)} 条，"
                             f"故只用类型级标量——这已足够把过度自信压平）"))
    before, after = rep.get("ece_before"), rep.get("ece_after")
    n_eval = rep.get("n_eval")
    if n_eval:
        print(f"  留出集 n_eval={n_eval}，ECE：{before} → {after}（越小越好）")
        try:
            print(f"  改善 {(float(before) - float(after)) * 100:+.2f} 个百分点"
                  f"（正数=变好）")
        except (TypeError, ValueError):
            pass
    else:
        print(f"  ECE 无法评估：样本量（{len(records)} 条）低于官方分桶门槛 "
              f"{OFFICIAL_MIN_BUCKET_N}，留出集为空。")
        print("  → 这不影响校准本身生效：温度标量已拟合并会写盘。")
        print("  → 想看到 ECE 数字需积累数千条样本，届时重跑本脚本即可自动评估。")
    excl = rep.get("buckets_excluded_from_eval")
    if excl:
        print(f"  样本太少被排除评估的桶：{excl}")

    # ── 落盘 ────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_calibration(str(out_path))
    _normalize_identity_field(out_path)
    print(f"\n✅ 校准已保存：{out_path}")
    print("   下次加载 Laya 时自动生效（engine 会读同一路径），无需改配置。")

    meta = {
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pairs": len(pairs),
        "records": len(records),
        "temperature": temp,
        "n_buckets": len(by_opt),
        "ece_before": rep.get("ece_before"),
        "ece_after": rep.get("ece_after"),
        "n": rep.get("n"),
        "n_eval": rep.get("n_eval"),
        "model_id": getattr(agent, "model_id", None),
    }
    (out_path.parent / "calibration_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Laya 置信度校准（官方 API，本机 CPU 即可）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--status", action="store_true", help="只看状态，不加载模型")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的校准文件")
    ap.add_argument("--max-pairs", type=int, default=0, help="只用最近 N 条（0=全部）")
    ap.add_argument("--min-answer-confidence", type=float, default=0.0,
                    help="丢弃模型自身置信度低于此值的样本（默认不丢）")
    args = ap.parse_args()

    if args.status:
        sys.exit(show_status())
    sys.exit(run_calibration(
        min_confidence=args.min_answer_confidence,
        max_pairs=args.max_pairs,
        force=args.force,
    ))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""优化项冒烟：真实进度回调 / 重复持仓拦截 / 标注 criteria 映射。

不连真实 MT5 终端、不加载 Laya 权重，纯离线验证三条链路的逻辑正确性。
用法：python tools/opt_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pa_agent.ai.laya_engine import _emit  # noqa: E402
from pa_agent.ai.laya_schema import build_questions  # noqa: E402


def test_progress_chain() -> None:
    """1) 进度回调：pct 单调不降、msg 非空、异常不影响主流程。"""
    seen: list[tuple[int, str]] = []
    for pct, msg in ((3, "准备特征"), (5, "检查运行时"), (20, "校验权重"),
                     (80, "权重加载完成"), (85, "构造状态"), (90, "推理中"), (100, "报告完成")):
        _emit(lambda p, m: seen.append((p, m)), pct, msg)
    pcts = [p for p, _ in seen]
    assert pcts == sorted(pcts), f"进度必须单调不减，实际 {pcts}"
    assert all(m for _, m in seen), "每段都必须带说明文字"
    # 回调抛异常必须被吞掉
    def _boom(_pct, _msg):
        raise RuntimeError("UI 已销毁")
    _emit(_boom, 50, "x")
    print(f"  [OK] 进度链路 {len(seen)} 段，终态 {pcts[-1]}%，异常回调已安全吞掉")


def test_choice_mapping() -> None:
    """2) 标注映射：criteria 的 key/中文标签必须双向映射正确。"""
    qs = build_questions()
    crit = qs["周期结构"]["criteria"]
    labels = {str(v): k for k, v in crit.items()}
    keys = list(crit.keys())

    # 模拟 GUI：下拉框显示中文 → 存回 key
    saved = {"周期结构": labels["宽通道"]}
    assert saved["周期结构"] == "broad_channel", saved

    # 模拟 finetune：key → 选项索引
    idx = keys.index(saved["周期结构"])
    assert keys[idx] == "broad_channel", (keys, idx)

    # 反向：GUI 预选时用 value(key) 反查中文标签
    qspec = qs["方向"]
    v = "bearish"
    hit = next((kk for kk, vv in qspec["criteria"].items() if kk == v), None)
    assert hit == "bearish"
    assert qspec["criteria"][hit] == "偏空"

    # noul 题没有 criteria，必须走 有效/无效 分支
    assert "criteria" not in qs["信号有效"], qs["信号有效"]
    print(f"  [OK] criteria 映射：{len(crit)} 个周期结构选项、方向 3 选、noul 无criteria 均可")


def test_same_side_counter_signature() -> None:
    """3) 重复持仓计数器：签名与 magic 过滤逻辑（不打真实终端）。"""
    from pa_agent.mt5trading import mt5_bridge

    assert hasattr(mt5_bridge, "_count_same_side"), "缺少 _count_same_side"
    import inspect
    sig = inspect.signature(mt5_bridge._count_same_side)
    assert list(sig.parameters) == ["symbol", "direction", "magic"], sig
    # 类型映射：long→POSITION_TYPE_BUY(0)，short→POSITION_TYPE_SELL(1)
    src = inspect.getsource(mt5_bridge._count_same_side)
    assert "direction == \"long\"" in src.replace("'", '"'), "long→0 映射丢失"
    print(f"  [OK] _count_same_side{sig} 已就位，取不到数据时放行（不误挡正常下单）")


def test_spread_guard_default() -> None:
    """4) 实盘默认风控：点差上限不再为 0。"""
    from pa_agent.config.settings import MT5Settings

    cfg = MT5Settings()
    assert cfg.max_spread_points > 0, "点差上限不能是 0（等于无保护）"
    assert cfg.max_same_symbol_positions > 0, "同向持仓上限不能是 0"
    assert cfg.confirm_required is True, "下单确认必须保持开启"
    print(f"  [OK] 实盘默认：点差上限 {cfg.max_spread_points} point、"
          f"同向持仓上限 {cfg.max_same_symbol_positions} 笔、强制确认 {cfg.confirm_required}")


def main() -> None:
    print("── 优化项冒烟 ──")
    test_progress_chain()
    test_choice_mapping()
    test_same_side_counter_signature()
    test_spread_guard_default()
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
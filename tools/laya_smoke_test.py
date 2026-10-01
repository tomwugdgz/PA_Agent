# -*- coding: utf-8 -*-
"""Laya 可行性冒烟测试（本地离线权重版）。

前置：先跑 `python tools/download_laya.py` 把 multilingual 权重拉到
      ~/laya-models/laya/multilingual/（Windows 即 %USERPROFILE% 下同名目录）

验证三件事：
  1. 本地目录能否被 laya.load 接受（不走联网）
  2. 中文 state + choice/noul 问题能否返回结构化结果
  3. CPU 上的加载耗时与推理延迟是否可接受

用法：  python tools/laya_smoke_test.py
"""
from __future__ import annotations

import os
import time

# 1) 不要探测 TensorFlow（会拖慢/在某些环境死锁）
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# 2) 强制离线：任何意外的联网尝试都会立刻报错，而不是静默等待
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import laya  # noqa: E402

MODEL_DIR = os.path.expanduser("~/laya-models/laya")
SUBFOLDER = "multilingual"

# PA_Agent 的 8 类周期位置（cycle_enums.CYCLE_ORDER），正好 <20 选项
CYCLES = {
    "spike": "急速单边尖峰，K 线连续同向、重叠极少",
    "micro_channel": "微型通道，极窄且紧密的推进",
    "tight_channel": "窄通道，回撤浅、重叠多",
    "normal_channel": "正常通道，推进与回撤均衡",
    "broad_channel": "宽通道，回撤深、波动大",
    "trending_tr": "趋势型交易区间，区间内有方向性倾斜",
    "trading_range": "交易区间，上下边界明确、方向中性",
    "extreme_tr": "极端交易区间，边界极宽、噪音大",
}

QUESTIONS = {
    "结构": {
        "type": "choice",
        "instructions": "这段行情属于哪种市场周期结构？",
        "criteria": CYCLES,
    },
    "方向": {
        "type": "choice",
        "instructions": "当前方向倾向？",
        "criteria": {
            "bullish": "偏多，低点抬高、买方主导",
            "bearish": "偏空，高点降低、卖方主导",
            "neutral": "中性，方向不明或区间震荡",
        },
    },
    "信号有效": {
        "type": "noul",
        "instructions": "当前是否存在可靠的交易信号（结构清晰、边界明确）？",
    },
}

# 模拟一段紧凑的市场状态（中文），即后续真正喂给 Laya 的特征文本
STATE = {
    "body": (
        "品种 USDJPY 1h。最近 100 根：区间上沿 157.683，下沿 156.510，"
        "区间宽度 2.31 倍 ATR。当前收盘 157.21，处于区间 0.62 分位（中部）。"
        "距上边界 1.9 倍 ATR，距下边界 3.2 倍 ATR。"
        "摆动结构为高低点持平，突破尝试在上沿失败（breakout_failure）。"
        "近 10 根重叠率 0.71，十字星占比 0.30，铁丝网评分 0.42。"
        "H1H2/L1L2 计数为 0。趋势上下文：长程偏空、近期中性。"
    )
}


def main() -> int:
    print(f"laya 版本 : {laya.__version__}")
    print(f"权重目录  : {MODEL_DIR} (subfolder={SUBFOLDER})")
    missing = [
        p
        for p in (
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "encoder/config.json",
        )
        if not os.path.exists(os.path.join(MODEL_DIR, SUBFOLDER, *p.split("/")))
    ]
    if missing:
        print("缺少文件（先跑 tools/download_laya.py）:", missing)
        return 2

    t0 = time.perf_counter()
    agent = laya.load(MODEL_DIR, subfolder=SUBFOLDER, device="cpu")
    load_s = time.perf_counter() - t0
    print(f"[加载] {load_s:.2f} s  device={agent.device}")
    print(f"[配置] max_len={agent.cfg.get('max_len')} "
          f"head_max_len={agent.cfg.get('head_max_len')} "
          f"encoder={agent.cfg.get('encoder')}")

    t1 = time.perf_counter()
    out = agent.predict(STATE, QUESTIONS, lang="zh")
    infer_ms = (time.perf_counter() - t1) * 1000
    print(f"[首次推理] {infer_ms:.0f} ms")

    usage = out.get("usage", {})
    print(f"[token] {usage}")

    for k, v in out.get("answers", {}).items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
            continue
        if "choice" in v:
            print(f"  {k}: choice={v['choice']!r} conf={v.get('confidence')} "
                  f"probs={v.get('probabilities')}")
        elif "noul" in v:
            print(f"  {k}: P(true)={v['noul']} conf={v.get('confidence')}")
        elif "score" in v:
            print(f"  {k}: score={v['score']} conf={v.get('confidence')}")
        else:
            print(f"  {k}: {v}")

    lat = []
    for _ in range(5):
        s = time.perf_counter()
        agent.predict(STATE, QUESTIONS, lang="zh")
        lat.append((time.perf_counter() - s) * 1000)
    lat.sort()
    print(f"[延迟] p50={lat[len(lat) // 2]:.0f}ms min={lat[0]:.0f}ms max={lat[-1]:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

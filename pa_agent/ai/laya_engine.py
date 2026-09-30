# -*- coding: utf-8 -*-
"""Laya 推理引擎封装：懒加载单例、线程安全、离线守卫、优雅降级。

关键点
------
1. **必须**在首次 `import laya`（连带 transformers）之前设置离线环境变量，
   否则 transformers 会尝试联网并卡死在代理 502 上。
2. 加载耗时 ~17 s（CPU），**绝不能**在 UI 线程里同步加载——GUI 侧必须丢进
   QThread/worker。本模块只提供线程安全接口，不做 Qt 依赖。
3. `laya` 包未安装 / 权重缺失时返回明确异常，由 GUI 层转成可读提示，
   而不是让整个程序崩掉。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LayaUnavailable(RuntimeError):
    """Laya 运行时或权重不可用。message 面向终端用户，需可直接展示。"""


def _prepare_offline_env() -> None:
    """在 import laya/transformers 之前调用。

    `USE_TF=0` / `TRANSFORMERS_NO_TF=1`：跳过 TensorFlow 探测（慢且可能死锁）。
    `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`：权重已本地化，
    任何联网尝试都应立即报错而非静默等待代理超时。
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def check_runtime() -> str | None:
    """预检：返回 None 表示可用，否则返回**面向用户**的失败原因。"""
    _prepare_offline_env()
    try:
        import laya  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"Laya 库未安装或导入失败：{exc}"
    return None


def check_weights(model_dir: str, subfolder: str) -> str | None:
    """校验本地权重完整性，返回 None 表示可用。"""
    root = Path(model_dir)
    if not root.is_dir():
        return f"权重目录不存在：{root}"
    sub = root / subfolder if subfolder else root
    required = [
        "rl_agent_config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "encoder/config.json",
    ]
    missing = [p for p in required if not (sub / p).exists()]
    if missing:
        return (
            f"权重目录 {sub} 缺少文件：{', '.join(missing)}。"
            f"请先运行 tools/download_laya.py"
        )
    return None


class LayaEngine:
    """单实例封装。`get()` 保证全局唯一（权重 ~1.6 GB 内存，禁止重复加载）。"""

    _lock = threading.Lock()
    _instance: "LayaEngine | None" = None

    def __init__(self, *, model_dir: str, subfolder: str, device: str) -> None:
        self.model_dir = model_dir
        self.subfolder = subfolder
        self.device_req = device
        self._agent: Any = None
        self._infer_lock = threading.Lock()
        self.load_ms: float = 0.0
        self.device: str = ""

    # ── 单例管理 ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, *, model_dir: str, subfolder: str, device: str) -> "LayaEngine":
        """按配置取单例；配置变化（如换权重目录）时重建。

        加载本身放到 `ensure_loaded()`，本方法只建壳，便于先在 UI 线程
        校验配置、再丢到 worker 线程做重活。
        """
        with cls._lock:
            inst = cls._instance
            if (
                inst is not None
                and inst.model_dir == model_dir
                and inst.subfolder == subfolder
                and inst.device_req == device
            ):
                return inst
            cls._instance = cls(model_dir=model_dir, subfolder=subfolder, device=device)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """丢弃单例（换权重 / 换设备后调用）。旧对象由 GC 回收。"""
        with cls._lock:
            cls._instance = None

    # ── 加载 ─────────────────────────────────────────────────────────────────

    def ensure_loaded(self) -> Any:
        """加载权重并返回 Agent。线程安全、幂等。

        Raises:
            LayaUnavailable: 库缺失 / 权重缺失 / 加载失败（message 可直接展示）。
        """
        if self._agent is not None:
            return self._agent
        with self._infer_lock:
            if self._agent is not None:
                return self._agent

            reason = check_runtime()
            if reason:
                raise LayaUnavailable(reason)
            reason = check_weights(self.model_dir, self.subfolder)
            if reason:
                raise LayaUnavailable(reason)

            _prepare_offline_env()
            t0 = time.perf_counter()
            try:
                import laya

                device = self.device_req
                if device in ("", "auto"):
                    device = None  # laya 自己解析：cuda > mps > xpu > cpu
                agent = laya.load(
                    self.model_dir,
                    subfolder=self.subfolder or None,
                    device=device,
                )
            except LayaUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                raise LayaUnavailable(f"Laya 权重加载失败：{type(exc).__name__}: {exc}") from exc
            self.load_ms = (time.perf_counter() - t0) * 1000
            self.device = str(getattr(agent, "device", ""))
            logger.info(
                "Laya loaded in %.1fs (device=%s, dir=%s/%s)",
                self.load_ms / 1000,
                self.device,
                self.model_dir,
                self.subfolder,
            )
            self._agent = agent
            return agent

    @property
    def loaded(self) -> bool:
        return self._agent is not None

    # ── 推理 ─────────────────────────────────────────────────────────────────

    def predict(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        *,
        lang: str = "zh",
    ) -> dict[str, Any]:
        """同步推理。串行化：并发调用同一份模型既无收益也有风险。

        Raises:
            LayaUnavailable: 未加载或推理失败。
        """
        agent = self.ensure_loaded()
        with self._infer_lock:
            try:
                return agent.predict(state, questions, lang=lang)
            except Exception as exc:  # noqa: BLE001
                raise LayaUnavailable(
                    f"Laya 推理失败：{type(exc).__name__}: {exc}"
                ) from exc

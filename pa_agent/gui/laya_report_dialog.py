# -*- coding: utf-8 -*-
"""「Laya 报告」对话框：后台推理 + 报告展示 + MD/HTML/JSON 导出。

交互设计
--------
- 首次点击要等权重加载（CPU ~17 s / GPU ~3-6 s），所以按钮按下后立即
  弹出对话框显示进度，推理完成后原地刷新内容——绝不冻结主窗口。
- 加载结果全局缓存（LayaEngine 单例），第二次点击只花一次推理时间。
- 展示用 QTextBrowser 渲染 HTML（无需依赖外部浏览器），另提供
  「导出 MD / 导出 HTML」落盘到 `logs/laya_reports/`。
"""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from pa_agent.report.laya_pipeline import generate_report
from pa_agent.report.laya_report import render_html, render_markdown

logger = logging.getLogger(__name__)


class LayaReportWorker(QThread):
    """后台跑 generate_report；加载+推理全程不碰 UI。"""

    ready = pyqtSignal(object)     # LayaReport
    failed = pyqtSignal(str)

    def __init__(self, frame: Any, settings: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._settings = settings

    def run(self) -> None:
        try:
            report = generate_report(self._frame, self._settings)
            self.ready.emit(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LayaReportWorker failed: %s", exc)
            self.failed.emit(str(exc))


class LayaReportDialog(QDialog):
    """Laya 报告窗口（模态）。同一时刻只允许一个 worker 在跑。"""

    def __init__(self, frame: Any, settings: Any, parent: Any = None,
                 frame_provider: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Laya 市场分析报告")
        self.resize(920, 760)
        self._frame = frame
        self._settings = settings
        #: 返回最新 KlineFrame 的回调（主窗口提供）；None = 用静态帧
        self._frame_provider = frame_provider
        self._worker: LayaReportWorker | None = None
        self._report: Any = None
        self._last_html = ""

        root = QVBoxLayout(self)

        self._status = QLabel("正在加载 Laya 权重并推理（首次约需几十秒）…")
        self._status.setObjectName("mutedLabel")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._view = QTextBrowser()
        self._view.setOpenExternalLinks(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._view.setFont(font)
        root.addWidget(self._view, 1)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("刷新（最新数据）")
        self._refresh_btn.setToolTip(
            "重新拉取主窗口当前图表的最新收盘 K 线并重新推理——\n"
            "每次刷新都用此刻的数据，不会复用旧报告。")
        self._refresh_btn.clicked.connect(self._refresh_with_latest)
        self._export_md_btn = QPushButton("导出 Markdown")
        self._export_html_btn = QPushButton("导出 HTML")
        self._open_dir_btn = QPushButton("打开报告目录")
        self._refresh_btn.setEnabled(self._frame_provider is not None)
        btn_row.addWidget(self._refresh_btn)
        for b in (self._export_md_btn, self._export_html_btn, self._open_dir_btn):
            b.setEnabled(False)
            btn_row.addWidget(b)
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._export_md_btn.clicked.connect(self._export_md)
        self._export_html_btn.clicked.connect(self._export_html)
        self._open_dir_btn.clicked.connect(self._open_dir)

        self._start()

    # ── 流程 ─────────────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._worker = LayaReportWorker(self._frame, self._settings, parent=self)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._refresh_btn.setEnabled(False)
        self._worker.start()

    def _refresh_with_latest(self) -> None:
        """用主窗口当前最新数据重建帧并重新推理。"""
        if self._worker is not None and self._worker.isRunning():
            return
        if self._frame_provider is not None:
            try:
                fresh = self._frame_provider()
            except Exception as exc:  # noqa: BLE001
                fresh = None
                logger.debug("frame_provider 失败: %s", exc)
            if fresh is not None and getattr(fresh, "bars", None):
                self._frame = fresh
        self._status.setText("正在用最新数据重新推理…")
        self._start()

    def _on_ready(self, report: Any) -> None:
        self._report = report
        self._last_html = render_html(report)
        self._view.setHtml(self._last_html)
        newest_ts = getattr(getattr(report, "bars_meta", None), "newest_ts", None)
        self._status.setText(
            f"{report.symbol} {report.timeframe} · 设备 {report.prediction.device or '?'}"
            f" · 推理 {report.prediction.latency_ms:.0f}ms"
            f"（权重加载 {report.prediction.load_ms / 1000:.1f}s）"
            f" · 报告时间 {report.generated_at}"
        )
        for b in (self._refresh_btn, self._export_md_btn, self._export_html_btn,
                  self._open_dir_btn):
            b.setEnabled(True)
        self._worker = None

    def _on_failed(self, msg: str) -> None:
        self._status.setText("生成失败")
        self._view.setHtml(
            "<div style='color:#B91C1C;font-size:15px;padding:12px;'>"
            f"无法生成 Laya 报告：<br><br><b>{msg}</b></div>"
        )
        self._worker = None

    # ── 导出 ─────────────────────────────────────────────────────────────────

    def _report_dir(self):
        from pa_agent.config.paths import LOGS_DIR

        d = LOGS_DIR / "laya_reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _base_name(self) -> str:
        r = self._report
        slug = r.generated_at.replace("-", "").replace(":", "").replace(" ", "_")
        return self._report_dir() / f"laya_{r.symbol}_{r.timeframe}_{slug}"

    def _export_md(self) -> None:
        if not self._report:
            return
        base = self._base_name()
        path = base.with_suffix(".md")
        try:
            path.write_text(render_markdown(self._report), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _export_html(self) -> None:
        if not self._report:
            return
        base = self._base_name()
        path = base.with_suffix(".html")
        try:
            path.write_text(self._last_html or render_html(self._report), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_dir(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report_dir())))

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt 命名
        w = self._worker
        if w is not None and w.isRunning():
            w.wait(1500)  # 推理是纯计算，1.5s 内不响应就随它去（守护线程由 Qt 回收）
        super().closeEvent(event)

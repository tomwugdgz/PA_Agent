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

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pa_agent.ai.laya_schema import build_questions
from pa_agent.report.laya_pipeline import generate_report
from pa_agent.report.laya_report import render_html, render_markdown

logger = logging.getLogger(__name__)


class LayaReportWorker(QThread):
    """后台跑 generate_report；加载+推理全程不碰 UI。"""

    ready = pyqtSignal(object)     # LayaReport
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # 真实加载/推理阶段进度

    def __init__(self, frame: Any, settings: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._settings = settings

    def run(self) -> None:
        try:
            report = generate_report(self._frame, self._settings, progress=self._on_progress)
            self.ready.emit(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LayaReportWorker failed: %s", exc)
            self.failed.emit(str(exc))

    def _on_progress(self, pct: int, msg: str) -> None:
        # 可能从 worker 内部线程发出；Qt 信号跨线程安全
        self.progress.emit(int(pct), str(msg))


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

        # 进度条：显示**真实**阶段进度（由 LayaEngine/推理回调驱动）
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("加载中... %p%")
        root.addWidget(self._progress)

        # 计时器：只更新「已等待 Ns」，不再伪造进度百分比
        self._elapsed_s = 0.0
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._tick_elapsed)
        self._progress_timer.start(1000)

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
        self._label_mode_btn = QPushButton("进入标注模式")
        self._label_mode_btn.setCheckable(True)
        self._label_mode_btn.setToolTip(
            "开启后可为 Laya 的每个问题答案打标签（多/空/观望等），\n"
            "用于后续微调训练；标签自动存入 experience/laya_annotations/")
        self._label_mode_btn.clicked.connect(self._toggle_label_mode)
        self._export_md_btn = QPushButton("导出 Markdown")
        self._export_html_btn = QPushButton("导出 HTML")
        self._open_dir_btn = QPushButton("打开报告目录")
        self._refresh_btn.setEnabled(self._frame_provider is not None)
        btn_row.addWidget(self._refresh_btn)
        btn_row.addWidget(self._label_mode_btn)
        for b in (self._export_md_btn, self._export_html_btn, self._open_dir_btn):
            b.setEnabled(False)
            btn_row.addWidget(b)
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # 标注面板（默认隐藏，开启标注模式后显示在报告下方）
        self._label_panel = self._build_label_panel()
        self._label_panel.setVisible(False)
        root.addWidget(self._label_panel)

        self._export_md_btn.clicked.connect(self._export_md)
        self._export_html_btn.clicked.connect(self._export_html)
        self._open_dir_btn.clicked.connect(self._open_dir)

        self._start()

    # ── 流程 ─────────────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._worker = LayaReportWorker(self._frame, self._settings, parent=self)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self._on_progress)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._elapsed_s = 0.0
        self._progress_timer.start(1000)
        self._refresh_btn.setEnabled(False)
        self._worker.start()

    def _tick_elapsed(self) -> None:
        """每秒更新等待秒数——真实反馈，不参与进度计算。"""
        self._elapsed_s += 1.0
        cur = self._progress.value()
        self._status.setText(f"Laya 分析中…（已等待 {self._elapsed_s:.0f} 秒，当前阶段 {cur}%）")

    def _on_progress(self, pct: int, msg: str) -> None:
        """真实阶段进度：只在阶段推进时更新，避免高频重绘。"""
        pct = max(0, min(100, int(pct)))
        if pct > self._progress.value():
            self._progress.setValue(pct)
            self._progress.setFormat(f"{msg} %p%")
        self._status.setText(f"Laya 分析中… {msg}")

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
        self._progress_timer.stop()
        self._progress.setVisible(False)
        self._report = report
        self._last_html = render_html(report)
        self._view.setHtml(self._last_html)
        device_name = report.prediction.device or "?"
        load_s = report.prediction.load_ms / 1000
        infer_ms = report.prediction.latency_ms
        self._status.setText(
            f"{report.symbol} {report.timeframe} · 设备 {device_name}"
            f" · 推理 {infer_ms:.0f}ms（权重加载 {load_s:.1f}s）"
        )
        for b in (self._refresh_btn, self._export_md_btn, self._export_html_btn, self._open_dir_btn):
            b.setEnabled(True)
        self._worker = None

    def _update_progress_simulated(self) -> None:
        """已废弃：改用真实阶段进度（见 _on_progress）。保留空实现以防旧调用。"""
        return

    def _on_failed(self, msg: str) -> None:
        self._progress_timer.stop()
        self._progress.setVisible(False)
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

    # ── 标注模式 ─────────────────────────────────────────────────────────────

    def _build_label_panel(self) -> QWidget:
        """构建标注面板：为 Laya 的每个问题提供下拉选择标签。"""
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        hint = QLabel("标注说明：为每个问题选择你认为正确的答案，用于后续微调训练。")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self._label_widgets: dict[str, QComboBox] = {}
        self._label_submit_btn = QPushButton("保存标注")
        self._label_submit_btn.clicked.connect(self._save_labels)
        v.addWidget(self._label_submit_btn)
        return panel

    def _toggle_label_mode(self, checked: bool) -> None:
        """切换标注模式显示/隐藏。"""
        self._label_panel.setVisible(checked)
        if checked and self._report is not None:
            self._populate_label_widgets()
        self._label_mode_btn.setText("退出标注模式" if checked else "进入标注模式")

    def _populate_label_widgets(self) -> None:
        """根据当前报告的 questions 填充标注下拉框。"""
        # 清空旧控件
        for i in reversed(range(self._label_panel.layout().count() - 1)):
            item = self._label_panel.layout().itemAt(i + 1)  # 跳过 hint
            if item and item.widget():
                item.widget().deleteLater()
        self._label_widgets.clear()

        from pa_agent.ai.laya_schema import build_questions

        questions = build_questions()
        answers = self._report.prediction.answers if self._report else {}

        for qid, qspec in questions.items():
            row = QHBoxLayout()
            # 注意：schema 里的键是 instructions / criteria，不是 question / options
            text = str(qspec.get("instructions") or qspec.get("question") or qid)
            label = QLabel(text[:30])  # 截断长问题
            label.setToolTip(text)
            combo = QComboBox()
            if qspec["type"] == "noul":
                options = ["有效", "无效"]
            elif qspec["type"] == "score":
                options = [str(i) for i in range(11)]
            else:
                # choice：criteria 是 {key: 中文标签}，用中文标签做选项，
                # 保存时再映射回 key（见 _label_key_of）
                options = [str(v) for v in (qspec.get("criteria") or {}).values()] or ["(无选项)"]
            combo.addItems(options)
            # 默认选中 Laya 给出的答案
            ans = answers.get(qid)
            if ans is not None:
                val = str(ans.value) if hasattr(ans, "value") else str(ans)
                idx = combo.findText(val)
                if idx < 0 and qspec["type"] != "noul":
                    # value 可能是 criteria 的 key，反查其中文标签
                    crit = qspec.get("criteria") or {}
                    for k, v in crit.items():
                        if str(k) == val:
                            idx = combo.findText(str(v))
                            break
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            row.addWidget(label)
            row.addWidget(combo, 1)
            self._label_widgets[qid] = combo
            self._label_panel.layout().addLayout(row)

    def _label_key_of(self, qid: str, text: str) -> str:
        """把下拉框里的中文标签映射回 criteria 的 key（noul 除外）。"""
        qspec = build_questions().get(qid, {})
        if qspec.get("type") == "choice":
            for k, v in (qspec.get("criteria") or {}).items():
                if str(v) == text:
                    return k
        return text

    def _save_labels(self) -> None:
        """把用户标注写入 JSONL 文件。"""
        if not self._report:
            return
        from pa_agent.ai.laya_annotation import append_sample

        questions = build_questions()
        labels: dict[str, str] = {}
        for qid, combo in self._label_widgets.items():
            labels[qid] = self._label_key_of(qid, combo.currentText())

        try:
            append_sample(
                state=self._report.state_text,
                questions=questions,
                answers={
                    qid: {"kind": a.kind, "value": a.value, "confidence": a.confidence}
                    for qid, a in self._report.prediction.answers.items()
                },
                context={
                    "symbol": self._report.symbol,
                    "timeframe": self._report.timeframe,
                    "close": self._report.close,
                    "atr": self._report.atr,
                },
                labels=labels,  # 新增字段
            )
            QMessageBox.information(self, "标注已保存", "标签已存入 experience/laya_annotations/")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))

"""外部 TradingAgents 项目的启动与存活检测。

背景
----
``D:\\download\\TradingAgents-0.6.0\\TradingAgents-0.6.0\\启动TradingAgents.bat``
的内容是::

    @echo off
    chcp 65001 >nul
    cd /d "D:\\download\\TradingAgents-0.6.0\\TradingAgents-0.6.0"
    .venv\\Scripts\\tradingagents.exe
    pause

**实测结论（重要）**：该exe 是 Typer CLI，**无参数运行只打印帮助然后退出**，
不会常驻界面；而该项目 0.6.0 **没有 Web/GUI 入口**（无 gradio / streamlit /
uvicorn），只有：
* CLI，唯一的子命令是 ``backtest``
* ``main.py`` —— 会真实执行 ``ta.propagate("NVDA", "2026-09-01")``

所以那个 bat 实际并没有"启动一个界面"。经Tom 确认，按钮改为跑
``main.py``（真正会干活的分析流程）。

为什么不调 bat
--------------
它只做三件事：切 UTF-8 编码、切工作目录、跑 venv 里的程序。
GUI 里不需要黑窗口，也不需要 ``pause``，因此直接跑解释器——等价但可控。

为什么用 QProcess 而非 os.startfile
-----------------------------------
``startfile`` 会弹独立控制台、拿不到退出码、无法判断是否已在运行。
``QProcess`` 两者都能做，且不阻塞 GUI 主线程。

进程识别的局限（必须诚实说明）
--------------------------------
``tasklist`` 只能按**进程名**匹配，无法区分是不是同一个项目目录下的。
若同时开了两个不同副本，本模块可能误判为"已在运行"。
要精确区分需记录 PID 到状态文件（引入持久化），当前单机单实例场景
不必要，故不做。
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: TradingAgents 项目根目录（bat 里的 cd 目标）
TRADINGAGENTS_DIR = Path(
    r"D:\download\TradingAgents-0.6.0\TradingAgents-0.6.0"
)

#: 入口脚本——真正会执行分析的那个（CLI 无参只会打印帮助后退出）
ENTRY_SCRIPT = "main.py"

#: 项目自带解释器；用它才能保证依赖与该项目的 .venv 一致
VENV_PYTHON = ".venv/Scripts/python.exe"

#: 进程名（tasklist 里显示的名字，不含路径）——用于查存活
PROC_NAME = "python.exe"
#: 只有命令行里含该标记的进程才算TradingAgents，避免和别的 python 混淆
PROC_MARKER = "main.py"


@dataclass(frozen=True)
class LaunchTarget:
    """启动目标的解析结果。"""

    python_exe: Path
    script: Path
    workdir: Path

    @property
    def exists(self) -> bool:
        return (
            self.python_exe.is_file()
            and self.script.is_file()
            and self.workdir.is_dir()
        )

    @property
    def missing(self) -> str:
        """返回第一个缺失项的说明；齐全则空串。供界面给出可操作的提示。"""
        if not self.workdir.is_dir():
            return f"工作目录不存在：{self.workdir}"
        if not self.python_exe.is_file():
            return f"解释器不存在：{self.python_exe}"
        if not self.script.is_file():
            return f"入口脚本不存在：{self.script}"
        return ""


def resolve_target(
    project_dir: Path | str = TRADINGAGENTS_DIR,
) -> LaunchTarget:
    """解析解释器、入口脚本与工作目录。"""
    d = Path(project_dir)
    return LaunchTarget(
        python_exe=d / VENV_PYTHON,
        script=d / ENTRY_SCRIPT,
        workdir=d,
    )


def is_running(proc_name: str = PROC_NAME) -> int | None:
    """检测是否已有同名进程在跑；返回 PID，未运行返回 None。

    用 ``tasklist`` 而非 psutil——项目不应为这个功能引入新依赖，
    且 Windows 上 tasklist 在系统层面总可用。

    ⚠️ 编码陷阱：简体中文 Windows 的 tasklist 输出是 **GBK**
    （未找到时返回"信息: 没有找到..."的 GBK 字节），不是 UTF-8。
    早先用 ``text=True``（默认按 UTF-8 解码）会抛 UnicodeDecodeError，
    被 except 吞掉后返回 None —— **重复启动保护会静默失效**。
    因此这里显式用 ``encoding="gbk", errors="replace"``。
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {proc_name}", "/FO", "CSV", "/NH"],
            capture_output=True,
            timeout=10,
            check=False,
            creationflags=getprocess_no_window(),
            encoding="gbk",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("tasklist 调用失败（按未运行处理）: %s", exc)
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    for line in out.stdout.splitlines():
        parts = [p.strip('" ') for p in line.split('","')]
        if len(parts) >= 2 and parts[0].lower() == proc_name.lower():
            try:
                return int(parts[1])
            except ValueError:
                continue
    return None


def find_ta_process() -> int | None:
    """精确定位 TradingAgents 进程，返回 PID；未运行返回 None。

    为什么不能只用进程名
    --------------------
    目标进程是 ``.venv\\Scripts\\python.exe main.py``，进程名就是
    ``python.exe``——本机常驻好几个 python（PA_Agent 自己就是），
    按名字查必然误报，把"没在跑"当成"已在跑"，按钮就永远点不动。

    实现选择（按可靠性排序，实测过）
    ------------------------------
    1. **PowerShell ``Get-CimInstance``**（当前采用）：Win10/11 通吃，
       能拿到完整 CommandLine，实测精准（启动后命中、终止后清空）。
    2. ``wmic process``：**Win11 已移除**，实测 ``WinError 2``。
    3. ``tasklist``：只有进程名，无法区分，会误报 —— 仅作最后兜底。

    每次调用约0.3~0.9 秒（PowerShell 冷启动较慢），所以只在用户点按钮时
    调用，不放进高频路径。
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             _PS_PROC_SCRIPT],
            capture_output=True,
            timeout=30,
            check=False,
            creationflags=getprocess_no_window(),
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if ENTRY_SCRIPT.lower() not in line.lower():
                    continue
                low = line.lower()
                if "tradingagents" not in low or "python" not in low:
                    continue
                pid_s, _, _cmd = line.partition("|")
                try:
                    return int(pid_s.strip())
                except ValueError:
                    continue
            return None
        logger.debug("PowerShell 查询失败（回退 tasklist）: rc=%s", out.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("PowerShell 调用失败（回退 tasklist）: %s", exc)

    # 兜底：wmic（老系统可能有），再退到 tasklist
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, timeout=15, check=False,
            creationflags=getprocess_no_window(),
            encoding="gbk", errors="replace",
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                low = line.lower()
                if ENTRY_SCRIPT.lower() in low and "tradingagents" in low:
                    for c in reversed([x.strip() for x in line.split(",")]):
                        if c.isdigit():
                            return int(c)
            return None
    except (OSError, subprocess.SubprocessError):
        pass

    return None


#: PowerShell 查询脚本：输出 "PID|CommandLine"，供Python 侧解析。
#: 用 ForEach-Object 手拼而非 ConvertTo-Csv——后者的引号转义规则在不同
#: PowerShell 版本下不一致，解析容易出错。
_PS_PROC_SCRIPT = (
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"
)


def getprocess_no_window() -> int:
    """Windows 下隐藏子进程控制台；非 Windows 返回 0。"""
    import os

    return getattr(os, "CREATE_NO_WINDOW", 0)


def build_command(target: LaunchTarget) -> list[str]:
    """构造命令行：``<venv python> main.py``。

    用 venv 解释器而非 exe——``main.py`` 是脚本，必须由解释器跑；
    且这样依赖与该项目的 ``.venv`` 完全一致，不会污染本程序环境。
    """
    return [str(target.python_exe), str(target.script)]
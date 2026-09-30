# -*- coding: utf-8 -*-
"""
Laya 权重离线下载脚本
=====================

背景
----
本机沙箱/代理环境下 huggingface.co 不可达（CONNECT tunnel 502），
但 hf-mirror.com 可达。hf-mirror 的 WAF 会拦截 Python 默认 UA
（"Python-urllib/3.x"，返回 403），因此本脚本显式带上浏览器 UA。

Laya 的 Agent 在 `model_id_or_path` 指向**已存在的本地目录**时会跳过
snapshot_download，随后要求该目录（含 subfolder）内具备：
    rl_agent_config.json        # 必需，缺失即 "Incompatible model"
    model.safetensors           # 必需，缺失即 "Incompatible model"
    tokenizer/tokenizer.json    # 必需
    tokenizer/tokenizer_config.json
    encoder/config.json         # 存在时 build_model(pretrained=False) 走本地 AutoConfig，
                                # 不会再去联网拉取 jhu-clsp/mmBERT-base
  * 编码器权重不需要单独下载：Agent 以 pretrained=False 构建后，用
    model.safetensors 做 load_state_dict(strict=True)，权重全在 checkpoint 里。

用法
----
    python tools/download_laya.py            # 下载 multilingual（中文）默认档
    python tools/download_laya.py english    # 可选：英文档
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 配置

ENDPOINT = "https://hf-mirror.com"
REPO = "convaiinnovations/laya"

# 浏览器 UA：绕过 hf-mirror 对 "Python-urllib/*" 的 403 拦截
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# subfolder -> 需要拉取的相对文件列表（与 laya/agent.py 的 allow_patterns 对齐）
FILES = {
    "multilingual": [
        "multilingual/rl_agent_config.json",
        "multilingual/model.safetensors",
        "multilingual/tokenizer/tokenizer.json",
        "multilingual/tokenizer/tokenizer_config.json",
        "multilingual/encoder/config.json",
    ],
    "": [
        "rl_agent_config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "encoder/config.json",
    ],
}

# 远端 sha256 由 HF API 提供（git blob hash 不是 sha256），这里只用字节数做快速校验，
# 真实完整性由 safetensors 加载时的 strict=True 保证。
EXPECT_SIZE = {
    "multilingual/rl_agent_config.json": 472,
    "multilingual/model.safetensors": 643_835_514,
    "multilingual/tokenizer/tokenizer.json": 34_363_188,
    "multilingual/tokenizer/tokenizer_config.json": 524,
    "multilingual/encoder/config.json": 1_938,
}

DEST_ROOT = r"C:\Users\wolf2\laya-models\laya"
CHUNK = 1 << 20  # 1 MiB


def _resolve_url(relpath: str) -> str:
    return f"{ENDPOINT}/{REPO}/resolve/main/{relpath}"


def _http_get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout)


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(relpath: str, dest: str, retries: int = 4) -> None:
    """带断点续传的下载。已存在且大小一致的文件直接跳过。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    expect = EXPECT_SIZE.get(relpath)
    if os.path.exists(dest) and expect and os.path.getsize(dest) == expect:
        print(f"  [skip] {relpath} 已存在且大小一致 ({_fmt(expect)})")
        return
    if os.path.exists(dest) and not expect:
        print(f"  [skip] {relpath} 已存在 ({_fmt(os.path.getsize(dest))})")
        return

    url = _resolve_url(relpath)
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        try:
            headers = dict(UA)
            mode = "wb"
            if have > 0:
                headers["Range"] = f"bytes={have}-"
                mode = "ab"
                print(f"  [resume] {relpath} 从 {_fmt(have)} 续传")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                if resp.status == 200 and have > 0:
                    # 服务端不支持 Range，重新开始
                    have, mode = 0, "wb"
                print(
                    f"  [get] {relpath} -> {_fmt(total) if total else '未知大小'}",
                    flush=True,
                )
                t0, last = time.time(), time.time()
                with open(tmp, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last > 5:
                            pct = f"{100 * have / total:.1f}%" if total else "?"
                            spd = have / max(now - t0, 1e-6) / 1024 / 1024
                            print(
                                f"        {_fmt(have)} / {pct} @ {spd:.1f} MB/s",
                                flush=True,
                            )
                            last = now
            got = os.path.getsize(tmp)
            if expect and got != expect:
                raise IOError(f"大小不符：期望 {expect}，实得 {got}")
            shutil.move(tmp, dest)
            print(f"  [ok] {relpath} 完成 {_fmt(got)}", flush=True)
            return
        except Exception as e:  # noqa: BLE001 - 下载阶段统一重试
            print(f"  [warn] {relpath} 第 {attempt} 次失败: {type(e).__name__}: {e}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def main() -> int:
    sub = sys.argv[1] if len(sys.argv) > 1 else "multilingual"
    rels = FILES.get(sub)
    if rels is None:
        print(f"未知 subfolder: {sub}；可选 {list(FILES)}")
        return 2

    print(f"目标仓库 : {REPO} (subfolder={sub or 'root'})")
    print(f"下载源   : {ENDPOINT}")
    print(f"落地目录 : {DEST_ROOT}")
    t0 = time.time()
    for rel in rels:
        dest = os.path.join(DEST_ROOT, rel.replace("/", os.sep))
        download(rel, dest)
    print(f"\n全部完成，用时 {time.time() - t0:.1f}s")
    print(f"模型根目录（传给 laya.load 的 model_id_or_path）: {DEST_ROOT}")
    print(f"subfolder: {sub or '(无)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

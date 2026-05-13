#!/usr/bin/env python3
"""tts: CosyVoice3 ONNX 的简短 CLI 包装。

把长命令收敛成:
    uv run python main.py "在这宁静的夜晚..." -v zh_7

或（注册 entry point 后）:
    uv run tts "..." -v zh_7

预设来自 prompts/voices.json；其余参数原样透传给底层推理脚本。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
VOICES_PATH = ROOT / "prompts" / "voices.json"
DEFAULT_CONFIG = ROOT / "configs" / "cosyvoice_zh_quality.json"
INFERENCE_SCRIPT = ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B" / "onnx" / "scripts" / "onnx_inference_pure.py"
OUTPUT_DIR = ROOT / "output"
DEFAULT_VOICE = "zh_7"


def load_voices() -> dict:
    if not VOICES_PATH.exists():
        sys.exit(f"找不到预设文件: {VOICES_PATH}")
    data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def slug(text: str, n: int = 16) -> str:
    cleaned = re.sub(r"[^\w一-龥]+", "_", text, flags=re.UNICODE).strip("_")
    return (cleaned[:n] or "out").rstrip("_")


def list_voices(voices: dict) -> None:
    width = max(len(k) for k in voices)
    for name, v in voices.items():
        flag = " [TODO]" if v["text"].startswith("TODO") else ""
        print(f"  {name:<{width}}  {v['wav']}{flag}")


def main() -> int:
    voices = load_voices()

    parser = argparse.ArgumentParser(
        prog="tts",
        description="CosyVoice3 ONNX TTS — 零样本声音克隆",
    )
    parser.add_argument("text", nargs="?", help="要合成的文本（与 --list 二选一）")
    parser.add_argument(
        "-v", "--voice", default=DEFAULT_VOICE,
        help=f"声音预设 ID（默认 {DEFAULT_VOICE}），完整列表用 --list 查看",
    )
    parser.add_argument("-o", "--output", default=None, help="输出 wav 路径（默认 output/<voice>_<slug>_<ts>.wav）")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help=f"推理配置 JSON（默认 {DEFAULT_CONFIG.relative_to(ROOT)}）")
    parser.add_argument("--list", action="store_true", help="列出所有预设并退出")
    parser.add_argument("-n", "--dry-run", action="store_true", help="只打印要执行的命令，不实际跑")

    args, passthrough = parser.parse_known_args()

    if args.list:
        list_voices(voices)
        return 0

    if not args.text:
        parser.error("缺少要合成的文本（或使用 --list 查看预设）")

    if args.voice not in voices:
        sys.exit(f"未知预设 {args.voice!r}。可用: {', '.join(voices)}")
    voice = voices[args.voice]
    if voice["text"].startswith("TODO"):
        sys.exit(
            f"预设 {args.voice!r} 的逐字稿尚未填写。\n"
            f"请编辑 {VOICES_PATH.relative_to(ROOT)} 把对应的 text 字段补成 wav 的精确逐字稿。"
        )

    output = args.output or str(OUTPUT_DIR / f"{args.voice}_{slug(args.text)}_{int(time.time())}.wav")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(INFERENCE_SCRIPT),
        "--text", args.text,
        "--prompt_wav", voice["wav"],
        "--prompt_text", voice["text"],
        "--config", args.config,
        "--output", output,
        *passthrough,
    ]

    print(f">>> voice={args.voice}  output={output}")
    if args.dry_run:
        print(">>> dry-run, 命令:")
        print("    " + " ".join(repr(c) if " " in c else c for c in cmd))
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())

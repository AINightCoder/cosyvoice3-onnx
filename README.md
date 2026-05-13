# cosyvoice3-onnx

纯 ONNX Runtime 推理版的 CosyVoice3 零样本语音克隆（zero-shot voice cloning）TTS。无需 PyTorch，CPU 即可运行。

模型来源：[Fun-CosyVoice3-0.5B](https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) 的 ONNX 导出版本（[`ayousanz/cosy-voice3-onnx`](https://huggingface.co/ayousanz/cosy-voice3-onnx)）。

## 特性

- **零 PyTorch 依赖**：只用 `onnxruntime` + `numpy` + `librosa` 跑全流程（LLM / Flow / HiFT）。
- **零样本声音克隆**：给一段 3–10 秒参考音频 + 它的逐字稿，即可用该音色合成任意文本。
- **多语言**：自动识别中文 / 英文 / 日文 / 韩文 / 德 / 西 / 法 / 意 / 俄 等，**不要**写 `<|zh|>` 这类语言标签。
- **可调推理参数**：采样、长度、流匹配步数等通过 JSON 配置或 CLI 覆盖，方便针对长文本 / 短文本调优。
- **`tts` 短命令**：内置 CLI 把声音预设（wav + 逐字稿）固化在 `prompts/voices.json`，调用时只需 `uv run tts "<文本>" -v <预设>`，无需再手抄逐字稿。

## 环境

- Python ≥ 3.10（实测 3.10.16）
- 关键依赖版本固定，**不要随手升级**：

  | 依赖             | 版本       |
  | ---------------- | ---------- |
  | `onnxruntime`  | `1.18.0` |
  | `numpy`        | `1.26.4` |
  | `librosa`      | `0.10.2` |
  | `transformers` | `4.51.3` |
  | `scipy`        | `1.13.1` |
  | `soundfile`    | `0.12.1` |

  `onnxruntime 1.19+` 可能在 FP16 ONNX 上不兼容，`numpy 2.x` 会破坏依赖链。

## 安装

```powershell
git clone https://github.com/AINightCoder/cosyvoice3-onnx.git
cd cosyvoice3-onnx
uv sync
```

下载 ONNX 模型（约 3.8 GB，不入 git）：

```powershell
# 可选：国内加速
$env:HF_ENDPOINT="https://hf-mirror.com"

uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ayousanz/cosy-voice3-onnx', local_dir='pretrained_models/Fun-CosyVoice3-0.5B/onnx')"
```

## 快速开始

声音预设登记在 [`prompts/voices.json`](prompts/voices.json)（`name -> { wav, text }`）。`uv sync` 之后用 `tts` 命令合成：

```powershell
# 列出所有预设
uv run tts --list

# 用 zh_7 预设合成；输出默认落到 output/zh_7_<slug>_<时间戳>.wav
uv run tts "在这宁静的夜晚，我们可以沿着小路慢慢走，感受微风拂面的轻柔，与自然融为一体。" -v zh_7

# 显式指定输出文件、覆盖配置、覆盖任意推理参数
uv run tts "短句也能跑。" -v zh_7 -o output/demo.wav -c configs/cosyvoice_zh_quality.json --temperature 0.7

# 预演（只打印底层命令，不实际推理）
uv run tts "..." -v zh_7 -n
```

新增自己的声音：把 wav 放到 `prompts/`，在 `voices.json` 里加一条 `{ "wav": "...", "text": "<逐字稿>" }` 即可。

> **关键**：`text` 必须是 `wav` 的精确逐字稿。哪怕只差几个字，模型也会"幻觉"，把 prompt_text 内容混入输出。

<details>
<summary>底层调用（不经过 <code>tts</code> 包装）</summary>

```powershell
uv run python "pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py" `
  --config "configs/cosyvoice_zh_quality.json" `
  --text "在这宁静的夜晚，我们可以沿着小路慢慢走，感受微风拂面的轻柔，与自然融为一体。" `
  --prompt_wav "prompts/Zh_7_prompt.wav" `
  --prompt_text "今夜的月光如此清亮，不做些什么真是浪费。随我一同去月下漫步吧，不许拒绝。" `
  --output "output/output_test.wav"
```

`tts` 只是把预设展开成等价的 `--prompt_wav` / `--prompt_text`，未被识别的参数原样透传给该脚本。
</details>

## 推理参数（`configs/*.json`）

CLI 参数会覆盖配置文件。常用项：

| 参数                  | 默认      | 含义                                                           |
| --------------------- | --------- | -------------------------------------------------------------- |
| `sampling_k`        | `25`    | Top-k 采样。`1` 为 greedy（最稳定但单调）                    |
| `temperature`       | `1.0`   | 采样温度。短文本建议 0.6–0.8                                  |
| `max_len`           | `500`   | 生成 speech token 数上限（1 token ≈ 40 ms）                   |
| `max_len_ratio`     | `20.0`  | 每个 tts text token 最多对应几个 speech token                  |
| `min_len_ratio`     | `2.0`   | 每个 tts text token 至少对应几个 speech token（影响 EOS 触发） |
| `max_silent_tokens` | `5`     | 连续静音/呼吸 token 上限                                       |
| `flow_steps`        | `10`    | Flow Matching 步数。`20–30` 音色更自然但 CPU 慢             |
| `trim_silence`      | `false` | 去除首尾低能量段                                               |
| `speed`             | `1.0`   | 语速倍率，> 1 更快                                             |
| `seed`              | `null`  | 随机种子，固定后可复现                                         |

### 输出多读 prompt_text / 漏掉 tts_text 后半段

zero-shot 模式典型故障，定位顺序：

1. **prompt_text 与 prompt_wav 是否严格匹配** —— 不匹配会触发幻觉，多读 prompt 内容。
2. **`max_len` 是否够大** —— 70 字中文 ≈ 至少 280 token；`max_len=200` 会硬截断。建议 `max_len=999`。
3. **调小 `min_len_ratio`** —— 从 `2.0` 降到 `0.5–1.0`，让模型 EOS 能更早触发，避免"硬撑"导致跑去复读 prompt。

## 项目结构

```
cosyvoice3-onnx/
├── main.py                           # `tts` CLI 入口（薄包装，转发到下方推理脚本）
├── configs/
│   └── cosyvoice_zh_quality.json     # 推理配置示例
├── docs/
│   └── usage.md                      # 完整使用教程（含性能/常见问题）
├── prompts/                          # 参考音频 + 声音预设登记
│   └── voices.json                   # name -> { wav, text }，由 `tts -v <name>` 使用
├── pretrained_models/
│   └── Fun-CosyVoice3-0.5B/onnx/
│       ├── *.onnx                    # 14 个 ONNX 模型（从 HF 下载，不入 git）
│       ├── vocab.json
│       ├── merges.txt
│       ├── tokenizer_config.json
│       └── scripts/
│           └── onnx_inference_pure.py
└── output/                           # 生成的 wav
```

## 性能

CPU 上跑 2.6 秒音频耗时约 30 秒（RTF ≈ 12），适合离线批量生成，不适合实时对话。GPU 加速：脚本会自动检测 `CUDAExecutionProvider`，安装对应 onnxruntime-gpu 即可。

## 更多

完整使用教程、常见故障排查见 [`docs/usage.md`](docs/usage.md)。

## License

Apache 2.0（继承自上游 CosyVoice3）。

# CosyVoice3 ONNX CPU 使用教程

本文记录本机已部署好的 `cosyvoice3-onnx` 的日常使用方法、声音克隆方式和常见坑。

## 1. 当前部署信息

- 项目目录：`D:\Git\AI\cosyvoice3-onnx`
- Python：`3.10.16`
- 推理方式：ONNX Runtime CPU，非 PyTorch
- 模型目录：`pretrained_models/Fun-CosyVoice3-0.5B/onnx`
- 推理脚本：`pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py`

关键依赖已固定：

| 依赖 | 版本 |
| --- | --- |
| `onnxruntime` | `1.18.0` |
| `numpy` | `1.26.4` |
| `librosa` | `0.10.2` |
| `transformers` | `4.51.3` |
| `scipy` | `1.13.1` |
| `soundfile` | `0.12.1` |

> 不要随手升级 `onnxruntime` 和 `numpy`。`onnxruntime 1.19+` 可能遇到 FP16 ONNX 兼容问题，`numpy 2.x` 可能导致依赖不兼容。

## 2. 最快开始

打开 PowerShell：

```powershell
cd D:\Git\AI\cosyvoice3-onnx
```

运行一个英文测试：

```powershell
uv run python "pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py" --text "Hello, this is a short CPU test." --prompt_wav "pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav" --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." --output "output.wav"
```

生成结果：

```text
D:\Git\AI\cosyvoice3-onnx\output.wav
```

## 3. 中文合成

中文直接把 `--text` 换成中文即可，不要加语言标签。

```powershell
uv run python "pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py" --text "你好，这是一次中文语音合成测试。" --prompt_wav "pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav" --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." --output "output_zh.wav"
```

注意：不要写成 `<|zh|>你好`，语言标签可能会被模型直接读出来。

## 4. 使用内置参考音频

模型目录自带两个参考音频：

- 女声：`prompts/en_female_nova_greeting.wav`
- 男声：`prompts/en_male_onyx_greeting.wav`

已验证可直接使用的是女声示例：

```powershell
uv run python "pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py" --text "This is a female voice test." --prompt_wav "pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav" --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." --output "output_female.wav"
```

男声文件也存在，但当前模型 README 没有提供可验证的男声逐字稿；使用前需要先确认 `en_male_onyx_greeting.wav` 的真实内容，再把逐字稿填入 `--prompt_text`。如果 `prompt_text` 与音频不匹配，音色相似度会明显下降。

## 5. 克隆自己的声音

准备两个东西：

1. 一段清晰的参考音频，建议 `3-10` 秒，WAV 格式更稳。
2. 这段音频的逐字稿，必须和音频内容严格一致。

推荐目录：

```text
D:\Git\AI\cosyvoice3-onnx\voices\my_voice.wav
```

运行示例：

```powershell
uv run python "pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py" --text "这是用我自己的声音生成的一段测试音频。" --prompt_wav "voices/my_voice.wav" --prompt_text "这里填写 my_voice.wav 里面真实说出的逐字内容。" --output "output_my_voice.wav"
```

参考音频建议：

- 环境安静，无音乐、无混响、无明显噪声。
- 只保留一个人的声音。
- 不要太短，少于 3 秒容易不像；不要太长，超过 10 秒会拖慢推理。
- `prompt_text` 要逐字匹配，包括中英文、数字读法、停顿处的标点。

## 6. 输出文件命名建议

短期测试可以直接输出到项目根目录：

```text
output.wav
output_zh.wav
output_my_voice.wav
```

正式使用建议新建 `outputs` 目录：

```powershell
mkdir outputs
```

然后指定输出：

```powershell
uv run python "pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py" --text "保存到 outputs 目录。" --prompt_wav "pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav" --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." --output "outputs/test.wav"
```

## 7. CPU 性能预期

本机已验证结果：

```text
输入文本：Hello, this is a short CPU test.
输出音频：2.60 秒
总耗时：31.53 秒
RTF：12.12
```

结论：CPU 可以跑通，适合离线生成短音频，不适合实时对话或高并发服务。

## 8. 常见问题

### 下载依赖很慢或超时

使用清华 PyPI 镜像安装：

```powershell
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "onnxruntime==1.18.0" "numpy==1.26.4" "soundfile==0.12.1" "librosa==0.10.2" "transformers==4.51.3" "scipy==1.13.1" "huggingface_hub>=0.30.0"
```

### 模型下载慢

使用 Hugging Face 镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ayousanz/cosy-voice3-onnx', local_dir='pretrained_models/Fun-CosyVoice3-0.5B/onnx')"
```

### 出现 NumPy 2.x 兼容错误

检查版本：

```powershell
uv run python -c "import numpy; print(numpy.__version__)"
```

如果不是 `1.26.4`，改回固定版本：

```powershell
uv pip install "numpy==1.26.4"
```

### 出现 ONNX Runtime FP16 相关错误

检查版本：

```powershell
uv run python -c "import onnxruntime; print(onnxruntime.__version__)"
```

如果不是 `1.18.0`，改回固定版本：

```powershell
uv pip install "onnxruntime==1.18.0"
```

### 生成声音不像

优先检查参考音频和逐字稿：

- 音频是否清晰。
- 是否只有一个说话人。
- `prompt_text` 是否与音频逐字一致。
- 参考音频是否在 `3-10` 秒之间。

### 语言标签被读出来

不要在文本里写 `<|zh|>`、`<|en|>`、`<|ja|>` 之类标签。CosyVoice3 会自动判断语言。

## 9. 常用检查命令

检查 Python 和关键依赖：

```powershell
uv run python -c "import sys, onnxruntime, numpy; print(sys.version); print('onnxruntime', onnxruntime.__version__); print('numpy', numpy.__version__)"
```

检查输出音频信息：

```powershell
uv run python -c "import soundfile as sf; print(sf.info('output.wav'))"
```

## 10. 推荐日常工作流

1. 准备 `--text`。
2. 选择内置参考音频，或准备自己的 `prompt_wav`。
3. 填写与参考音频严格一致的 `prompt_text`。
4. 指定 `--output`。
5. 运行命令并试听 WAV。
6. 如果音色不像，先换更干净的参考音频，而不是调依赖版本。

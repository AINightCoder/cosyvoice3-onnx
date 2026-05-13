---
license: apache-2.0
language:
- en
- zh
- ja
- ko
- de
- es
- fr
- it
- ru
tags:
- text-to-speech
- tts
- onnx
- voice-cloning
- cosyvoice
library_name: onnxruntime
pipeline_tag: text-to-speech
base_model:
- FunAudioLLM/Fun-CosyVoice3-0.5B-2512
---

# CosyVoice3 ONNX Models

ONNX-exported models for [Fun-CosyVoice3-0.5B](https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512), enabling PyTorch-free inference.

## Model Description

CosyVoice3 is a multilingual text-to-speech (TTS) system with zero-shot voice cloning capabilities. This repository contains ONNX-converted models for pure ONNX Runtime inference without PyTorch dependencies.

### Supported Languages

CosyVoice3 automatically detects the language from text content. Supported languages include:
- English, Chinese, Japanese, Korean
- German, Spanish, French, Italian, Russian
- Cantonese and other Chinese dialects

**Important**: Do NOT use language tags like `<|en|>` or `<|ja|>` - they will be pronounced as literal text.

## Repository Contents

This repository includes **everything needed for inference** (no additional downloads required):

```
.
├── *.onnx                              # ONNX model files (14 files, ~3.8GB total)
├── vocab.json                          # Qwen2 tokenizer vocabulary
├── merges.txt                          # Qwen2 tokenizer merges
├── tokenizer_config.json               # Qwen2 tokenizer config
├── scripts/
│   └── onnx_inference_pure.py          # Inference script
└── prompts/
    ├── en_female_nova_greeting.wav     # Sample female voice
    └── en_male_onyx_greeting.wav       # Sample male voice
```

## Model Files

### ONNX Models (This Repository)

| File | Size | Precision | Description |
|------|------|-----------|-------------|
| `campplus.onnx` | 28MB | FP32 | Speaker embedding extraction |
| `speech_tokenizer_v3.onnx` | 969MB | FP32 | Speech tokenization |
| `text_embedding_fp32.onnx` | 544MB | FP32 | Text token embedding |
| `llm_backbone_initial_fp16.onnx` | 717MB | FP16 | LLM initial pass (KV cache generation) |
| `llm_backbone_decode_fp16.onnx` | 717MB | FP16 | LLM decode step |
| `llm_decoder_fp16.onnx` | 12MB | FP16 | Logits output |
| `llm_speech_embedding_fp16.onnx` | 12MB | FP16 | Speech token embedding |
| `flow_token_embedding_fp16.onnx` | 1MB | FP16 | Flow token embedding |
| `flow_pre_lookahead_fp16.onnx` | 1MB | FP16 | Flow pre-lookahead |
| `flow_speaker_projection_fp16.onnx` | 31KB | FP16 | Speaker projection |
| `flow.decoder.estimator.fp16.onnx` | 664MB | FP16 | Flow DiT (Diffusion Transformer) |
| `hift_f0_predictor_fp32.onnx` | 13MB | FP32 | F0 prediction |
| `hift_source_generator_fp32.onnx` | 259MB | FP32 | Source signal generation |
| `hift_decoder_fp32.onnx` | 70MB | FP32 | HiFT decoder (waveform generation) |

### Tokenizer Files (Included)

Qwen2 tokenizer files are **included in this repository**:
- `vocab.json` - Vocabulary (3.54MB)
- `merges.txt` - BPE merges (1.54MB)
- `tokenizer_config.json` - Configuration

No additional downloads are required.

## Architecture

```
1. Prompt Audio Processing
   ├── campplus.onnx → Speaker embedding (192-dim)
   ├── speech_tokenizer_v3.onnx → Speech tokens (for LLM context)
   └── librosa → Mel spectrogram (for Flow conditioning)

2. LLM Inference (Zero-Shot Mode)
   ├── text_embedding → [prompt_text + tts_text] embedding
   ├── llm_speech_embedding → Prompt speech token embedding
   ├── llm_backbone_initial → Initial pass (KV cache)
   ├── llm_backbone_decode → Decode steps (loop)
   └── llm_decoder → Logits → Token sampling

3. Flow Inference (Mel Generation)
   ├── flow_token_embedding → Token embedding
   ├── flow_pre_lookahead → Feature extraction
   ├── flow_speaker_projection → Speaker projection
   └── flow.decoder.estimator → DiT (10-step Euler)

4. HiFT Inference (Waveform)
   ├── hift_f0_predictor → F0 prediction
   ├── hift_source_generator → Source signal
   ├── STFT → Spectral decomposition
   ├── hift_decoder → Magnitude/phase prediction
   └── ISTFT → Waveform reconstruction
```

## Quick Start

### 1. Create Environment

```bash
uv init cosyvoice-onnx --python 3.10
cd cosyvoice-onnx
uv add "onnxruntime==1.18.0" "numpy==1.26.4" "soundfile==0.12.1" "librosa==0.10.2" "transformers==4.51.3" "scipy==1.13.1" "huggingface_hub>=0.30.0"
```

**Note**: The `--python 3.10` flag is required because onnxruntime 1.18.0 only supports Python 3.8-3.12.

### 2. Download Models

```bash
# Download everything (ONNX models + tokenizer + inference script + sample prompts)
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ayousanz/cosy-voice3-onnx', local_dir='pretrained_models/Fun-CosyVoice3-0.5B/onnx')"
```

**Note**: All required files including the Qwen2 tokenizer are included in this repository. No additional downloads needed.

### 3. Run Inference

```bash
# English
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "Hello, this is a test." \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav \
    --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." \
    --output output.wav

# Japanese
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "こんにちは、今日はいい天気ですね。" \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav \
    --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." \
    --output output_ja.wav
```

## Detailed Setup

### Version Requirements

**Important**: The original CosyVoice (PyTorch version) also has the same version constraints. This is not specific to ONNX inference.

| Package | Version | Purpose |
|---------|---------|---------|
| `onnxruntime` | 1.18.0 | ONNX inference engine (newer versions have FP16 issues) |
| `numpy` | 1.26.4 | Numerical computation (1.x required) |
| `soundfile` | 0.12.1 | WAV file output |
| `librosa` | 0.10.2 | Audio loading, mel spectrogram extraction |
| `transformers` | 4.51.3 | Qwen2 tokenizer |
| `scipy` | 1.13.1 | Signal processing |
| `huggingface_hub` | >=0.30.0 | Download from Hugging Face |

### GPU Support (Optional)

```bash
uv remove onnxruntime && uv add "onnxruntime-gpu==1.18.0"
```

**Note**: Requires CUDA 11.8 or 12.x with cuDNN 8.x.

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--text` | Yes | Text to synthesize (do NOT include language tags) |
| `--prompt_wav` | Yes | Prompt audio file path |
| `--prompt_text` | Yes | Transcript of the prompt audio |
| `--output` | No | Output file path (default: output_onnx_pure.wav) |
| `--model_dir` | No | Model directory (default: pretrained_models/Fun-CosyVoice3-0.5B) |
| `--fp32` | No | Use FP32 precision (default: FP16) |

### Important: Zero-Shot Voice Cloning

CosyVoice3 uses **zero-shot voice cloning**. Both prompt audio AND prompt text are required:

- **Prompt audio**: Reference voice sample (3-10 seconds recommended)
- **Prompt text**: Transcript of what is spoken in the prompt audio

This provides better voice cloning quality than cross-lingual mode.

## Performance

| Phase | CPU Time |
|-------|----------|
| Prompt Processing | 2-3s |
| LLM Inference | 40-160s |
| Flow Inference | 30-60s |
| HiFT Inference | 1-3s |

**Note**: CPU-only inference. GPU (CUDA) significantly improves performance.

## Technical Notes

### Pure ONNX Inference (PyTorch-Free)

This inference script runs **completely without PyTorch**. All processing is implemented using ONNX Runtime and NumPy/SciPy:
- Neural network inference: ONNX Runtime
- STFT/ISTFT: NumPy + SciPy (not PyTorch)
- Audio processing: librosa

### HiFT Parameters (CosyVoice3 Specific)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `upsample_rates` | [8, 5, 3] | HiFT upsampling rates (120x total) |
| `n_fft` | 16 | FFT window size |
| `hop_length` | 4 | Hop length |
| `center` | True | Signal padding (PyTorch-compatible) |

**Note**: CosyVoice2 uses upsample_rates=[8, 8] (64x), but CosyVoice3 uses [8, 5, 3] (120x).

Expected STFT frames = mel_frames × 120 + 1

### Precision Selection

- **FP16**: LLM and Flow components (memory efficient)
- **FP32**: HiFT components (numerical stability required), text embedding, speaker models

### KV Cache

LLM uses split KV cache architecture:
- `llm_backbone_initial`: Generates initial KV cache from full context
- `llm_backbone_decode`: Updates KV cache with single token per step

## Troubleshooting

### ONNX Runtime version error with FP16 models

```
RuntimeException: Attempting to get index by a name which does not exist
```

**Solution**: Use `onnxruntime==1.18.0`. Newer versions (1.19+) have compatibility issues with FP16 models.

### NumPy 2.x incompatibility

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Solution**: Use `numpy==1.26.4`. This is a constraint shared with the original CosyVoice.

### Tokenizer loading issues

If you encounter tokenizer loading errors, ensure you downloaded the complete repository including:
- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`

Re-download with:
```bash
huggingface-cli download ayousanz/cosy-voice3-onnx --local-dir pretrained_models/Fun-CosyVoice3-0.5B/onnx
```

### Language tags being pronounced

If you hear "<|en|>" or similar being spoken, remove the language tags from your text. CosyVoice3 automatically detects language.

## License

Apache 2.0 (same as original CosyVoice)

## Credits

- Original model: [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- CosyVoice repository: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

## Related Links

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [ModelScope Model](https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [Paper: CosyVoice](https://arxiv.org/abs/2407.05407)
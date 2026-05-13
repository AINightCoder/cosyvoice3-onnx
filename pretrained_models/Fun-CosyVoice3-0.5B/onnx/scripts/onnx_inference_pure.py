#!/usr/bin/env python3
# Copyright (c) 2024 CosyVoice Unity ONNX Export
# Apache License 2.0

"""
Pure ONNX CosyVoice3 Inference (No PyTorch model loading) - Zero-Shot Mode

This script performs TTS inference using ONLY ONNX models in zero-shot voice cloning mode.
Suitable for porting to Unity Sentis.

IMPORTANT: CosyVoice is a voice cloning TTS system. Both prompt audio AND prompt text
are REQUIRED for proper inference. This uses the zero-shot mode which provides better
voice cloning quality than cross-lingual mode.

NOTE: Do NOT include language tags like <|en|> or <|ja|> in the text. CosyVoice3
automatically detects the language from the text content. Language tags will be
pronounced as literal text.

Usage:
    python scripts/onnx_inference_pure.py \\
        --text "Hello world" \\
        --prompt_wav asset/prompts/en_female_nova_greeting.wav \\
        --prompt_text "Hello, my name is Sarah."

    python scripts/onnx_inference_pure.py \\
        --text "こんにちは、今日はいい天気ですね。" \\
        --prompt_wav my_voice.wav \\
        --prompt_text "Recording text that matches the prompt audio"

Requirements:
    - Prompt audio: 3-10 seconds recommended, clear speech, minimal noise
    - Prompt text: Transcript of the prompt audio content
"""

import argparse
import os
import sys
from pathlib import Path
import time
import json

import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

# Add project root to path (only for tokenizer)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PureOnnxCosyVoice3:
    """Pure ONNX-based CosyVoice3 inference (no PyTorch)"""

    def __init__(self, model_dir: str, use_fp16: bool = True):
        self.model_dir = model_dir
        self.onnx_dir = os.path.join(model_dir, 'onnx')
        self.use_fp16 = use_fp16
        self.sample_rate = 24000

        # Model parameters for CosyVoice3
        self.hidden_dim = 896
        self.speech_token_size = 6561
        self.sos = self.speech_token_size + 0      # 6561
        self.eos_token = self.speech_token_size + 1  # 6562
        self.task_id = self.speech_token_size + 2    # 6563
        self.silent_tokens = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}

        self._load_tokenizer()
        self._load_onnx_models()

    def _load_tokenizer(self):
        """Load Qwen2 tokenizer for text processing"""
        print("Loading tokenizer...")
        from transformers import AutoTokenizer

        # Try multiple paths for tokenizer files
        qwen_path = os.path.join(self.model_dir, 'CosyVoice-BlankEN')
        onnx_tokenizer_path = self.onnx_dir  # Tokenizer files in ONNX directory

        if os.path.exists(qwen_path):
            print(f"  Loading from {qwen_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        elif os.path.exists(os.path.join(onnx_tokenizer_path, 'vocab.json')):
            print(f"  Loading from {onnx_tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(onnx_tokenizer_path, trust_remote_code=True)
        else:
            print("  Loading from Qwen/Qwen2-0.5B")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
        print("  Tokenizer loaded!")

    def _load_onnx_models(self):
        """Load all ONNX models"""
        suffix = '_fp16' if self.use_fp16 else '_fp32'

        so = ort.SessionOptions()
        so.log_severity_level = 3

        # Try to use CUDA if available and working
        providers = ['CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            try:
                # Test if CUDA actually works
                test_session = ort.InferenceSession(
                    os.path.join(self.onnx_dir, 'text_embedding_fp32.onnx'),
                    so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("  Using CUDA GPU acceleration")
                del test_session
            except Exception as e:
                print(f"  CUDA not available ({str(e)[:50]}...), using CPU")
                providers = ['CPUExecutionProvider']
        else:
            print("  Using CPU (CUDA not available)")

        print("Loading ONNX models...")

        # Text embedding
        self.text_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'text_embedding_fp32.onnx'),
            so, providers=providers
        )
        print("  Text embedding loaded")

        # Audio processing models (for prompt audio)
        # Try multiple locations: model_dir first, then onnx_dir
        campplus_path = os.path.join(self.model_dir, 'campplus.onnx')
        if not os.path.exists(campplus_path):
            campplus_path = os.path.join(self.onnx_dir, 'campplus.onnx')
        if not os.path.exists(campplus_path):
            campplus_path = os.path.join(self.onnx_dir, f'campplus{suffix}.onnx')
        if os.path.exists(campplus_path):
            self.campplus = ort.InferenceSession(campplus_path, so, providers=providers)
            print(f"  Campplus loaded from {os.path.basename(campplus_path)}")
        else:
            self.campplus = None
            print("  Campplus not found (prompt audio disabled)")

        speech_tokenizer_path = os.path.join(self.model_dir, 'speech_tokenizer_v3.onnx')
        if not os.path.exists(speech_tokenizer_path):
            speech_tokenizer_path = os.path.join(self.onnx_dir, 'speech_tokenizer_v3.onnx')
        if not os.path.exists(speech_tokenizer_path):
            speech_tokenizer_path = os.path.join(self.onnx_dir, f'speech_tokenizer_v3{suffix}.onnx')
        if os.path.exists(speech_tokenizer_path):
            self.speech_tokenizer = ort.InferenceSession(speech_tokenizer_path, so, providers=providers)
            print(f"  Speech tokenizer loaded from {os.path.basename(speech_tokenizer_path)}")
        else:
            self.speech_tokenizer = None
            print("  Speech tokenizer not found (prompt audio disabled)")

        # LLM models
        self.llm_backbone_initial = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_backbone_initial{suffix}.onnx'),
            so, providers=providers
        )
        self.llm_backbone_decode = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_backbone_decode{suffix}.onnx'),
            so, providers=providers
        )
        self.llm_decoder = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_decoder{suffix}.onnx'),
            so, providers=providers
        )
        self.llm_speech_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'llm_speech_embedding{suffix}.onnx'),
            so, providers=providers
        )
        print("  LLM models loaded")

        # Flow models
        self.flow_token_embedding = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_token_embedding{suffix}.onnx'),
            so, providers=providers
        )
        self.flow_speaker_projection = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_speaker_projection{suffix}.onnx'),
            so, providers=providers
        )
        self.flow_pre_lookahead = ort.InferenceSession(
            os.path.join(self.onnx_dir, f'flow_pre_lookahead{suffix}.onnx'),
            so, providers=providers
        )
        self.flow_estimator = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'flow.decoder.estimator.fp16.onnx') if self.use_fp16 else
            os.path.join(self.model_dir, 'flow.decoder.estimator.fp32.onnx'),
            so, providers=providers
        )
        print("  Flow models loaded")

        # HiFT models (always use FP32 for numerical stability)
        # F0 predictor has large intermediate values that cause FP16 overflow
        self.hift_f0_predictor = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'hift_f0_predictor_fp32.onnx'),
            so, providers=providers
        )
        self.hift_source_generator = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'hift_source_generator_fp32.onnx'),
            so, providers=providers
        )
        self.hift_decoder = ort.InferenceSession(
            os.path.join(self.onnx_dir, 'hift_decoder_fp32.onnx'),
            so, providers=providers
        )
        print("  HiFT models loaded")

        print("All ONNX models loaded!")

    def tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text using Qwen2 tokenizer"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return np.array([tokens], dtype=np.int64)

    def get_text_embedding(self, token_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings using ONNX"""
        return self.text_embedding.run(None, {'input_ids': token_ids})[0]

    def get_speech_embedding(self, token_ids: np.ndarray) -> np.ndarray:
        """Get speech token embeddings using ONNX"""
        return self.llm_speech_embedding.run(None, {'token': token_ids})[0]

    def extract_speaker_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from prompt audio using Campplus"""
        if self.campplus is None:
            raise RuntimeError("Campplus model not loaded")

        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = audio.astype(np.float32)

        # Compute Kaldi-style fbank features (80 mels)
        # Using librosa for compatibility
        mel = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=400, hop_length=160,
            n_mels=80, fmin=20, fmax=7600
        )
        # Log mel
        log_mel = np.log(np.maximum(mel, 1e-10))
        # Transpose to [frames, 80]
        log_mel = log_mel.T

        # Mean normalization (as in Kaldi)
        log_mel = log_mel - log_mel.mean(axis=0, keepdims=True)

        # Add batch dimension: [1, frames, 80]
        feat = log_mel[np.newaxis, :, :].astype(np.float32)

        # Run campplus
        input_name = self.campplus.get_inputs()[0].name
        embedding = self.campplus.run(None, {input_name: feat})[0]

        # Flatten to [1, 192]
        embedding = embedding.flatten()[np.newaxis, :]
        return embedding.astype(np.float32)

    def extract_speech_tokens(self, audio_path: str) -> np.ndarray:
        """Extract speech tokens from prompt audio using speech tokenizer"""
        if self.speech_tokenizer is None:
            raise RuntimeError("Speech tokenizer model not loaded")

        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = audio.astype(np.float32)

        # Compute Whisper-style log mel spectrogram (128 mels)
        # Whisper uses: n_fft=400, hop_length=160, n_mels=128
        mel = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=400, hop_length=160,
            n_mels=128, fmin=0, fmax=8000
        )
        # Log mel (Whisper style)
        log_mel = np.log10(np.maximum(mel, 1e-10))
        # Normalize
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        # Shape: [1, 128, frames]
        feat = log_mel[np.newaxis, :, :].astype(np.float32)
        feat_len = np.array([feat.shape[2]], dtype=np.int32)

        # Run speech tokenizer
        input_names = [inp.name for inp in self.speech_tokenizer.get_inputs()]
        speech_token = self.speech_tokenizer.run(None, {
            input_names[0]: feat,
            input_names[1]: feat_len
        })[0]

        # Flatten and return as [1, seq_len]
        speech_token = speech_token.flatten()[np.newaxis, :]
        return speech_token.astype(np.int64)

    def extract_speech_mel(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram from prompt audio for flow conditioning"""
        # Load audio at 24kHz (CosyVoice native rate)
        audio, sr = librosa.load(audio_path, sr=24000)
        audio = audio.astype(np.float32)

        # Compute mel spectrogram (80 mels, matching flow model)
        # CosyVoice uses hop_length=256 at 24kHz
        mel = librosa.feature.melspectrogram(
            y=audio, sr=24000, n_fft=1024, hop_length=256,
            n_mels=80, fmin=0, fmax=12000
        )
        # Log mel
        log_mel = np.log(np.maximum(mel, 1e-10))

        # Shape: [1, frames, 80]
        mel_feat = log_mel.T[np.newaxis, :, :].astype(np.float32)
        return mel_feat

    def llm_inference(
        self,
        text: str,
        prompt_text: str,
        prompt_speech_tokens: np.ndarray,
        sampling_k: int = 25,
        temperature: float = 1.0,
        max_len: int = 500,
        min_len: int = 10,
        max_len_ratio: float = 20.0,
        max_silent_tokens: int = 5
    ) -> np.ndarray:
        """Generate speech tokens using pure ONNX (zero-shot mode)

        Args:
            text: Text to synthesize (with language tag)
            prompt_text: Transcript of the prompt audio
            prompt_speech_tokens: Speech tokens from prompt audio [1, seq_len]
            sampling_k: Top-k sampling parameter
            temperature: Sampling temperature. Lower values are more deterministic.
            max_len: Maximum output length
            min_len: Minimum output length
            max_len_ratio: Maximum generated speech tokens per TTS text token
            max_silent_tokens: Maximum consecutive silent/breath tokens to keep

        Returns:
            Generated speech tokens [1, seq_len]
        """
        print("  Tokenizing text...")

        # Tokenize prompt_text and text
        prompt_text_tokens = self.tokenize_text(prompt_text)
        tts_text_tokens = self.tokenize_text(text)

        prompt_text_len = prompt_text_tokens.shape[1]
        tts_text_len = tts_text_tokens.shape[1]

        # Concatenate prompt_text and tts_text tokens (zero-shot mode)
        combined_text_tokens = np.concatenate([prompt_text_tokens, tts_text_tokens], axis=1)
        combined_text_len = combined_text_tokens.shape[1]
        print(f"    Prompt text tokens: {prompt_text_len}, TTS text tokens: {tts_text_len}, Combined: {combined_text_len}")

        # Get text embedding for combined text
        text_emb = self.get_text_embedding(combined_text_tokens)

        # Note: In PyTorch, speaker embedding goes through spk_embed_affine_layer (Linear 192->896)
        # but those weights are not saved in llm.pt checkpoint.
        # For zero-shot mode, the key information comes from prompt_text and prompt_speech_tokens,
        # so we skip the speaker embedding in LLM (use zero-length embedding like PyTorch does when embedding.shape[0]==0)

        # Get SOS and TASK_ID embeddings
        sos_emb = self.get_speech_embedding(np.array([[self.sos]], dtype=np.int64))
        task_id_emb = self.get_speech_embedding(np.array([[self.task_id]], dtype=np.int64))

        # Get prompt speech token embedding
        if prompt_speech_tokens is not None and prompt_speech_tokens.shape[1] > 0:
            prompt_speech_emb = self.get_speech_embedding(prompt_speech_tokens)
            print(f"    Prompt speech tokens: {prompt_speech_tokens.shape[1]}")
        else:
            prompt_speech_emb = np.zeros((1, 0, self.hidden_dim), dtype=np.float32)

        # Build initial input: [SOS, text_emb, TASK_ID, prompt_speech_emb] (zero-shot mode)
        # Note: We skip embedding here since spk_embed_affine_layer weights are not available
        lm_input = np.concatenate([sos_emb, text_emb, task_id_emb, prompt_speech_emb], axis=1).astype(np.float32)

        # Initial forward pass
        seq_len = lm_input.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.float32)

        initial_outputs = self.llm_backbone_initial.run(
            None, {'inputs_embeds': lm_input, 'attention_mask': attention_mask}
        )
        hidden_states = initial_outputs[0]

        # Get KV cache from initial pass (if available)
        if len(initial_outputs) > 1:
            past_key_values = initial_outputs[1]
            print(f"    Initial KV cache shape: {past_key_values.shape}")
        else:
            # Fallback: create zeros cache (old behavior, will cause issues)
            print("    WARNING: No KV cache from initial pass, using zeros")
            past_key_values = np.zeros((48, 1, 2, seq_len, 64), dtype=np.float32)

        # Get initial logits
        logits = self.llm_decoder.run(
            None, {'hidden_state': hidden_states[:, -1:, :]}
        )[0]

        # Calculate min/max length based on TTS text (not including prompt text)
        min_len = max(min_len, int(tts_text_len * 2))
        max_len = min(max_len, int(tts_text_len * max_len_ratio))
        print(f"    Generating {min_len}-{max_len} tokens...")

        out_tokens = []
        skipped_silent_tokens = 0
        silent_run = 0
        hit_max_len = True
        for i in range(max_len):
            # Sample from logits (top-k sampling). Use sampling_k <= 1 for greedy decoding.
            raw_logits = logits.squeeze()
            if sampling_k <= 1:
                token_id = int(np.argmax(raw_logits))
            else:
                logp = self._log_softmax(raw_logits / temperature)
                top_k_idx = np.argsort(logp)[-sampling_k:]
                top_k_logp = logp[top_k_idx]
                top_k_probs = self._softmax(top_k_logp)
                sampled_idx = np.random.choice(len(top_k_idx), p=top_k_probs)
                token_id = int(top_k_idx[sampled_idx])

            # Check EOS
            if token_id == self.eos_token and i >= min_len:
                hit_max_len = False
                break

            should_append = True
            if max_silent_tokens >= 0 and token_id in self.silent_tokens:
                silent_run += 1
                if silent_run > max_silent_tokens:
                    should_append = False
                    skipped_silent_tokens += 1
            else:
                silent_run = 0

            if should_append:
                out_tokens.append(token_id)

            if i % 50 == 0:
                print(f"    Generated {i+1} tokens... (kept {len(out_tokens)})")

            # Get next token embedding
            next_emb = self.get_speech_embedding(np.array([[token_id]], dtype=np.int64))

            # Update attention mask
            total_len = seq_len + i + 1
            attention_mask = np.ones((1, total_len), dtype=np.float32)

            # Decode step
            outputs = self.llm_backbone_decode.run(
                None, {
                    'inputs_embeds': next_emb.astype(np.float32),
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values
                }
            )
            hidden_states = outputs[0]
            past_key_values = outputs[1]

            # Get next logits
            logits = self.llm_decoder.run(
                None, {'hidden_state': hidden_states}
            )[0]

        if hit_max_len:
            print("    WARNING: reached max_len before EOS; lower max_len or adjust sampling if output is too long")
        if skipped_silent_tokens:
            print(f"    Skipped {skipped_silent_tokens} excessive silent/breath tokens")
        print(f"    Generated {len(out_tokens)} speech tokens")
        return np.array([out_tokens], dtype=np.int64)

    def flow_inference(
        self,
        speech_tokens: np.ndarray,
        embedding: np.ndarray,
        prompt_tokens: np.ndarray = None,
        prompt_mel: np.ndarray = None,
        n_timesteps: int = 10
    ) -> np.ndarray:
        """Convert speech tokens to mel using pure ONNX Flow

        Args:
            speech_tokens: Generated speech tokens [1, seq_len]
            embedding: Speaker embedding [1, 192]
            prompt_tokens: Prompt speech tokens [1, prompt_seq_len] (optional)
            prompt_mel: Prompt mel features [1, prompt_mel_len, 80] (optional)
            n_timesteps: Number of flow steps
        """

        # Normalize and project speaker embedding
        embedding_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        spks = self.flow_speaker_projection.run(
            None, {'embedding': embedding_norm.astype(np.float32)}
        )[0]

        # Concatenate prompt tokens and generated tokens
        if prompt_tokens is not None and prompt_tokens.shape[1] > 0:
            all_tokens = np.concatenate([prompt_tokens, speech_tokens], axis=1)
            prompt_token_len = prompt_tokens.shape[1]
        else:
            all_tokens = speech_tokens
            prompt_token_len = 0

        # Embed tokens
        token_embedded = self.flow_token_embedding.run(
            None, {'token': all_tokens.astype(np.int64)}
        )[0]

        # Pre-lookahead (already includes repeat_interleave with token_mel_ratio=2)
        h = self.flow_pre_lookahead.run(
            None, {'token_embedded': token_embedded.astype(np.float32)}
        )[0]

        # CosyVoice3 uses token_mel_ratio=2 (each token produces 2 mel frames)
        # Note: pre_lookahead ONNX model already applies repeat_interleave
        token_mel_ratio = 2

        # Calculate mel lengths
        mel_len = h.shape[1]  # h.shape[1] = token_seq_len * token_mel_ratio (from pre_lookahead)

        if prompt_tokens is not None and prompt_token_len > 0:
            mel_len1 = prompt_token_len * token_mel_ratio
            mel_len2 = mel_len - mel_len1
        else:
            mel_len1 = 0
            mel_len2 = mel_len

        # Build conditions
        conds = np.zeros((1, 80, mel_len), dtype=np.float32)
        if prompt_mel is not None and prompt_mel.shape[1] > 0 and mel_len1 > 0:
            # prompt_mel is [1, frames, 80], transpose to [1, 80, frames]
            prompt_mel_t = prompt_mel.transpose(0, 2, 1)

            # Resize prompt_mel to match expected mel_len1
            # This is necessary because the mel extraction rate may differ from token_mel_ratio
            from scipy.ndimage import zoom
            src_len = prompt_mel_t.shape[2]
            if src_len != mel_len1:
                # Resize along time axis
                zoom_factor = mel_len1 / src_len
                prompt_mel_resized = zoom(prompt_mel_t, (1, 1, zoom_factor), order=1)
                conds[:, :, :mel_len1] = prompt_mel_resized[:, :, :mel_len1]
            else:
                conds[:, :, :mel_len1] = prompt_mel_t[:, :, :mel_len1]

        # Prepare mu and mask
        mu = h.transpose(0, 2, 1)
        mask = np.ones((1, 1, mel_len), dtype=np.float32)

        # Initialize x with noise
        x = np.random.randn(1, 80, mel_len).astype(np.float32)

        # Batch for estimator (needs batch=2)
        x_batch = np.concatenate([x, x], axis=0)
        mask_batch = np.concatenate([mask, mask], axis=0)
        mu_batch = np.concatenate([mu, mu], axis=0)
        spks_batch = np.concatenate([spks, spks], axis=0)
        conds_batch = np.concatenate([conds, conds], axis=0)

        print(f"    Running {n_timesteps} flow steps (mel_len={mel_len}, prompt={mel_len1}, generated={mel_len2})...")

        # Euler solver
        for step in range(n_timesteps):
            t = np.array([step / n_timesteps, step / n_timesteps], dtype=np.float32)

            velocity = self.flow_estimator.run(None, {
                'x': x_batch,
                'mask': mask_batch,
                'mu': mu_batch,
                't': t,
                'spks': spks_batch,
                'cond': conds_batch
            })[0]

            dt = 1.0 / n_timesteps
            x_batch = x_batch + velocity * dt

        # Extract only the generated portion (exclude prompt)
        mel = x_batch[:1]
        if mel_len1 > 0:
            mel = mel[:, :, mel_len1:]  # Remove prompt portion
            print(f"    Extracted generated portion: mel shape {mel.shape}")
        return mel

    def _stft(self, x: np.ndarray, n_fft: int = 16, hop_len: int = 4, center: bool = True) -> tuple:
        """Compute STFT using numpy (PyTorch-compatible implementation)

        This implements the same STFT as PyTorch's torch.stft with:
        - n_fft=16, hop_length=4, win_length=16
        - Hann window
        - center=True (default): pads signal by n_fft//2 on each side
        """
        from scipy.signal import get_window

        # Create Hann window matching PyTorch HiFT
        window = get_window("hann", n_fft, fftbins=True).astype(np.float32)

        x = x.astype(np.float32)

        # Apply centering (same as PyTorch default)
        if center:
            pad_len = n_fft // 2
            x = np.pad(x, (pad_len, pad_len), mode='reflect')

        # Calculate number of frames
        n_frames = 1 + (len(x) - n_fft) // hop_len

        # Allocate output arrays
        n_freqs = n_fft // 2 + 1  # 9 for n_fft=16
        real = np.zeros((n_freqs, n_frames), dtype=np.float32)
        imag = np.zeros((n_freqs, n_frames), dtype=np.float32)

        # Compute STFT frame by frame
        for i in range(n_frames):
            start = i * hop_len
            frame = x[start:start + n_fft] * window

            # FFT
            spectrum = np.fft.rfft(frame)
            real[:, i] = np.real(spectrum)
            imag[:, i] = np.imag(spectrum)

        return real, imag

    def _istft(self, magnitude: np.ndarray, phase: np.ndarray, n_fft: int = 16, hop_len: int = 4) -> np.ndarray:
        """Compute ISTFT from magnitude and phase using scipy (PyTorch-free implementation)

        This implements the same ISTFT as PyTorch's torch.istft with:
        - n_fft=16, hop_length=4, win_length=16
        - Hann window
        - Proper overlap-add normalization
        """
        from scipy.signal import get_window

        # Create Hann window matching PyTorch HiFT
        window = get_window("hann", n_fft, fftbins=True).astype(np.float32)

        # Clip magnitude to prevent numerical issues (same as HiFT._istft)
        magnitude = np.clip(magnitude, a_min=None, a_max=100.0)

        # Reconstruct complex STFT from magnitude and phase
        complex_spec = magnitude * np.exp(1j * phase)

        n_frames = complex_spec.shape[1]

        # Calculate output length
        output_length = n_fft + (n_frames - 1) * hop_len

        # Allocate output arrays
        audio = np.zeros(output_length, dtype=np.float32)
        window_sum = np.zeros(output_length, dtype=np.float32)

        # Overlap-add reconstruction
        for i in range(n_frames):
            start = i * hop_len

            # Inverse FFT
            frame_spec = complex_spec[:, i]
            frame = np.fft.irfft(frame_spec, n=n_fft).astype(np.float32)

            # Apply window and add to output
            audio[start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window ** 2

        # Normalize by window sum (avoid division by zero)
        window_sum = np.maximum(window_sum, 1e-8)
        audio = audio / window_sum

        return audio.astype(np.float32)

    def hift_inference(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel to audio using pure ONNX HiFT

        HiFT CosyVoice3 uses:
        - mel hop_length = 480 (mel spectrogram)
        - n_fft = 16, hop_length = 4 for internal STFT
        - upsample_rates = [8, 5, 3] (total 120x)
        - Expected STFT frames = mel_frames * 120
        """

        # Predict F0
        f0 = self.hift_f0_predictor.run(None, {'mel': mel.astype(np.float32)})[0]
        print(f"    F0 shape: {f0.shape}, range: [{f0.min():.1f}, {f0.max():.1f}]")

        # Generate source from F0
        # F0 needs to be reshaped to [batch, 1, time]
        f0_input = f0[:, np.newaxis, :]  # [batch, 1, time]
        source = self.hift_source_generator.run(None, {'f0': f0_input.astype(np.float32)})[0]
        print(f"    Source shape: {source.shape}")

        # Compute STFT of source signal
        # HiFT CosyVoice3 uses n_fft=16, hop_len=4, center=True
        # With centering, expected STFT frames = mel_frames * 120 + 1
        source_squeezed = source.squeeze()  # [time]

        try:
            stft_real, stft_imag = self._stft(source_squeezed, n_fft=16, hop_len=4, center=True)
            # Combine real and imaginary: [batch, 18, time] (9 real + 9 imaginary)
            source_stft = np.concatenate([stft_real, stft_imag], axis=0)
            source_stft = source_stft[np.newaxis, :, :]  # Add batch dim
            print(f"    Source STFT shape: {source_stft.shape} (expected: mel_frames*120 = {mel.shape[2]*120})")

            # Decode mel + source_stft to magnitude and phase
            outputs = self.hift_decoder.run(None, {
                'mel': mel.astype(np.float32),
                'source_stft': source_stft.astype(np.float32)
            })
            magnitude = outputs[0]  # [batch, 9, time]
            phase = outputs[1]      # [batch, 9, time]
            print(f"    Magnitude shape: {magnitude.shape}, Phase shape: {phase.shape}")

            # Apply ISTFT to convert magnitude and phase to audio
            # Use same n_fft=16, hop_len=4 as STFT
            audio = self._istft(magnitude.squeeze(0), phase.squeeze(0), n_fft=16, hop_len=4)

            # Apply audio clipping (same as HiFT: audio_limit=0.99)
            audio = np.clip(audio, -0.99, 0.99)

            print(f"    Audio shape: {audio.shape}")
            return audio
        except Exception as e:
            print(f"    HiFT decoder failed: {e}")
            print("    Using Griffin-Lim fallback...")
            return self._griffin_lim(mel)

    def _griffin_lim(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """Simple Griffin-Lim vocoder as fallback"""
        try:
            import librosa
            mel_db = mel.squeeze()
            # Assume mel is in linear scale, convert to power
            mel_power = np.maximum(mel_db, 1e-10)
            audio = librosa.feature.inverse.mel_to_audio(
                mel_power,
                sr=self.sample_rate,
                hop_length=256,
                n_fft=1024,
                n_iter=n_iter
            )
            return audio
        except Exception as e:
            print(f"    Griffin-Lim failed: {e}")
            # Return noise as last resort
            audio_len = mel.shape[2] * 256
            return np.random.randn(audio_len).astype(np.float32) * 0.01

    def inference(
        self,
        text: str,
        prompt_wav: str,
        prompt_text: str,
        sampling_k: int = 25,
        temperature: float = 1.0,
        max_len: int = 500,
        min_len: int = 10,
        flow_steps: int = 10,
        max_len_ratio: float = 20.0,
        max_silent_tokens: int = 5,
        speed: float = 1.0,
        trim_silence: bool = False,
        trim_threshold: float = 0.006,
        trim_pad_ms: int = 120
    ) -> np.ndarray:
        """Full TTS inference using pure ONNX (zero-shot mode)

        Args:
            text: Text to synthesize (with language tag, e.g., "<|en|>Hello")
            prompt_wav: Path to prompt audio file for voice cloning (REQUIRED)
            prompt_text: Transcript of the prompt audio (REQUIRED for zero-shot mode)
            sampling_k: Top-k sampling parameter. Use 1 for greedy decoding.
            temperature: Sampling temperature. Lower values reduce randomness.
            max_len: Maximum generated speech token length before dynamic clipping.
            min_len: Minimum generated speech token length before EOS can stop generation.
            flow_steps: Number of flow matching steps. Higher values can improve naturalness.
            max_len_ratio: Maximum generated speech tokens per TTS text token.
            max_silent_tokens: Maximum consecutive silent/breath tokens to keep.
            speed: Speech speed multiplier. Values above 1.0 make output faster.
            trim_silence: Whether to trim leading/trailing silence from output.
            trim_threshold: RMS threshold used by silence trimming.
            trim_pad_ms: Padding kept around detected speech when trimming.

        Returns:
            Audio waveform as numpy array

        Raises:
            ValueError: If prompt_wav or prompt_text is not provided
        """
        # Validate inputs (required for zero-shot mode)
        if not prompt_wav:
            raise ValueError(
                "prompt_wav is required for CosyVoice inference. "
                "CosyVoice is a voice cloning TTS system that requires a reference audio. "
                "Please provide a prompt audio file (3-10 seconds recommended)."
            )
        if not os.path.exists(prompt_wav):
            raise ValueError(f"Prompt audio file not found: {prompt_wav}")
        if not prompt_text:
            raise ValueError(
                "prompt_text is required for zero-shot mode. "
                "Please provide the transcript of the prompt audio."
            )
        if sampling_k < 1:
            raise ValueError("sampling_k must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        if min_len < 0:
            raise ValueError("min_len must be >= 0")
        if flow_steps < 1:
            raise ValueError("flow_steps must be >= 1")
        if max_len_ratio <= 0:
            raise ValueError("max_len_ratio must be > 0")
        if max_silent_tokens < 0:
            raise ValueError("max_silent_tokens must be >= 0")
        if speed <= 0:
            raise ValueError("speed must be > 0")
        if trim_threshold <= 0:
            raise ValueError("trim_threshold must be > 0")
        if trim_pad_ms < 0:
            raise ValueError("trim_pad_ms must be >= 0")

        print(f"\nInput text: {text}")
        print(f"Prompt audio: {prompt_wav}")
        print(f"Prompt text: {prompt_text}")
        print(
            f"Sampling: top_k={sampling_k}, temperature={temperature}, "
            f"min_len={min_len}, max_len={max_len}, max_len_ratio={max_len_ratio}"
        )
        print(f"Flow steps: {flow_steps}")
        print(f"Speed: {speed}")
        print(f"Trim silence: {trim_silence}")
        start_time = time.time()

        # Process prompt audio (required)
        print("\n0. Processing prompt audio...")
        prompt_start = time.time()

        # Extract speaker embedding
        embedding = self.extract_speaker_embedding(prompt_wav)
        print(f"   Speaker embedding: {embedding.shape}")

        # Extract speech tokens for LLM and flow
        prompt_tokens = self.extract_speech_tokens(prompt_wav)
        print(f"   Prompt speech tokens: {prompt_tokens.shape} ({prompt_tokens.shape[1]} tokens)")

        # Extract mel for flow conditioning
        prompt_mel = self.extract_speech_mel(prompt_wav)
        print(f"   Prompt mel: {prompt_mel.shape} ({prompt_mel.shape[1]} frames)")

        prompt_time = time.time() - prompt_start
        print(f"   Prompt processing time: {prompt_time:.2f}s")

        # Step 1: LLM inference (zero-shot mode with prompt_text and prompt_speech_tokens)
        print("\n1. Running LLM (ONNX) - Zero-shot mode...")
        llm_start = time.time()
        speech_tokens = self.llm_inference(
            text,
            prompt_text=prompt_text,
            prompt_speech_tokens=prompt_tokens,
            sampling_k=sampling_k,
            temperature=temperature,
            max_len=max_len,
            min_len=min_len,
            max_len_ratio=max_len_ratio,
            max_silent_tokens=max_silent_tokens
        )
        llm_time = time.time() - llm_start
        print(f"   LLM time: {llm_time:.2f}s")

        # Step 2: Flow inference
        print("\n2. Running Flow (ONNX)...")
        flow_start = time.time()
        mel = self.flow_inference(
            speech_tokens, embedding,
            prompt_tokens=prompt_tokens,
            prompt_mel=prompt_mel,
            n_timesteps=flow_steps
        )
        flow_time = time.time() - flow_start
        print(f"   Mel shape: {mel.shape}")
        print(f"   Flow time: {flow_time:.2f}s")

        if speed != 1.0:
            original_frames = mel.shape[2]
            mel = self.apply_speed_to_mel(mel, speed)
            print(f"   Applied speed {speed}: mel frames {original_frames} -> {mel.shape[2]}")

        # Step 3: HiFT vocoder
        print("\n3. Running HiFT (ONNX)...")
        hift_start = time.time()
        audio = self.hift_inference(mel)
        hift_time = time.time() - hift_start
        print(f"   HiFT time: {hift_time:.2f}s")

        if trim_silence:
            original_samples = len(audio)
            audio = self.trim_silence(audio, self.sample_rate, trim_threshold, trim_pad_ms)
            trimmed_samples = original_samples - len(audio)
            print(f"   Trimmed silence: {trimmed_samples / self.sample_rate:.2f}s removed")

        total_time = time.time() - start_time
        audio_len = len(audio) / self.sample_rate

        print(f"\n" + "=" * 50)
        print(f"Total time: {total_time:.2f}s")
        print(f"Audio length: {audio_len:.2f}s")
        print(f"RTF: {total_time / audio_len:.2f}")
        print("=" * 50)

        return audio

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def _log_softmax(x):
        return x - np.log(np.sum(np.exp(x - np.max(x)))) - np.max(x)

    @staticmethod
    def apply_speed_to_mel(mel: np.ndarray, speed: float) -> np.ndarray:
        """Adjust speech speed by resizing the mel time axis before HiFT."""
        if speed == 1.0:
            return mel
        from scipy.ndimage import zoom

        target_frames = max(1, int(round(mel.shape[2] / speed)))
        zoom_factor = target_frames / mel.shape[2]
        return zoom(mel, (1, 1, zoom_factor), order=1).astype(np.float32)

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = 0.006,
        pad_ms: int = 120
    ) -> np.ndarray:
        """Trim leading/trailing low-energy audio while keeping a small natural pad."""
        if audio.size == 0:
            return audio

        frame = max(1, int(sample_rate * 0.02))
        hop = max(1, int(sample_rate * 0.01))
        if audio.size < frame:
            return audio

        rms = []
        for start in range(0, audio.size - frame + 1, hop):
            segment = audio[start:start + frame]
            rms.append(float(np.sqrt(np.mean(segment * segment) + 1e-12)))

        voiced = np.where(np.array(rms) > threshold)[0]
        if voiced.size == 0:
            return audio

        pad = int(sample_rate * pad_ms / 1000)
        start_sample = max(0, int(voiced[0]) * hop - pad)
        end_sample = min(audio.size, int(voiced[-1]) * hop + frame + pad)
        return audio[start_sample:end_sample]


DEFAULT_INFERENCE_OPTIONS = {
    'sampling_k': 25,
    'temperature': 1.0,
    'max_len': 500,
    'min_len': 10,
    'flow_steps': 10,
    'max_len_ratio': 20.0,
    'max_silent_tokens': 5,
    'speed': 1.0,
    'trim_silence': False,
    'trim_threshold': 0.006,
    'trim_pad_ms': 120,
    'seed': None,
    'prompt_prefix': '',
}


def load_inference_options(args):
    """Load inference options from defaults, optional JSON config, then CLI overrides."""
    options = DEFAULT_INFERENCE_OPTIONS.copy()
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a JSON object")
        unknown_keys = sorted(set(config) - set(options))
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")
        options.update(config)

    for key in options:
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    return options


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        description='Pure ONNX CosyVoice3 TTS Inference (Zero-Shot Voice Cloning)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/onnx_inference_pure.py \\
      --text "Hello, this is a test." \\
      --prompt_wav asset/prompts/en_female_nova_greeting.wav \\
      --prompt_text "Hello, my name is Sarah."

  python scripts/onnx_inference_pure.py \\
      --text "こんにちは、今日はいい天気ですね。" \\
      --prompt_wav my_voice.wav \\
      --prompt_text "Your prompt audio transcript here" \\
      --output output.wav

Note:
  CosyVoice3 uses zero-shot voice cloning. Both prompt audio AND prompt text (transcript)
  are REQUIRED for proper inference. The prompt text should match what is spoken in the
  prompt audio file.

  Do NOT include language tags like <|en|> or <|ja|> - they will be pronounced as text.
  CosyVoice3 automatically detects the language.
        """
    )
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='Path to model directory')
    parser.add_argument('--text', type=str, default='Hello, this is a test of pure ONNX inference.',
                        help='Text to synthesize (do NOT include language tags)')
    parser.add_argument('--prompt_wav', type=str, required=True,
                        help='Path to prompt audio for voice cloning (REQUIRED, 3-10 seconds recommended)')
    parser.add_argument('--prompt_text', type=str, required=True,
                        help='Transcript of the prompt audio (REQUIRED for zero-shot mode)')
    parser.add_argument('--output', type=str, default='output_onnx_pure.wav',
                        help='Output audio file path')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision instead of FP16')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional JSON config for inference options')
    parser.add_argument('--sampling_k', type=int, default=None,
                        help='Top-k sampling. Use 1 for greedy decoding; lower values reduce hallucination risk')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature. Try 0.6-0.9 for more stable output')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Maximum generated speech token length before dynamic clipping')
    parser.add_argument('--min_len', type=int, default=None,
                        help='Minimum generated speech token length before EOS can stop generation')
    parser.add_argument('--flow_steps', type=int, default=None,
                        help='Flow matching steps. Try 20 or 30 for less synthetic sound; slower on CPU')
    parser.add_argument('--max_len_ratio', type=float, default=None,
                        help='Maximum generated speech tokens per TTS text token')
    parser.add_argument('--max_silent_tokens', type=int, default=None,
                        help='Maximum consecutive silent/breath tokens to keep')
    parser.add_argument('--speed', type=float, default=None,
                        help='Speech speed multiplier. Values above 1.0 make output faster')
    parser.add_argument('--trim_silence', action=argparse.BooleanOptionalAction, default=None,
                        help='Trim leading/trailing low-energy audio from the output')
    parser.add_argument('--trim_threshold', type=float, default=None,
                        help='RMS threshold for output silence trimming')
    parser.add_argument('--trim_pad_ms', type=int, default=None,
                        help='Padding kept around detected speech when trimming')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible sampling')
    parser.add_argument('--prompt_prefix', type=str, default=None,
                        help='Optional prefix prepended to prompt_text when it is not already present')

    args = parser.parse_args()
    options = load_inference_options(args)
    prompt_text = args.prompt_text
    prompt_prefix = options['prompt_prefix']
    if prompt_prefix and not prompt_text.startswith(prompt_prefix):
        prompt_text = prompt_prefix + prompt_text

    if options['seed'] is not None:
        np.random.seed(options['seed'])

    print("=" * 60)
    print("Pure ONNX CosyVoice3 Inference (Zero-Shot Mode)")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Text: {args.text}")
    print(f"Prompt audio: {args.prompt_wav}")
    print(f"Prompt text: {prompt_text}")
    print(f"Use FP16: {not args.fp32}")
    print(
        "Options: "
        f"sampling_k={options['sampling_k']}, "
        f"temperature={options['temperature']}, "
        f"min_len={options['min_len']}, "
        f"max_len={options['max_len']}, "
        f"max_len_ratio={options['max_len_ratio']}, "
        f"flow_steps={options['flow_steps']}, "
        f"max_silent_tokens={options['max_silent_tokens']}, "
        f"speed={options['speed']}, "
        f"trim_silence={options['trim_silence']}, "
        f"seed={options['seed']}"
    )
    print()

    # Create engine
    engine = PureOnnxCosyVoice3(args.model_dir, use_fp16=not args.fp32)

    # Run inference (zero-shot mode)
    audio = engine.inference(
        args.text,
        prompt_wav=args.prompt_wav,
        prompt_text=prompt_text,
        sampling_k=options['sampling_k'],
        temperature=options['temperature'],
        max_len=options['max_len'],
        min_len=options['min_len'],
        flow_steps=options['flow_steps'],
        max_len_ratio=options['max_len_ratio'],
        max_silent_tokens=options['max_silent_tokens'],
        speed=options['speed'],
        trim_silence=options['trim_silence'],
        trim_threshold=options['trim_threshold'],
        trim_pad_ms=options['trim_pad_ms']
    )

    # Save output
    sf.write(args.output, audio, engine.sample_rate)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()

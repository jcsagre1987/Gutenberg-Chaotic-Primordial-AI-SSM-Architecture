#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gutenberg Chaotic Primordial v0.3.0 — Pretraining, Expansion, Distillation & Inference Engine.

Author: Juan Carlo Sagre
License: Apache-2.0

Copyright 2026 Juan Carlo Sagre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE_NAME = "Gutenberg Chaotic Primordial"
ARCHITECTURE_VERSION = "0.3.0"
COMPATIBLE_VERSION_PREFIX = "0.3."
CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINT_NAME_PATTERN = re.compile(r"^Test(\d+)$")

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DELIM_TOKEN_ID = 3
SPECIAL_TOKENS = 4
BYTE_VOCAB_SIZE = SPECIAL_TOKENS + 256  # 260

_CHUNK_SCAN_BUFFER_CACHE: dict[
    tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_chunk_scan_buffers(
    c_len: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache static triangular masks and identity matrices across steps and layers."""
    key = (c_len, str(device), dtype)
    cached = _CHUNK_SCAN_BUFFER_CACHE.get(key)
    if cached is not None:
        return cached
    strict_tril_bool = torch.tril(
        torch.ones((c_len, c_len), device=device, dtype=torch.bool),
        diagonal=-1,
    )
    strict_tril_float = strict_tril_bool.to(dtype=dtype)
    eye = torch.eye(c_len, device=device, dtype=dtype).view(1, 1, c_len, c_len)
    buffers = (strict_tril_bool, strict_tril_float, eye)
    _CHUNK_SCAN_BUFFER_CACHE[key] = buffers
    return buffers


@dataclass(frozen=True)
class ModelConfig:
    """Serializable model configuration for pretraining and inference."""

    vocab_size: int = BYTE_VOCAB_SIZE
    d_model: int = 256
    num_layers: int = 4
    state_size: int = 16
    memory_heads: int = 8
    conv_kernel_size: int = 3
    ssm_chunk_size: int = 64
    ffn_expansion: float = 2.0
    dropout: float = 0.05
    use_associative_memory: bool = True

    def validate(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")
        if self.d_model < 8:
            raise ValueError("d_model must be at least 8")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.state_size < 1:
            raise ValueError("state_size must be at least 1")
        if self.memory_heads < 1:
            raise ValueError("memory_heads must be at least 1")
        if self.d_model % self.memory_heads != 0:
            raise ValueError("d_model must be divisible by memory_heads")
        if self.conv_kernel_size < 1:
            raise ValueError("conv_kernel_size must be at least 1")
        if self.ssm_chunk_size < 1:
            raise ValueError("ssm_chunk_size must be at least 1")
        if self.ffn_expansion <= 0.0:
            raise ValueError("ffn_expansion must be strictly positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config


class ByteLevelTokenizer:
    """Lossless UTF-8 byte-level tokenizer with zero out-of-vocabulary tokens."""

    def __init__(self) -> None:
        self.pad_id = PAD_TOKEN_ID
        self.bos_id = BOS_TOKEN_ID
        self.eos_id = EOS_TOKEN_ID
        self.delim_id = DELIM_TOKEN_ID
        self.vocab_size = BYTE_VOCAB_SIZE

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        raw = text.encode("utf-8", errors="replace")
        tokens = [b + SPECIAL_TOKENS for b in raw]
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        byte_vals: list[int] = []
        for tid in token_ids:
            if tid < SPECIAL_TOKENS:
                if not skip_special and tid == self.delim_id:
                    byte_vals.append(ord("\n"))
                continue
            if tid < self.vocab_size:
                byte_vals.append(tid - SPECIAL_TOKENS)
        return bytes(byte_vals).decode("utf-8", errors="replace")


def _extract_record_text(record: Any) -> str:
    """Extract normalized training text from arbitrary dataset record schemas."""
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return ""

    if "messages" in record and isinstance(record["messages"], list):
        parts: list[str] = []
        for msg in record["messages"]:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user")).strip()
                content = str(msg.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}")
        return "\n".join(parts).strip()

    if "instruction" in record and ("output" in record or "response" in record):
        inst = str(record.get("instruction", "")).strip()
        inp = str(record.get("input", "")).strip()
        out = str(record.get("output", record.get("response", ""))).strip()
        if inst and out:
            if inp:
                return f"Instruction: {inst}\nInput: {inp}\nResponse: {out}".strip()
            return f"Instruction: {inst}\nResponse: {out}".strip()

    if "word" in record and ("definition" in record or "meaning" in record):
        w = str(record.get("word", "")).strip()
        defn = str(record.get("definition", record.get("meaning", ""))).strip()
        ex = str(record.get("example", "")).strip()
        if w and defn:
            base = f"Word: {w} | Meaning: {defn}"
            return f"{base} | Example: {ex}" if ex else base

    for key in ("text", "story", "content", "document", "body", "passage"):
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key].strip()

    if "prompt" in record and ("response" in record or "completion" in record):
        prompt = str(record.get("prompt", "")).strip()
        reply = str(record.get("response", record.get("completion", ""))).strip()
        if prompt or reply:
            return f"User: {prompt}\nAssistant: {reply}".strip()

    if "input" in record and "output" in record:
        inp = str(record.get("input", "")).strip()
        out = str(record.get("output", "")).strip()
        if inp or out:
            return f"Input: {inp}\nOutput: {out}".strip()

    if "question" in record and "answer" in record:
        q = str(record.get("question", "")).strip()
        a = str(record.get("answer", "")).strip()
        if q or a:
            return f"Question: {q}\nAnswer: {a}".strip()

    return ""


class PackedSequenceDataset:
    """Packs tokenized documents into contiguous [N, seq_len + 1] blocks."""

    def __init__(self, token_stream: torch.Tensor, seq_len: int) -> None:
        if seq_len < 4:
            raise ValueError("seq_len must be at least 4")
        block_size = seq_len + 1
        num_blocks = token_stream.numel() // block_size
        if num_blocks < 1:
            raise ValueError(
                f"Insufficient tokens ({token_stream.numel()}) for seq_len={seq_len}. "
                f"Provide more text or reduce --seq-len."
            )
        trimmed = token_stream[: num_blocks * block_size]
        self.blocks = trimmed.view(num_blocks, block_size).contiguous()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def sample_batch(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = len(self)
        if batch_size <= total:
            indices = torch.randperm(total, generator=generator)[:batch_size]
        else:
            indices = torch.randint(0, total, (batch_size,), generator=generator)
        batch = self.blocks[indices].to(device=device, dtype=torch.long)
        return batch[:, :-1], batch[:, 1:]


def _pack_isolated_documents(
    docs: list[list[int]], seq_len: int, pad_gap: int = 4
) -> PackedSequenceDataset:
    """Separate documents with PAD_TOKEN_ID gaps to eliminate cross-document state bleed."""
    stream: list[int] = []
    gap = [PAD_TOKEN_ID] * pad_gap
    for doc in docs:
        stream.extend(doc)
        stream.extend(gap)
    flat = torch.tensor(stream, dtype=torch.int32)
    return PackedSequenceDataset(flat, seq_len=seq_len)


def _iter_records_from_file(file_path: Path) -> Iterator[Any]:
    """Yield text records from .jsonl, .json, .txt, .md, .parquet, or HF .arrow files."""
    suffix = file_path.suffix.lower()

    if suffix in (".jsonl", ".ndjson"):
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    continue
        return

    if suffix == ".json":
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return
        if isinstance(payload, list):
            for item in payload:
                yield item
        elif isinstance(payload, dict):
            for key in ("data", "rows", "records", "examples", "train"):
                if isinstance(payload.get(key), list):
                    for item in payload[key]:
                        yield item
                    return
            yield payload
        return

    if suffix in (".txt", ".md"):
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = raw.split("\n\n") if "\n\n" in raw else raw.splitlines()
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned:
                yield cleaned
        return

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(file_path)
            for row in table.to_pylist():
                yield row
        except ImportError:
            pass
        return

    if suffix == ".arrow":
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.ipc as pa_ipc  # type: ignore

            with pa.memory_map(str(file_path), "r") as source:
                try:
                    reader = pa_ipc.open_stream(source)
                    for batch in reader:
                        for row in batch.to_pylist():
                            yield row
                except Exception:
                    source.seek(0)
                    reader = pa_ipc.open_file(source)
                    for i in range(reader.num_record_batches):
                        for row in reader.get_batch(i).to_pylist():
                            yield row
            return
        except ImportError:
            pass
        try:
            from datasets import Dataset  # type: ignore

            ds = Dataset.from_file(str(file_path))
            for row in ds:
                yield row
        except ImportError:
            pass


def load_jsonl_datasets(
    data_path: Path,
    tokenizer: ByteLevelTokenizer,
    seq_len: int,
    val_ratio: float = 0.05,
    seed: int = 1701,
    max_docs: int = 25000,
    math_replay_ratio: float = 0.20,
) -> tuple[PackedSequenceDataset, PackedSequenceDataset, dict[str, Any]]:
    """Load .jsonl/.json/.txt/.parquet/.arrow data via reservoir sampling + math replay."""
    resolved = data_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")

    supported_ext = {".jsonl", ".ndjson", ".json", ".txt", ".md", ".parquet", ".arrow"}
    if resolved.is_file():
        files = [resolved]
    else:
        files = sorted(
            p
            for p in resolved.rglob("*")
            if p.is_file()
            and p.suffix.lower() in supported_ext
            and p.name.lower() not in {"dataset_info.json", "state.json"}
        )
    if not files:
        found_sample = (
            [p.name for p in list(resolved.rglob("*"))[:10] if p.is_file()]
            if resolved.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"No supported dataset files ({sorted(supported_ext)}) found under: {resolved}. "
            f"Sample files seen: {found_sample}"
        )

    rng = random.Random(seed)
    sampled_texts: list[str] = []
    seen_valid = 0
    skipped_lines = 0

    for file_path in files:
        for obj in _iter_records_from_file(file_path):
            text = _extract_record_text(obj)
            if len(text) < 4:
                skipped_lines += 1
                continue
            seen_valid += 1
            if max_docs <= 0 or len(sampled_texts) < max_docs:
                sampled_texts.append(text)
            else:
                slot = rng.randint(0, seen_valid - 1)
                if slot < max_docs:
                    sampled_texts[slot] = text

    if len(sampled_texts) < 2:
        raise ValueError(
            f"Found only {len(sampled_texts)} valid records in {resolved} "
            f"(files scanned: {[f.name for f in files[:5]]}). "
            f"If these are .arrow/.parquet files, ensure 'pyarrow' or 'datasets' is installed."
        )

    rng.shuffle(sampled_texts)
    split_idx = max(1, int(len(sampled_texts) * (1.0 - val_ratio)))
    if split_idx >= len(sampled_texts):
        split_idx = len(sampled_texts) - 1

    lang_train_texts = sampled_texts[:split_idx]
    lang_val_texts = sampled_texts[split_idx:]

    replay_train_texts: list[str] = []
    replay_val_texts: list[str] = []
    if math_replay_ratio > 0.0:
        math_tr, math_va = _builtin_stage2_math_splits(seed=seed + 77)
        target_tr = int(
            len(lang_train_texts)
            * math_replay_ratio
            / max(1e-6, 1.0 - math_replay_ratio)
        )
        target_va = max(
            16,
            int(
                len(lang_val_texts)
                * math_replay_ratio
                / max(1e-6, 1.0 - math_replay_ratio)
            ),
        )
        replay_train_texts = [
            rng.choice(math_tr) for _ in range(min(target_tr, len(math_tr) * 2))
        ]
        replay_val_texts = [
            rng.choice(math_va) for _ in range(min(target_va, len(math_va) * 2))
        ]

    combined_train_texts = lang_train_texts + replay_train_texts
    combined_val_texts = lang_val_texts + replay_val_texts
    rng.shuffle(combined_train_texts)
    rng.shuffle(combined_val_texts)

    train_docs = [
        tokenizer.encode(t, add_bos=True, add_eos=True) for t in combined_train_texts
    ]
    val_docs = [
        tokenizer.encode(t, add_bos=True, add_eos=True) for t in combined_val_texts
    ]

    train_ds = _pack_isolated_documents(train_docs, seq_len=seq_len)
    val_ds = _pack_isolated_documents(val_docs, seq_len=seq_len)

    stats = {
        "files_loaded": len(files),
        "total_valid_records_scanned": seen_valid,
        "language_train_documents": len(lang_train_texts),
        "language_val_documents": len(lang_val_texts),
        "math_replay_train_documents": len(replay_train_texts),
        "math_replay_val_documents": len(replay_val_texts),
        "math_replay_ratio": math_replay_ratio,
        "skipped_records": skipped_lines,
        "train_blocks": len(train_ds),
        "val_blocks": len(val_ds),
        "seq_len": seq_len,
    }
    return train_ds, val_ds, stats


def _init_multi_timescale_log_decay(
    d_model: int,
    state_size: int,
    min_rate: float = 0.01,
    max_rate: float = 4.0,
) -> torch.Tensor:
    if state_size == 1:
        rates = torch.tensor([math.sqrt(min_rate * max_rate)], dtype=torch.float32)
    else:
        rates = torch.logspace(
            math.log10(min_rate),
            math.log10(max_rate),
            steps=state_size,
            dtype=torch.float32,
        )
    inv_softplus = torch.log(torch.expm1(rates))
    return inv_softplus.unsqueeze(0).expand(d_model, state_size).clone()


class CausalDepthwiseConv1d(nn.Module):
    """Strictly causal depthwise 1D convolution with exact chunk-state continuation."""

    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1
        self.weight = nn.Parameter(torch.empty(d_model, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        inputs: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, width = inputs.shape
        if width != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, received {width}")

        x_t = inputs.transpose(1, 2)
        if self.pad_len > 0:
            expected_state = (batch_size, self.d_model, self.pad_len)
            if conv_state is None:
                prefix = x_t.new_zeros(expected_state)
            else:
                if tuple(conv_state.shape) != expected_state:
                    raise ValueError(
                        f"conv_state must have shape {expected_state}, "
                        f"received {tuple(conv_state.shape)}"
                    )
                if conv_state.device != inputs.device:
                    raise ValueError("conv_state and inputs must be on the same device")
                prefix = conv_state.to(dtype=x_t.dtype)
            padded = torch.cat((prefix, x_t), dim=-1)
            next_state = padded[:, :, -self.pad_len :]
        else:
            padded = x_t
            next_state = x_t.new_zeros(batch_size, self.d_model, 0)

        out = F.conv1d(
            padded,
            self.weight.to(dtype=x_t.dtype),
            self.bias.to(dtype=x_t.dtype),
            padding=0,
            groups=self.d_model,
        ).transpose(1, 2)

        if return_state:
            return out, next_state
        return out


class SwiGLUFFN(nn.Module):
    """Pre-normalized SwiGLU channel-mixing residual block."""

    def __init__(
        self, d_model: int, expansion: float = 2.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden_dim = max(8, int(round(d_model * expansion)))
        self.norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.norm(hidden)
        gated = F.silu(self.gate_proj(normed)) * self.up_proj(normed)
        return hidden + self.dropout(self.down_proj(gated))


class SelectiveDiagonalSSM(nn.Module):
    """Chunkwise-parallel input-selective diagonal SSM with exact ZOH discretization."""

    def __init__(
        self, d_model: int, state_size: int, chunk_size: int = 64
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.log_decay = nn.Parameter(
            _init_multi_timescale_log_decay(d_model, state_size)
        )
        self.delta_projection = nn.Linear(d_model, d_model)
        self.input_projection = nn.Linear(d_model, state_size)
        self.read_projection = nn.Linear(d_model, state_size)
        self.output_scale = nn.Parameter(torch.ones(d_model))

    def _scan_chunk(
        self,
        scaled_decay: torch.Tensor,
        driven_inputs: torch.Tensor,
        readouts: torch.Tensor,
        state: torch.Tensor,
        inv_sqrt_state: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact O(log C) associative parallel prefix scan on 4D [B, C, D, S] tensors."""
        chunk_len = scaled_decay.shape[1]
        a_cum = torch.exp(-scaled_decay)
        if chunk_len == 1:
            next_state = a_cum[:, 0] * state + driven_inputs[:, 0]
            out = (
                (next_state * readouts[:, 0].unsqueeze(1)).sum(dim=-1, keepdim=True)
                * inv_sqrt_state
            ).transpose(1, 2)
            return out, next_state

        u_cum = driven_inputs
        stride = 1
        while stride < chunk_len:
            u_shifted = a_cum[:, stride:] * u_cum[:, :-stride] + u_cum[:, stride:]
            a_shifted = a_cum[:, stride:] * a_cum[:, :-stride]
            u_cum = torch.cat((u_cum[:, :stride], u_shifted), dim=1)
            a_cum = torch.cat((a_cum[:, :stride], a_shifted), dim=1)
            stride <<= 1

        state_chunk = a_cum * state.unsqueeze(1) + u_cum
        chunk_out = (
            (state_chunk * readouts.unsqueeze(2)).sum(dim=-1) * inv_sqrt_state
        )
        return chunk_out, state_chunk[:, -1]

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("SSM inputs must have shape [batch, sequence, d_model]")
        batch_size, sequence_length, width = inputs.shape
        if width != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, received {width}")
        if sequence_length < 1:
            raise ValueError("SSM sequence length must be at least 1")

        expected_state_shape = (batch_size, self.d_model, self.state_size)
        if initial_state is None:
            state = inputs.new_zeros(expected_state_shape)
        else:
            if tuple(initial_state.shape) != expected_state_shape:
                raise ValueError(
                    f"initial_state must have shape {expected_state_shape}, "
                    f"received {tuple(initial_state.shape)}"
                )
            if initial_state.device != inputs.device:
                raise ValueError("initial_state and inputs must be on the same device")
            state = initial_state.to(dtype=inputs.dtype)

        decay_rates = F.softplus(self.log_decay).to(dtype=inputs.dtype) + 1e-4
        deltas = F.softplus(self.delta_projection(inputs)) + 1e-4
        input_drives = torch.tanh(self.input_projection(inputs))
        readouts = torch.tanh(self.read_projection(inputs))

        decay_view = decay_rates.view(1, 1, self.d_model, self.state_size)
        scaled_decay = deltas.unsqueeze(-1) * decay_view
        forcing = inputs.unsqueeze(-1) * input_drives.unsqueeze(2)
        driven_inputs = (-torch.expm1(-scaled_decay) / decay_view) * forcing

        inv_sqrt_state = 1.0 / math.sqrt(self.state_size)
        chunk_outputs: list[torch.Tensor] = []

        for start in range(0, sequence_length, self.chunk_size):
            end = min(start + self.chunk_size, sequence_length)
            c_out, state = self._scan_chunk(
                scaled_decay=scaled_decay[:, start:end],
                driven_inputs=driven_inputs[:, start:end],
                readouts=readouts[:, start:end],
                state=state,
                inv_sqrt_state=inv_sqrt_state,
            )
            chunk_outputs.append(c_out)

        result = torch.cat(chunk_outputs, dim=1) * self.output_scale.to(dtype=inputs.dtype)
        if return_state:
            return result, state
        return result


class SelectiveSSMBlock(nn.Module):
    """Pre-norm Causal Conv1d + Chunkwise Selective SSM + SwiGLU block."""

    def __init__(
        self,
        d_model: int,
        state_size: int,
        conv_kernel_size: int,
        ssm_chunk_size: int,
        ffn_expansion: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = CausalDepthwiseConv1d(d_model, kernel_size=conv_kernel_size)
        self.ssm = SelectiveDiagonalSSM(
            d_model, state_size, chunk_size=ssm_chunk_size
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = SwiGLUFFN(d_model, expansion=ffn_expansion, dropout=dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        initial_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ssm_state = initial_state[0] if initial_state is not None else None
        conv_state = initial_state[1] if initial_state is not None else None

        normed = self.norm(hidden)
        conv_out, next_conv_state = self.conv(
            normed, conv_state=conv_state, return_state=True
        )
        ssm_out, next_ssm_state = self.ssm(
            F.silu(conv_out), initial_state=ssm_state, return_state=True
        )
        hidden = hidden + self.dropout(ssm_out)
        hidden = self.ffn(hidden)

        if return_state:
            return hidden, (next_ssm_state, next_conv_state)
        return hidden


def _gated_delta_chunkwise_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    log_gamma: torch.Tensor,
    memory_state: torch.Tensor,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact chunkwise-parallel gated delta-rule scan with cached triangular buffers."""
    batch_size, num_heads, seq_len, head_dim = query.shape
    compute_dtype = (
        torch.float32
        if query.dtype in (torch.float16, torch.bfloat16)
        else query.dtype
    )
    reads = torch.empty(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    state = memory_state.to(dtype=compute_dtype)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        c_len = end - start

        q_c = query[:, :, start:end, :].to(dtype=compute_dtype)
        k_c = key[:, :, start:end, :].to(dtype=compute_dtype)
        v_c = value[:, :, start:end, :].to(dtype=compute_dtype)
        beta_c = beta[:, :, start:end].to(dtype=compute_dtype)
        lg_c = log_gamma[:, :, start:end].to(dtype=compute_dtype)

        if c_len == 1:
            q_0 = q_c[:, :, 0, :]
            k_0 = k_c[:, :, 0, :]
            v_0 = v_c[:, :, 0, :]
            beta_0 = beta_c[:, :, 0].unsqueeze(-1).unsqueeze(-1)
            gamma_0 = torch.exp(lg_c[:, :, 0]).unsqueeze(-1).unsqueeze(-1)

            qk_0 = torch.stack((q_0, k_0), dim=-2)
            read_and_pred = torch.matmul(qk_0, state)
            reads[:, :, start, :] = read_and_pred[:, :, 0, :].to(dtype=query.dtype)
            pred_v = read_and_pred[:, :, 1, :]
            err = v_0 - pred_v
            state = gamma_0 * state + beta_0 * (
                k_0.unsqueeze(-1) * err.unsqueeze(-2)
            )
            continue

        strict_tril_bool, strict_tril_float, eye = _get_chunk_scan_buffers(
            c_len, query.device, compute_dtype
        )

        cs = torch.cumsum(lg_c, dim=-1)
        gamma_from_start = torch.exp(cs)
        prev_cs = cs - lg_c
        gamma_before = torch.exp(prev_cs)

        prev_diff = prev_cs.unsqueeze(-1) - cs.unsqueeze(-2)
        safe_prev_diff = prev_diff.masked_fill(~strict_tril_bool, 0.0)
        prev_decay = torch.exp(safe_prev_diff) * strict_tril_float

        kk = torch.matmul(k_c, k_c.transpose(-1, -2))
        lower_m = prev_decay * beta_c.unsqueeze(-2) * kk

        k_state_pred = gamma_before.unsqueeze(-1) * torch.matmul(k_c, state)
        rhs = v_c - k_state_pred
        pseudo_u = torch.linalg.solve_triangular(
            eye + lower_m, rhs, upper=False, unitriangular=True
        )

        qk = torch.matmul(q_c, k_c.transpose(-1, -2))
        read_weights = prev_decay * beta_c.unsqueeze(-2) * qk
        q_state_read = gamma_before.unsqueeze(-1) * torch.matmul(q_c, state)
        reads[:, :, start:end, :] = (
            q_state_read + torch.matmul(read_weights, pseudo_u)
        ).to(dtype=query.dtype)

        decay_to_end = torch.exp(cs[..., -1:] - cs) * beta_c
        weighted_u = decay_to_end.unsqueeze(-1) * pseudo_u
        end_decay = gamma_from_start[..., -1].unsqueeze(-1).unsqueeze(-1)
        state = end_decay * state + torch.matmul(k_c.transpose(-1, -2), weighted_u)

    return reads, state.to(dtype=memory_state.dtype)


class DeltaRuleAssociativeMemory(nn.Module):
    """Dual-stream fast-weight memory with 1-step key shift & bigram phrase support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int = 3,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        self.phrase_mix = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.phrase_mix.weight)

        self.gate_conv = CausalDepthwiseConv1d(d_model, kernel_size=conv_kernel_size)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.beta_projection = nn.Linear(d_model, num_heads)
        self.gamma_projection = nn.Linear(d_model, num_heads)
        self.output_projection = nn.Linear(d_model, d_model)

        nn.init.constant_(self.beta_projection.bias, 0.0)
        nn.init.constant_(self.gamma_projection.bias, 4.0)

    def _project_inputs(
        self,
        token_stream: torch.Tensor,
        context_hidden: torch.Tensor,
        prev_token: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, sequence_length, _ = token_stream.shape
        shape = (batch_size, sequence_length, self.num_heads, self.head_dim)

        if prev_token is None:
            prev_raw = token_stream.new_zeros(batch_size, 1, self.d_model)
            prev_local = token_stream.new_zeros(batch_size, 1, self.d_model)
        else:
            expected_prev = (batch_size, 2, self.d_model)
            if tuple(prev_token.shape) != expected_prev:
                raise ValueError(
                    f"prev_token must have shape {expected_prev}, "
                    f"received {tuple(prev_token.shape)}"
                )
            if prev_token.device != token_stream.device:
                raise ValueError("prev_token and token_stream must be on the same device")
            typed_prev = prev_token.to(dtype=token_stream.dtype)
            prev_raw = typed_prev[:, 0:1, :]
            prev_local = typed_prev[:, 1:2, :]

        shifted_tokens = torch.cat((prev_raw, token_stream[:, :-1, :]), dim=1)

        local_tokens = token_stream + self.phrase_mix(shifted_tokens)
        shifted_local = torch.cat((prev_local, local_tokens[:, :-1, :]), dim=1)
        next_prev_token = torch.cat(
            (token_stream[:, -1:, :], local_tokens[:, -1:, :]), dim=1
        )

        gate_conv_out, next_conv_state = self.gate_conv(
            context_hidden, conv_state=conv_state, return_state=True
        )
        gate_features = F.silu(gate_conv_out)

        query = self.query_projection(local_tokens).view(shape).transpose(1, 2)
        key = self.key_projection(shifted_local).view(shape).transpose(1, 2)
        value = self.value_projection(token_stream).view(shape).transpose(1, 2)

        query = F.normalize(query, p=2.0, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2.0, dim=-1, eps=1e-6)

        beta = torch.sigmoid(self.beta_projection(gate_features)).transpose(1, 2)
        log_gamma = F.logsigmoid(self.gamma_projection(gate_features)).transpose(1, 2)
        return query, key, value, beta, log_gamma, next_prev_token, next_conv_state

    def forward(
        self,
        token_stream: torch.Tensor,
        context_hidden: Optional[torch.Tensor] = None,
        initial_state: Optional[
            torch.Tensor
            | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if token_stream.ndim != 3:
            raise ValueError("token_stream must have shape [batch, sequence, d_model]")
        if context_hidden is None:
            context_hidden = token_stream
        if context_hidden.shape != token_stream.shape:
            raise ValueError("context_hidden and token_stream must have identical shapes")

        batch_size, sequence_length, width = token_stream.shape
        if width != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, received {width}")
        if sequence_length < 1:
            raise ValueError("Memory sequence length must be at least 1")

        state_shape = (batch_size, self.num_heads, self.head_dim, self.head_dim)
        if initial_state is None:
            memory_state = token_stream.new_zeros(state_shape)
            prev_token = None
            conv_state = None
        elif isinstance(initial_state, tuple):
            raw_mem, prev_token, conv_state = initial_state
            if tuple(raw_mem.shape) != state_shape:
                raise ValueError(
                    f"initial memory_state must have shape {state_shape}, "
                    f"received {tuple(raw_mem.shape)}"
                )
            if raw_mem.device != token_stream.device:
                raise ValueError("initial_state and token_stream must be on the same device")
            memory_state = raw_mem.to(dtype=token_stream.dtype)
        else:
            if tuple(initial_state.shape) != state_shape:
                raise ValueError(
                    f"initial_state must have shape {state_shape}, "
                    f"received {tuple(initial_state.shape)}"
                )
            if initial_state.device != token_stream.device:
                raise ValueError("initial_state and token_stream must be on the same device")
            memory_state = initial_state.to(dtype=token_stream.dtype)
            prev_token = None
            conv_state = None

        query, key, value, beta, log_gamma, next_prev_token, next_conv_state = (
            self._project_inputs(
                token_stream=token_stream,
                context_hidden=context_hidden,
                prev_token=prev_token,
                conv_state=conv_state,
            )
        )
        read_sequence, memory_state = _gated_delta_chunkwise_scan(
            query=query,
            key=key,
            value=value,
            beta=beta,
            log_gamma=log_gamma,
            memory_state=memory_state,
            chunk_size=self.chunk_size,
        )
        read_sequence = read_sequence.transpose(1, 2).contiguous()
        read_sequence = read_sequence.view(batch_size, sequence_length, self.d_model)
        output = self.output_projection(read_sequence)
        if return_state:
            return output, (memory_state, next_prev_token, next_conv_state)
        return output


class GutenbergHybridLayer(nn.Module):
    """Interleaved Causal-Conv SSM + Dual-Stream Delta Memory + SwiGLU Layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ssm_block = SelectiveSSMBlock(
            d_model=config.d_model,
            state_size=config.state_size,
            conv_kernel_size=config.conv_kernel_size,
            ssm_chunk_size=config.ssm_chunk_size,
            ffn_expansion=config.ffn_expansion,
            dropout=config.dropout,
        )
        if config.use_associative_memory:
            self.token_memory_norm: Optional[nn.LayerNorm] = nn.LayerNorm(config.d_model)
            self.memory_norm: Optional[nn.LayerNorm] = nn.LayerNorm(config.d_model)
            self.memory: Optional[DeltaRuleAssociativeMemory] = (
                DeltaRuleAssociativeMemory(
                    d_model=config.d_model,
                    num_heads=config.memory_heads,
                    conv_kernel_size=config.conv_kernel_size,
                    chunk_size=min(32, config.ssm_chunk_size),
                )
            )
            self.memory_gate: Optional[nn.Parameter] = nn.Parameter(torch.tensor(0.0))
            self.memory_ffn: Optional[SwiGLUFFN] = SwiGLUFFN(
                d_model=config.d_model,
                expansion=config.ffn_expansion,
                dropout=config.dropout,
            )
        else:
            self.token_memory_norm = None
            self.memory_norm = None
            self.memory = None
            self.memory_gate = None
            self.memory_ffn = None

    def forward(
        self,
        hidden: torch.Tensor,
        token_embeds: torch.Tensor,
        layer_state: Optional[tuple[Any, Any]] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[Any, Any]]:
        ssm_init = layer_state[0] if layer_state is not None else None
        mem_init = layer_state[1] if layer_state is not None else None

        hidden, next_ssm_state = self.ssm_block(
            hidden, initial_state=ssm_init, return_state=True
        )
        next_mem_state = None
        if (
            self.memory is not None
            and self.token_memory_norm is not None
            and self.memory_norm is not None
            and self.memory_gate is not None
            and self.memory_ffn is not None
        ):
            memory_output, next_mem_state = self.memory(
                token_stream=self.token_memory_norm(token_embeds),
                context_hidden=self.memory_norm(hidden),
                initial_state=mem_init,
                return_state=True,
            )
            hidden = (
                hidden
                + torch.sigmoid(self.memory_gate).to(dtype=memory_output.dtype)
                * memory_output
            )
            hidden = self.memory_ffn(hidden)

        if return_state:
            return hidden, (next_ssm_state, next_mem_state)
        return hidden


class GutenbergChaoticPrimordial(nn.Module):
    """Multi-Layer Hybrid SSM + Associative Memory Language Model with O(1) Streaming."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(
            self.token_embedding.weight, mean=0.0, std=1.0 / math.sqrt(config.d_model)
        )
        self.layers = nn.ModuleList(
            [GutenbergHybridLayer(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.language_model_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.language_model_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        state: Optional[list[tuple[Any, Any]]] = None,
        return_state: bool = False,
        validate_inputs: bool = True,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, list[tuple[Any, Any]]]
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if validate_inputs:
            if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
                raise ValueError(
                    "input_ids contain token IDs outside the configured vocabulary"
                )
        if state is not None and len(state) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} layer states, got {len(state)}")

        token_embeds = self.token_embedding(input_ids)
        hidden = token_embeds
        next_states: list[tuple[Any, Any]] = []

        for idx, layer in enumerate(self.layers):
            layer_init = state[idx] if state is not None else None
            hidden, next_layer_state = layer(
                hidden,
                token_embeds=token_embeds,
                layer_state=layer_init,
                return_state=True,
            )
            if return_state:
                next_states.append(next_layer_state)

        hidden = self.final_norm(hidden)
        logits = self.language_model_head(hidden)

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same [batch, sequence] shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size).float(),
                labels.reshape(-1).long(),
                ignore_index=PAD_TOKEN_ID,
            )
            return logits, loss

        if return_state:
            return logits, next_states
        return logits

    def memory_gate_summary(self) -> list[float]:
        gates: list[float] = []
        for layer in self.layers:
            if layer.memory_gate is not None:
                gates.append(round(float(torch.sigmoid(layer.memory_gate).item()), 4))
        return gates

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: Optional[int] = EOS_TOKEN_ID,
    ) -> torch.Tensor:
        """O(1)-per-token stateful autoregressive generation."""
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1:
            raise ValueError("prompt_ids must have shape [batch, prompt_len >= 1]")
        was_training = self.training
        self.eval()

        logits, state = self.forward(
            prompt_ids, state=None, return_state=True, validate_inputs=True
        )
        generated = [prompt_ids]
        next_logits = logits[:, -1, :].float()

        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if 0 < top_k < scaled.shape[-1]:
                    top_vals, _ = torch.topk(scaled, k=top_k, dim=-1)
                    cutoff = top_vals[:, -1:].expand_as(scaled)
                    scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)
            if (
                eos_token_id is not None
                and prompt_ids.shape[0] == 1
                and int(next_token.item()) == eos_token_id
            ):
                break

            step_logits, state = self.forward(
                next_token, state=state, return_state=True, validate_inputs=False
            )
            next_logits = step_logits[:, -1, :].float()

        if was_training:
            self.train()
        return torch.cat(generated, dim=1)


def expand_checkpoint_to_model(
    source_checkpoint: Path,
    target_config: ModelConfig,
    device: torch.device,
) -> GutenbergChaoticPrimordial:
    """Function-preserving width & depth expansion from a smaller v0.3.x checkpoint."""
    src_model = load_checkpoint(source_checkpoint, device)
    src_cfg = src_model.config
    if target_config.d_model < src_cfg.d_model:
        raise ValueError(
            f"Target d_model ({target_config.d_model}) must be >= source ({src_cfg.d_model})"
        )
    if target_config.num_layers < src_cfg.num_layers:
        raise ValueError(
            f"Target layers ({target_config.num_layers}) must be >= source ({src_cfg.num_layers})"
        )
    if target_config.state_size < src_cfg.state_size:
        raise ValueError(
            f"Target state_size ({target_config.state_size}) must be >= source ({src_cfg.state_size})"
        )
    if target_config.memory_heads != src_cfg.memory_heads:
        raise ValueError(
            f"Target heads ({target_config.memory_heads}) must match source ({src_cfg.memory_heads})"
        )

    dst_model = GutenbergChaoticPrimordial(target_config).to(device)
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    norm_scale = math.sqrt(float(src_cfg.d_model) / float(target_config.d_model))

    with torch.no_grad():
        for name, dst_tensor in dst_state.items():
            if name in src_state:
                src_tensor = src_state[name].to(device=device, dtype=dst_tensor.dtype)
                if dst_tensor.ndim == 0:
                    dst_tensor.copy_(src_tensor)
                    continue
                # Initialize extra channels with small symmetry-breaking noise
                if dst_tensor.is_floating_point():
                    dst_tensor.normal_(mean=0.0, std=1e-3)
                slices = tuple(slice(0, s) for s in src_tensor.shape)
                dst_tensor[slices].copy_(src_tensor)
                # Compensate LayerNorm across wider d_model so hidden scale stays invariant
                if name.endswith("norm.weight") and target_config.d_model > src_cfg.d_model:
                    dst_tensor[: src_cfg.d_model].mul_(norm_scale)

        # Initialize newly added upper layers as near-identity residual blocks
        for layer_idx in range(src_cfg.num_layers, target_config.num_layers):
            layer = dst_model.layers[layer_idx]
            layer.ssm_block.ssm.output_scale.fill_(0.05)
            layer.ssm_block.ffn.down_proj.weight.mul_(0.05)
            if layer.memory_gate is not None:
                layer.memory_gate.fill_(-3.0)
            if layer.memory_ffn is not None:
                layer.memory_ffn.down_proj.weight.mul_(0.05)

    return dst_model


def build_optimizer_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Apply weight decay only to 2D+ matrices; protect 1D gates/norms/decays."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if (
            param.ndim < 2
            or name.endswith(".bias")
            or "norm" in name
            or "log_decay" in name
            or "memory_gate" in name
            or "output_scale" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_threads(requested_threads: int) -> int:
    if requested_threads < 0:
        raise ValueError("--threads must be >= 0 (0 = auto-detect physical cores)")
    if requested_threads > 0:
        return requested_threads
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this PyTorch environment")
        return torch.device("cuda")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device selection: {device_name}")


def resolve_amp_dtype(dtype_name: str, device: torch.device) -> Optional[torch.dtype]:
    if dtype_name == "float32":
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    raise ValueError(f"Unsupported dtype selection: {dtype_name}")


def _assert_finite(tensor: torch.Tensor, description: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise AssertionError(f"Non-finite values detected in {description}")


def _verify_all_gradients(model: nn.Module, context: str) -> int:
    gradient_tensors = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    missing_gradients = sum(gradient is None for gradient in gradient_tensors)
    finite_gradients = sum(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in gradient_tensors
    )
    if missing_gradients:
        raise AssertionError(
            f"{missing_gradients} trainable parameters did not receive gradients in {context}"
        )
    if finite_gradients != len(gradient_tensors):
        raise AssertionError(f"Non-finite gradients detected in {context}")
    return finite_gradients


def run_self_tests(device: torch.device) -> dict[str, Any]:
    """Verify chunkwise SSM scan, chunkwise delta memory, and streaming generation."""
    results: dict[str, Any] = {
        "architecture_version": ARCHITECTURE_VERSION,
        "device": str(device),
        "tests": [],
    }

    set_reproducible_seed(1701)
    block = SelectiveSSMBlock(
        d_model=16,
        state_size=4,
        conv_kernel_size=3,
        ssm_chunk_size=16,
        ffn_expansion=2.0,
        dropout=0.0,
    ).to(device)
    x = torch.randn(2, 73, 16, device=device)
    full_out, full_state = block(x, return_state=True)
    left_out, mid_state = block(x[:, :29, :], return_state=True)
    right_out, right_state = block(x[:, 29:, :], initial_state=mid_state, return_state=True)
    chunk_out = torch.cat((left_out, right_out), dim=1)
    _assert_finite(full_out, "SelectiveSSMBlock output")
    if not torch.allclose(full_out, chunk_out, atol=2e-5, rtol=2e-4):
        raise AssertionError("SelectiveSSMBlock full and chunked outputs differ")
    if not torch.allclose(full_state[0], right_state[0], atol=2e-5, rtol=2e-4):
        raise AssertionError("SelectiveSSMBlock SSM states differ")
    if not torch.allclose(full_state[1], right_state[1], atol=2e-6, rtol=2e-5):
        raise AssertionError("SelectiveSSMBlock Conv1d states differ")
    results["tests"].append(
        {
            "name": "chunkwise_ssm_block_continuation",
            "status": "PASS",
            "max_chunk_output_abs_error": float((full_out - chunk_out).abs().max().item()),
        }
    )

    memory = DeltaRuleAssociativeMemory(
        d_model=16, num_heads=4, conv_kernel_size=3, chunk_size=8
    ).to(device)
    tok_in = torch.randn(2, 19, 16, device=device)
    ctx_in = torch.randn(2, 19, 16, device=device)
    act_out, act_state = memory(tok_in, ctx_in, return_state=True)
    left_m_out, left_m_state = memory(tok_in[:, :7, :], ctx_in[:, :7, :], return_state=True)
    right_m_out, right_m_state = memory(
        tok_in[:, 7:, :], ctx_in[:, 7:, :], initial_state=left_m_state, return_state=True
    )
    chunk_m_out = torch.cat((left_m_out, right_m_out), dim=1)
    if not torch.allclose(act_out, chunk_m_out, atol=2e-5, rtol=2e-4):
        raise AssertionError("DeltaRuleAssociativeMemory full and chunked outputs differ")
    if not torch.allclose(act_state[0], right_m_state[0], atol=2e-5, rtol=2e-4):
        raise AssertionError("DeltaRuleAssociativeMemory matrix states differ")
    results["tests"].append(
        {
            "name": "chunkwise_delta_memory_continuation",
            "status": "PASS",
            "max_chunk_output_abs_error": float((act_out - chunk_m_out).abs().max().item()),
        }
    )

    cfg = ModelConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        d_model=32,
        num_layers=2,
        state_size=8,
        memory_heads=4,
        conv_kernel_size=3,
        ssm_chunk_size=16,
        ffn_expansion=1.5,
        dropout=0.0,
        use_associative_memory=True,
    )
    model = GutenbergChaoticPrimordial(cfg).to(device)
    model.eval()
    ids = torch.randint(SPECIAL_TOKENS, cfg.vocab_size, (2, 25), device=device)
    with torch.no_grad():
        full_logits = model(ids)
        step_logits_list: list[torch.Tensor] = []
        stream_state = None
        for t in range(ids.shape[1]):
            s_out, stream_state = model(
                ids[:, t : t + 1], state=stream_state, return_state=True
            )
            step_logits_list.append(s_out)
        step_logits = torch.cat(step_logits_list, dim=1)
    if not torch.allclose(full_logits, step_logits, atol=2e-5, rtol=2e-4):
        raise AssertionError("Full forward logits differ from O(1) token-by-token streaming")

    model.train()
    logits, loss = model(ids, labels=ids)
    _assert_finite(logits, "full model logits")
    _assert_finite(loss, "full model loss")
    loss.backward()
    grad_count = _verify_all_gradients(model, "full_hybrid_model")
    results["tests"].append(
        {
            "name": "full_model_streaming_equivalence_and_gradients",
            "status": "PASS",
            "max_streaming_logit_abs_error": float(
                (full_logits - step_logits).abs().max().item()
            ),
            "trainable_tensors_with_finite_gradients": grad_count,
        }
    )

    results["overall_status"] = "PASS"
    return results


@torch.no_grad()
def evaluate_language_model(
    model: GutenbergChaoticPrimordial,
    val_ds: PackedSequenceDataset,
    batch_size: int,
    eval_batches: int,
    seed: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> tuple[float, float]:
    """Evaluate held-out cross-entropy loss and perplexity."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed + 4242)
    total_loss = 0.0
    steps = max(1, eval_batches)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )
    for _ in range(steps):
        x, y = val_ds.sample_batch(batch_size, gen, device)
        with amp_ctx:
            _, loss = model(x, labels=y, validate_inputs=False)
        total_loss += float(loss.item())
    avg_loss = total_loss / steps
    perplexity = math.exp(min(avg_loss, 20.0))
    if was_training:
        model.train()
    return avg_loss, perplexity


# ---------------------------------------------------------------------------
# Stage-1 & Stage-2 Math Curriculum (Used Standalone & As Consolidation Replay)
# ---------------------------------------------------------------------------


def _inline_add(a: int, b: int) -> str:
    total = a + b
    a_t, a_o = divmod(a, 10)
    b_t, b_o = divmod(b, 10)
    c1, d_o = divmod(a_o + b_o, 10)
    return f"{a}+{b}[O:{a_o}+{b_o}={d_o}(c{c1}),T:{a_t}+{b_t}+c{c1}={a_t + b_t + c1}]={total}"


def _inline_sub(hi: int, lo: int) -> str:
    diff = hi - lo
    h_t, h_o = divmod(hi, 10)
    l_t, l_o = divmod(lo, 10)
    if h_o < l_o:
        b1 = 1
        d_o = (h_o + 10) - l_o
        o_str = f"O:{h_o}<{l_o}->{h_o + 10}-{l_o}={d_o}(b1)"
    else:
        b1 = 0
        d_o = h_o - l_o
        o_str = f"O:{h_o}>={l_o}->{h_o}-{l_o}={d_o}(b0)"
    d_t = (h_t - b1) - l_t
    return f"{hi}-{lo}[{o_str},T:{h_t - b1}-{l_t}={d_t}]={diff}"


def _inline_div(num: int, den: int) -> str:
    q = num // den
    return f"{num}/{den}[{den}*{q}={num}]={q}"


def _format_carry_addition(a: int, b: int) -> str:
    total = a + b
    a_h, rem_a = divmod(a, 100)
    a_t, a_o = divmod(rem_a, 10)
    b_h, rem_b = divmod(b, 100)
    b_t, b_o = divmod(rem_b, 10)

    c1, d_o = divmod(a_o + b_o, 10)
    c2, d_t = divmod(a_t + b_t + c1, 10)
    d_h = a_h + b_h + c2
    return (
        f"Add: {a} + {b} | A=({a_h},{a_t},{a_o}) B=({b_h},{b_t},{b_o}) -> "
        f"Ones: {a_o}+{b_o}={d_o}(c{c1}), Tens: {a_t}+{b_t}+c{c1}={d_t}(c{c2}), "
        f"Hundreds: {a_h}+{b_h}+c{c2}={d_h} -> Sum: {total}"
    )


def _format_borrow_subtraction(hi: int, lo: int) -> str:
    diff = hi - lo
    h_h, rem_h = divmod(hi, 100)
    h_t, h_o = divmod(rem_h, 10)
    l_h, rem_l = divmod(lo, 100)
    l_t, l_o = divmod(rem_l, 10)

    if h_o < l_o:
        b1 = 1
        d_o = (h_o + 10) - l_o
        ones_trace = f"{h_o}<{l_o}->{h_o + 10}-{l_o}={d_o}(b1)"
    else:
        b1 = 0
        d_o = h_o - l_o
        ones_trace = f"{h_o}>={l_o}->{h_o}-{l_o}={d_o}(b0)"

    if b1 == 1 and h_t == 0:
        adj_t = 9
        pre_b2 = 1
        tens_top_trace = "0-1=9(b1)"
    else:
        adj_t = h_t - b1
        pre_b2 = 0
        tens_top_trace = f"{h_t}-{b1}={adj_t}"

    if adj_t < l_t:
        b2 = 1
        d_t = (adj_t + 10) - l_t
        tens_trace = f"{adj_t}<{l_t}->{adj_t + 10}-{l_t}={d_t}(b1)"
    else:
        b2 = pre_b2
        d_t = adj_t - l_t
        tens_trace = f"{adj_t}>={l_t}->{adj_t}-{l_t}={d_t}(b{b2})"

    d_h = (h_h - b2) - l_h
    return (
        f"Sub: {hi} - {lo} | A=({h_h},{h_t},{h_o}) B=({l_h},{l_t},{l_o}) -> "
        f"Ones: {ones_trace}, TensTop: {tens_top_trace}, "
        f"Tens: {tens_trace}, Hundreds: {h_h}-{l_h}-b{b2}={d_h} -> Diff: {diff}"
    )


def _format_single_digit_mul(a: int, m: int) -> str:
    prod = a * m
    a_t, a_o = divmod(a, 10)
    p_o = a_o * m
    c1, d_o = divmod(p_o, 10)
    p_t = a_t * m
    d_rest = p_t + c1
    return (
        f"Mul: {a} * {m} | A=({a_t},{a_o})*{m} -> "
        f"OnesMul: {a_o}*{m}={p_o} -> {d_o}(c{c1}), "
        f"TensMul: {a_t}*{m}={p_t}, CarryAdd: {p_t}+{c1}={d_rest} -> Prod: {prod}"
    )


def _format_base10_to_base2(n: int) -> str:
    bits = f"{n:08b}"
    hi_nib, lo_nib = divmod(n, 16)
    return (
        f"Base10: {n} -> Div16: {n}={hi_nib}*16+{lo_nib} -> "
        f"Nibbles: {hi_nib:04b}_{lo_nib:04b} -> Base2: {bits}"
    )


def _gen_prealgebra_record(rng: random.Random) -> str:
    kind = rng.randint(0, 3)
    if kind == 0:
        a = rng.randint(2, 45)
        b = rng.randint(2, 12)
        c = rng.randint(2, 12)
        prod = b * c
        add_step = _inline_add(a, prod)
        return (
            f"PreAlg: Eval {a} + {b} * {c} -> "
            f"Mul: {b}*{c}={prod} -> Add: {add_step} -> Ans: {a + prod}"
        )
    if kind == 1:
        base = rng.randint(2, 15)
        sq = base * base
        if rng.random() < 0.5:
            return f"PreAlg: Power {base}^2 -> {base}*{base}={sq} -> Ans: {sq}"
        return f"PreAlg: Sqrt({sq}) -> {base}*{base}={sq} -> Ans: {base}"
    if kind == 2:
        g = rng.randint(2, 12)
        p = rng.randint(1, 9)
        q = rng.randint(p + 1, 11)
        while math.gcd(p, q) != 1:
            q += 1
        num, den = p * g, q * g
        return (
            f"PreAlg: Simplify {num}/{den} -> "
            f"GCD={g} -> {_inline_div(num, g)}, {_inline_div(den, g)} -> Ans: {p}/{q}"
        )
    pct = rng.choice((10, 20, 25, 30, 40, 50, 75))
    mult = rng.randint(2, 16)
    val = mult * 20
    q20 = val // 20
    p5 = pct // 5
    ans = q20 * p5
    return (
        f"PreAlg: {pct}% of {val} -> "
        f"Step1: {_inline_div(val, 20)} -> Step2: {q20}*({pct}/5)={q20}*{p5}={ans} -> Ans: {ans}"
    )


def _gen_algebra_record(rng: random.Random) -> str:
    kind = rng.randint(0, 3)
    if kind == 0:
        a = rng.randint(2, 12)
        x = rng.randint(1, 15)
        b = rng.randint(1, 35)
        c = a * x + b
        rhs = c - b
        sub_step = _inline_sub(c, b)
        div_step = _inline_div(rhs, a)
        return (
            f"Alg: Solve {a}x + {b} = {c} -> "
            f"Sub {b}: {a}x = {sub_step} -> Div {a}: x = {div_step} -> Ans: x={x}"
        )
    if kind == 1:
        d = rng.randint(1, 8)
        diff_a = rng.randint(2, 7)
        a = d + diff_a
        x = rng.randint(1, 14)
        b = rng.randint(1, 20)
        e = a * x + b - d * x
        rhs = e - b
        return (
            f"Alg: Solve {a}x + {b} = {d}x + {e} -> "
            f"Collect: ({a}-{d})={diff_a}x, {_inline_sub(e, b)} -> Div: x = {_inline_div(rhs, diff_a)} -> Ans: x={x}"
        )
    if kind == 2:
        r1 = rng.randint(1, 11)
        r2 = rng.randint(1, 11)
        b_coef = r1 + r2
        c_coef = r1 * r2
        return (
            f"Alg: Expand (x + {r1})(x + {r2}) -> "
            f"Sum={r1}+{r2}={b_coef}, Prod={r1}*{r2}={c_coef} -> Ans: x^2 + {b_coef}x + {c_coef}"
        )
    r1 = rng.randint(1, 12)
    r2 = rng.randint(r1 + 1, 15)
    return (
        f"Alg: Roots of (x - {r1})(x - {r2}) = 0 -> "
        f"Factors: x-{r1}=0, x-{r2}=0 -> Ans: x={r1}, {r2}"
    )


def _gen_geometry_record(rng: random.Random) -> str:
    kind = rng.randint(0, 3)
    if kind == 0:
        w = rng.randint(3, 20)
        h = rng.randint(3, 20)
        area = w * h
        wh_sum = w + h
        perim = 2 * wh_sum
        return (
            f"Geom: Rectangle w={w} h={h} -> "
            f"Area={w}*{h}={area}, Sum={_inline_add(w, h)}, Perim=2*{wh_sum}={perim} -> Ans: Area={area}, Perim={perim}"
        )
    if kind == 1:
        b = rng.randint(2, 14) * 2
        h = rng.randint(3, 18)
        prod = b * h
        area = prod // 2
        return (
            f"Geom: Triangle base={b} height={h} -> "
            f"Prod={b}*{h}={prod} -> Half={_inline_div(prod, 2)} -> Ans: Area={area}"
        )
    if kind == 2:
        base_triples = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25))
        a0, b0, c0 = rng.choice(base_triples)
        k = rng.randint(1, 4)
        a, b, c = a0 * k, b0 * k, c0 * k
        sq_a, sq_b = a * a, b * b
        sum_step = (
            _inline_add(sq_a, sq_b)
            if (sq_a < 100 and sq_b < 100)
            else f"{sq_a}+{sq_b}={sq_a + sq_b}"
        )
        return (
            f"Geom: RightTriangle legs a={a} b={b} -> "
            f"Sq: {a}^2={sq_a}, {b}^2={sq_b} -> Sum={sum_step}={c}^2 -> Ans: c={c}"
        )
    r = rng.randint(2, 16)
    return (
        f"Geom: Circle radius r={r} -> "
        f"Circum=2*{r}pi={2 * r}pi, Area={r}*{r}pi={r * r}pi -> Ans: C={2 * r}pi, A={r * r}pi"
    )


def _gen_trigonometry_record(rng: random.Random) -> str:
    kind = rng.randint(0, 2)
    if kind == 0:
        table = (
            ("sin(0 deg)", "0"),
            ("cos(0 deg)", "1"),
            ("tan(0 deg)", "0"),
            ("sin(30 deg)", "1/2"),
            ("cos(30 deg)", "sqrt(3)/2"),
            ("tan(30 deg)", "1/sqrt(3)"),
            ("sin(45 deg)", "sqrt(2)/2"),
            ("cos(45 deg)", "sqrt(2)/2"),
            ("tan(45 deg)", "1"),
            ("sin(60 deg)", "sqrt(3)/2"),
            ("cos(60 deg)", "1/2"),
            ("tan(60 deg)", "sqrt(3)"),
            ("sin(90 deg)", "1"),
            ("cos(90 deg)", "0"),
        )
        expr, val = rng.choice(table)
        return f"Trig: Exact {expr} -> UnitCircle -> Ans: {val}"
    if kind == 1:
        deg_map = (
            (30, "pi/6"),
            (45, "pi/4"),
            (60, "pi/3"),
            (90, "pi/2"),
            (120, "2pi/3"),
            (135, "3pi/4"),
            (150, "5pi/6"),
            (180, "pi"),
            (270, "3pi/2"),
            (360, "2pi"),
        )
        deg, rad = rng.choice(deg_map)
        return f"Trig: {deg} deg to rad -> {deg} * (pi/180) -> Ans: {rad}"
    triples = ((3, 4, 5), (5, 12, 13), (8, 15, 17))
    opp, adj, hyp = rng.choice(triples)
    fn = rng.choice(("sin", "cos", "tan"))
    if fn == "sin":
        return f"Trig: RightTri opp={opp} adj={adj} hyp={hyp} find sin(theta) -> opp/hyp -> Ans: {opp}/{hyp}"
    if fn == "cos":
        return f"Trig: RightTri opp={opp} adj={adj} hyp={hyp} find cos(theta) -> adj/hyp -> Ans: {adj}/{hyp}"
    return f"Trig: RightTri opp={opp} adj={adj} hyp={hyp} find tan(theta) -> opp/adj -> Ans: {opp}/{adj}"


def _gen_calculus_record(rng: random.Random) -> str:
    kind = rng.randint(0, 3)
    if kind == 0:
        a = rng.randint(2, 12)
        n = rng.randint(2, 6)
        b = rng.randint(1, 15)
        c = rng.randint(1, 20)
        da = a * n
        new_n = n - 1
        x_term = f"{da}x" if new_n == 1 else f"{da}x^{new_n}"
        return (
            f"Calc: d/dx [{a}x^{n} + {b}x + {c}] -> "
            f"PowerRule: {a}*{n}={da}, exp={n}-1={new_n} -> Ans: {x_term} + {b}"
        )
    if kind == 1:
        a = rng.randint(1, 8)
        b = rng.randint(1, 12)
        x0 = rng.randint(1, 7)
        two_a = 2 * a
        prod = two_a * x0
        deriv_at_x0 = prod + b
        return (
            f"Calc: f(x)={a}x^2+{b}x, find f'({x0}) -> "
            f"f'(x)={two_a}x+{b} -> {two_a}*{x0}={prod} -> {_inline_add(prod, b)} -> Ans: {deriv_at_x0}"
        )
    if kind == 2:
        n = rng.randint(1, 5)
        new_n = n + 1
        mult = rng.randint(1, 8)
        k = mult * new_n
        return (
            f"Calc: Int ({k}x^{n}) dx -> "
            f"NewExp={n}+1={new_n} -> Coef={_inline_div(k, new_n)} -> Ans: {mult}x^{new_n} + C"
        )
    a = rng.randint(1, 9)
    b = rng.randint(2, 8)
    b_sq = b * b
    ans = a * b_sq
    return (
        f"Calc: Int_0^{b} ({2 * a}x) dx -> "
        f"Antideriv: {a}x^2 -> {b}^2={b_sq} -> {a}*{b_sq}={ans} -> Ans: {ans}"
    )


def _builtin_stage2_math_splits(seed: int = 1701) -> tuple[list[str], list[str]]:
    """Build balanced Stage-1 + Stage-2 math corpus (standalone or distillation replay)."""
    rng = random.Random(seed)
    records: list[str] = []

    for _ in range(4):
        for n in range(256):
            records.append(_format_base10_to_base2(n))

    for _ in range(1200):
        hi = rng.randint(30, 255)
        lo = rng.randint(5, hi)
        a = rng.randint(0, 255)
        b = rng.randint(0, 255)
        m_a = rng.randint(10, 99)
        m_b = rng.randint(2, 9)
        records.append(_format_borrow_subtraction(hi, lo))
        records.append(_format_carry_addition(a, b))
        records.append(_format_single_digit_mul(m_a, m_b))

    for _ in range(1000):
        records.append(_gen_geometry_record(rng))
        records.append(_gen_prealgebra_record(rng))
        records.append(_gen_algebra_record(rng))
        records.append(_gen_calculus_record(rng))
        records.append(_gen_trigonometry_record(rng))

    rng.shuffle(records)
    split_idx = int(len(records) * 0.95)
    return records[:split_idx], records[split_idx:]


def evaluate_math_proficiency(
    model: GutenbergChaoticPrimordial,
    device: torch.device,
    seed: int = 9999,
) -> dict[str, Any]:
    """Run deterministic pass/fail benchmark on 40 unseen arithmetic & base problems."""
    tokenizer = ByteLevelTokenizer()
    rng = random.Random(seed)
    cases: list[tuple[str, str, str]] = []

    for _ in range(10):
        a, b = rng.randint(10, 250), rng.randint(10, 250)
        cases.append(("add", f"Add: {a} + {b} |", f"Sum: {a + b}"))
    for _ in range(10):
        hi, lo = rng.randint(50, 255), rng.randint(10, 200)
        if hi < lo:
            hi, lo = lo, hi
        cases.append(("sub", f"Sub: {hi} - {lo} |", f"Diff: {hi - lo}"))
    for _ in range(10):
        a, m = rng.randint(10, 99), rng.randint(2, 9)
        cases.append(("mul", f"Mul: {a} * {m} |", f"Prod: {a * m}"))
    for _ in range(10):
        n = rng.randint(1, 255)
        cases.append(("base2", f"Base10: {n} ->", f"Base2: {n:08b}"))

    stats: dict[str, dict[str, int]] = {
        cat: {"passed": 0, "total": 0} for cat in ("add", "sub", "mul", "base2")
    }
    failed_samples: list[dict[str, Any]] = []

    for cat, prompt, expected_marker in cases:
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt, add_bos=True, add_eos=False)],
            dtype=torch.long,
            device=device,
        )
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=160,
            temperature=0.0,
            top_k=1,
            eos_token_id=EOS_TOKEN_ID,
        )
        text = tokenizer.decode(out_ids[0].tolist(), skip_special=True)
        passed = expected_marker in text
        stats[cat]["total"] += 1
        if passed:
            stats[cat]["passed"] += 1
        else:
            failed_samples.append(
                {
                    "category": cat,
                    "prompt": prompt,
                    "expected": expected_marker,
                    "generated": text,
                }
            )

    total_passed = sum(v["passed"] for v in stats.values())
    total_cases = sum(v["total"] for v in stats.values())
    overall_pct = round(100.0 * total_passed / max(1, total_cases), 1)
    breakdown = {
        cat: f"{v['passed']}/{v['total']} ({round(100.0 * v['passed'] / max(1, v['total']), 1)}%)"
        for cat, v in stats.items()
    }
    return {
        "passed": total_passed,
        "total": total_cases,
        "percentage": overall_pct,
        "overall_score": f"{total_passed}/{total_cases} ({overall_pct}%)",
        "foundation_100_pct": total_passed == total_cases,
        "breakdown": breakdown,
        "failed_samples": failed_samples[:8],
    }


def evaluate_higher_math_proficiency(
    model: GutenbergChaoticPrimordial,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark 25 unseen problems across Pre-Algebra, Algebra, Geometry, Trig, and Calculus."""
    tokenizer = ByteLevelTokenizer()
    cases: list[tuple[str, str, str]] = [
        ("prealg", "PreAlg: Eval 14 + 6 * 7 ->", "Ans: 56"),
        ("prealg", "PreAlg: Power 13^2 ->", "Ans: 169"),
        ("prealg", "PreAlg: Sqrt(144) ->", "Ans: 12"),
        ("prealg", "PreAlg: Simplify 24/36 ->", "Ans: 2/3"),
        ("prealg", "PreAlg: 25% of 240 ->", "Ans: 60"),
        ("algebra", "Alg: Solve 4x + 7 = 31 ->", "Ans: x=6"),
        ("algebra", "Alg: Solve 7x + 5 = 3x + 29 ->", "Ans: x=6"),
        ("algebra", "Alg: Expand (x + 3)(x + 8) ->", "Ans: x^2 + 11x + 24"),
        ("algebra", "Alg: Roots of (x - 4)(x - 9) = 0 ->", "Ans: x=4, 9"),
        ("algebra", "Alg: Solve 5x + 12 = 47 ->", "Ans: x=7"),
        ("geometry", "Geom: Rectangle w=9 h=14 ->", "Ans: Area=126, Perim=46"),
        ("geometry", "Geom: Triangle base=12 height=9 ->", "Ans: Area=54"),
        ("geometry", "Geom: RightTriangle legs a=6 b=8 ->", "Ans: c=10"),
        ("geometry", "Geom: Circle radius r=7 ->", "Ans: C=14pi, A=49pi"),
        ("geometry", "Geom: RightTriangle legs a=5 b=12 ->", "Ans: c=13"),
        ("trig", "Trig: Exact sin(30 deg) ->", "Ans: 1/2"),
        ("trig", "Trig: Exact cos(45 deg) ->", "Ans: sqrt(2)/2"),
        ("trig", "Trig: Exact tan(60 deg) ->", "Ans: sqrt(3)"),
        ("trig", "Trig: 120 deg to rad ->", "Ans: 2pi/3"),
        ("trig", "Trig: RightTri opp=3 adj=4 hyp=5 find sin(theta) ->", "Ans: 3/5"),
        ("calculus", "Calc: d/dx [4x^3 + 7x + 9] ->", "Ans: 12x^2 + 7"),
        ("calculus", "Calc: f(x)=3x^2+4x, find f'(5) ->", "Ans: 34"),
        ("calculus", "Calc: Int (12x^3) dx ->", "Ans: 3x^4 + C"),
        ("calculus", "Calc: Int_0^4 (6x) dx ->", "Ans: 48"),
        ("calculus", "Calc: d/dx [sin(x)] ->", "Ans: cos(x)"),
    ]

    stats: dict[str, dict[str, int]] = {
        cat: {"passed": 0, "total": 0}
        for cat in ("prealg", "algebra", "geometry", "trig", "calculus")
    }
    failed_samples: list[dict[str, Any]] = []

    for cat, prompt, expected_marker in cases:
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt, add_bos=True, add_eos=False)],
            dtype=torch.long,
            device=device,
        )
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=160,
            temperature=0.0,
            top_k=1,
            eos_token_id=EOS_TOKEN_ID,
        )
        text = tokenizer.decode(out_ids[0].tolist(), skip_special=True)
        passed = expected_marker in text
        stats[cat]["total"] += 1
        if passed:
            stats[cat]["passed"] += 1
        else:
            failed_samples.append(
                {
                    "category": cat,
                    "prompt": prompt,
                    "expected": expected_marker,
                    "generated": text,
                }
            )

    total_passed = sum(v["passed"] for v in stats.values())
    total_cases = sum(v["total"] for v in stats.values())
    overall_pct = round(100.0 * total_passed / max(1, total_cases), 1)
    breakdown = {
        cat: f"{v['passed']}/{v['total']} ({round(100.0 * v['passed'] / max(1, v['total']), 1)}%)"
        for cat, v in stats.items()
    }
    return {
        "passed": total_passed,
        "total": total_cases,
        "percentage": overall_pct,
        "overall_score": f"{total_passed}/{total_cases} ({overall_pct}%)",
        "breakdown": breakdown,
        "failed_samples": failed_samples[:8],
    }


def pretrain_language_model(
    config: ModelConfig,
    train_ds: PackedSequenceDataset,
    val_ds: PackedSequenceDataset,
    dataset_stats: dict[str, Any],
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    eval_batches: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    prompt_text: str,
    max_new_tokens: int,
    seed: int,
    device: torch.device,
    log_interval: int,
    init_checkpoint: Optional[Path] = None,
    expand_checkpoint: Optional[Path] = None,
    teacher_checkpoints: Optional[list[Path]] = None,
    distill_alpha: float = 0.35,
    amp_dtype: Optional[torch.dtype] = None,
) -> tuple[GutenbergChaoticPrimordial, dict[str, Any]]:
    """Run pretraining, checkpoint expansion, or multi-teacher consolidation."""
    set_reproducible_seed(seed)
    tokenizer = ByteLevelTokenizer()
    if init_checkpoint is not None and expand_checkpoint is not None:
        raise ValueError("Specify either --init-checkpoint or --expand-checkpoint, not both")

    if init_checkpoint is not None:
        model = load_checkpoint(init_checkpoint, device)
        if model.config != config:
            raise ValueError(
                "CLI model architecture flags do not match --init-checkpoint config "
                "(use --expand-checkpoint to widen/deepen an existing checkpoint)"
            )
    elif expand_checkpoint is not None:
        model = expand_checkpoint_to_model(expand_checkpoint, config, device)
    else:
        model = GutenbergChaoticPrimordial(config).to(device)

    teachers: list[GutenbergChaoticPrimordial] = []
    if teacher_checkpoints:
        for t_path in teacher_checkpoints:
            t_model = load_checkpoint(t_path, device)
            t_model.eval()
            for p in t_model.parameters():
                p.requires_grad_(False)
            teachers.append(t_model)

    param_groups = build_optimizer_param_groups(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    def _lr_multiplier(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx <= warmup_steps:
            return float(step_idx) / float(max(1, warmup_steps))
        progress = float(step_idx - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_multiplier)
    train_gen = torch.Generator(device="cpu").manual_seed(seed + 101)

    init_val_loss, init_val_ppl = evaluate_language_model(
        model, val_ds, batch_size, eval_batches, seed, device, amp_dtype=amp_dtype
    )

    milestones: list[dict[str, Any]] = []
    tokens_per_step = batch_size * grad_accum * train_ds.seq_len
    started = time.perf_counter()
    distill_temp = 1.5

    model.train()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            bx, by = train_ds.sample_batch(batch_size, train_gen, device)
            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with amp_ctx:
                student_logits, ce_loss = model(bx, labels=by, validate_inputs=False)
                if teachers and distill_alpha > 0.0:
                    with torch.no_grad():
                        avg_probs: Optional[torch.Tensor] = None
                        for t_model in teachers:
                            t_logits = t_model(bx, validate_inputs=False)
                            t_probs = F.softmax(t_logits.float() / distill_temp, dim=-1)
                            avg_probs = (
                                t_probs if avg_probs is None else (avg_probs + t_probs)
                            )
                        assert avg_probs is not None
                        avg_probs = avg_probs / float(len(teachers))
                    student_log_probs = F.log_softmax(
                        student_logits.float() / distill_temp, dim=-1
                    )
                    kl_loss = F.kl_div(
                        student_log_probs, avg_probs, reduction="batchmean"
                    ) * (distill_temp * distill_temp)
                    loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kl_loss
                else:
                    loss = ce_loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss encountered at step {step}")
            scaled_loss = loss / float(grad_accum)
            scaled_loss.backward()
            accum_loss += float(loss.detach().item()) / float(grad_accum)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % log_interval == 0 or step == max_steps:
            v_loss, v_ppl = evaluate_language_model(
                model, val_ds, batch_size, eval_batches, seed, device, amp_dtype=amp_dtype
            )
            elapsed = max(time.perf_counter() - started, 1e-6)
            tok_sec = (step * tokens_per_step) / elapsed
            entry = {
                "step": step,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 6),
                "train_loss": round(accum_loss, 4),
                "val_loss": round(v_loss, 4),
                "val_perplexity": round(v_ppl, 2),
                "tokens_per_sec": round(tok_sec, 1),
            }
            milestones.append(entry)
            print(
                f"[step {step:04d}/{max_steps:04d}] "
                f"train_loss={entry['train_loss']:.4f} | "
                f"val_loss={entry['val_loss']:.4f} | "
                f"val_ppl={entry['val_perplexity']:.2f} | "
                f"tok/s={entry['tokens_per_sec']:.1f}",
                file=sys.stderr,
            )

    total_elapsed = time.perf_counter() - started
    final_val_loss, final_val_ppl = evaluate_language_model(
        model, val_ds, batch_size, eval_batches, seed, device, amp_dtype=amp_dtype
    )

    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt_text, add_bos=True, add_eos=False)],
        dtype=torch.long,
        device=device,
    )
    gen_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=32,
        eos_token_id=EOS_TOKEN_ID,
    )
    generated_text = tokenizer.decode(gen_ids[0].tolist(), skip_special=True)

    report = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "init_checkpoint": str(init_checkpoint.resolve()) if init_checkpoint else None,
        "expand_checkpoint": (
            str(expand_checkpoint.resolve()) if expand_checkpoint else None
        ),
        "teacher_checkpoints": (
            [str(p.resolve()) for p in teacher_checkpoints]
            if teacher_checkpoints
            else []
        ),
        "distill_alpha": distill_alpha if teacher_checkpoints else 0.0,
        "compute_dtype": str(amp_dtype) if amp_dtype is not None else "torch.float32",
        "dataset": dataset_stats,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_step": tokens_per_step,
        "total_tokens_processed": max_steps * tokens_per_step,
        "learning_rate": lr,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "initial_val_loss": round(init_val_loss, 4),
        "initial_val_perplexity": round(init_val_ppl, 2),
        "final_val_loss": round(final_val_loss, 4),
        "final_val_perplexity": round(final_val_ppl, 2),
        "layer_memory_gates_sigmoid": model.memory_gate_summary(),
        "elapsed_seconds": round(total_elapsed, 4),
        "avg_tokens_per_sec": round(
            (max_steps * tokens_per_step) / max(total_elapsed, 1e-6), 1
        ),
        "progress": milestones,
        "sample_generation": {
            "prompt": prompt_text,
            "generated_text": generated_text,
        },
    }
    return model, report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _next_checkpoint_id(models_dir: Path, reports_dir: Path) -> str:
    max_index = 0
    for directory, suffix in ((models_dir, ".pt"), (reports_dir, ".json")):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if not entry.is_file() or entry.suffix != suffix:
                continue
            match = CHECKPOINT_NAME_PATTERN.match(entry.stem)
            if match is not None:
                max_index = max(max_index, int(match.group(1)))
    return f"Test{max_index + 1:02d}"


def find_latest_checkpoint(output_root: Path) -> Path:
    models_dir = output_root.expanduser().resolve() / "Models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    pt_files = [
        entry
        for entry in models_dir.iterdir()
        if entry.is_file() and entry.suffix == ".pt"
    ]
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint (.pt) files found in: {models_dir}")

    numbered: list[tuple[int, float, Path]] = []
    for entry in pt_files:
        match = CHECKPOINT_NAME_PATTERN.match(entry.stem)
        if match is not None:
            numbered.append((int(match.group(1)), entry.stat().st_mtime, entry))
    if numbered:
        numbered.sort(key=lambda item: (item[0], item[1]))
        return numbered[-1][2]
    pt_files.sort(key=lambda entry: entry.stat().st_mtime)
    return pt_files[-1]


def save_checkpoint_with_report(
    model: GutenbergChaoticPrimordial,
    output_root: Path,
    seed: int,
    run_metadata: dict[str, Any],
) -> tuple[Path, Path, str]:
    """Save sequential TestXX.pt and TestXX.json atomically."""
    output_root = output_root.expanduser().resolve()
    models_dir = output_root / "Models"
    reports_dir = output_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = _next_checkpoint_id(models_dir, reports_dir)
    saved_at_utc = datetime.now(timezone.utc).isoformat()
    checkpoint_path = models_dir / f"{run_id}.pt"
    report_path = reports_dir / f"{run_id}.json"
    tmp_ckpt: Optional[Path] = None

    checkpoint_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture_name": ARCHITECTURE_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "seed": int(seed),
        "saved_at_utc": saved_at_utc,
        "environment": {
            "python": sys.version,
            "pytorch": str(torch.__version__),
            "platform": platform.platform(),
        },
        "run_metadata": run_metadata,
    }

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=models_dir,
            prefix=f".{run_id}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_ckpt = Path(handle.name)
        torch.save(checkpoint_payload, tmp_ckpt)
        with tmp_ckpt.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint_path}")
        os.replace(tmp_ckpt, checkpoint_path)
        tmp_ckpt = None
    except Exception:
        if tmp_ckpt is not None:
            tmp_ckpt.unlink(missing_ok=True)
        raise

    checkpoint_hash = sha256_file(checkpoint_path)
    report_payload = {
        "report_schema_version": 3,
        "run_id": run_id,
        "saved_at_utc": saved_at_utc,
        "architecture_name": ARCHITECTURE_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_config": asdict(model.config),
        "seed": int(seed),
        "environment": checkpoint_payload["environment"],
        "run_metadata": run_metadata,
        "checkpoint": {
            "path": checkpoint_path.relative_to(output_root).as_posix(),
            "sha256": checkpoint_hash,
            "bytes": checkpoint_path.stat().st_size,
            "saved": True,
        },
        "report_status": "complete",
    }
    _atomic_write_json(report_path, report_payload)
    return checkpoint_path, report_path, checkpoint_hash


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> GutenbergChaoticPrimordial:
    """Load a v0.3.x checkpoint in evaluation mode after schema/config validation."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)

    required = {
        "checkpoint_schema_version",
        "architecture_name",
        "architecture_version",
        "model_config",
        "model_state_dict",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    if payload["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    if payload["architecture_name"] != ARCHITECTURE_NAME:
        raise ValueError("Checkpoint architecture name does not match")
    ckpt_version = str(payload["architecture_version"])
    if not ckpt_version.startswith(COMPATIBLE_VERSION_PREFIX):
        raise ValueError(
            f"Checkpoint version {ckpt_version!r} is incompatible with {ARCHITECTURE_VERSION!r}"
        )

    config = ModelConfig.from_dict(payload["model_config"])
    model = GutenbergChaoticPrimordial(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def run_interactive_chat(
    model: GutenbergChaoticPrimordial,
    ckpt_path: Path,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> None:
    """Interactive PowerShell REPL for live math and English testing."""
    tokenizer = ByteLevelTokenizer()
    print(f"Loaded {ckpt_path.name} on {device}. Type a prompt and press Enter ('exit' to quit).")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt, add_bos=True, add_eos=False)],
            dtype=torch.long,
            device=device,
        )
        started = time.perf_counter()
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=EOS_TOKEN_ID,
        )
        elapsed = max(time.perf_counter() - started, 1e-6)
        new_tokens = int(out_ids.shape[1] - prompt_ids.shape[1])
        text = tokenizer.decode(out_ids[0].tolist(), skip_special=True)
        print(f"Output ({new_tokens} tok, {new_tokens / elapsed:.1f} tok/s):\n{text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{ARCHITECTURE_NAME} v{ARCHITECTURE_VERSION} Pretraining, Expansion & Inference Engine"
    )
    parser.add_argument(
        "--mode",
        choices=("pretrain", "generate", "chat", "eval-math", "eval-higher-math", "eval-all-math", "self-test"),
        default="pretrain",
        help="execution mode: pretrain, generate, chat, eval-math, eval-higher-math, eval-all-math, or self-test",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional path to external dataset (.jsonl/.json/.txt/.parquet/.arrow); omit to use fast built-in math curriculum",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=25000,
        help="Reservoir sample cap for external dataset records (0 = load all)",
    )
    parser.add_argument(
        "--math-replay-ratio",
        type=float,
        default=0.20,
        help="Fraction of Stage-1/2 math interleaved into --data-path language training",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Exact-shape checkpoint (.pt) to continue pretraining from",
    )
    parser.add_argument(
        "--expand-checkpoint",
        type=Path,
        default=None,
        help="Smaller v0.3.x checkpoint (.pt) to expand (Net2Net width+depth) into a larger target architecture",
    )
    parser.add_argument(
        "--teacher-checkpoints",
        type=str,
        default=None,
        help="Comma-separated teacher checkpoint paths for multi-teacher distillation (e.g. Models\\Test25.pt,Models\\Test27.pt)",
    )
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=0.35,
        help="Weight of teacher KL distillation loss when --teacher-checkpoints is set (default 0.35)",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--conv-kernel", type=int, default=3)
    parser.add_argument("--ssm-chunk-size", type=int, default=64)
    parser.add_argument("--ffn-expansion", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU threads (0 = auto-detect physical cores for portability)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "auto"), default="float32")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        active_threads = resolve_threads(args.threads)
        torch.set_num_threads(active_threads)
        device = resolve_device(args.device)
        amp_dtype = resolve_amp_dtype(args.dtype, device)

        if args.mode == "self-test":
            report = run_self_tests(device)
            report["threads"] = active_threads
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report.get("overall_status") == "PASS" else 1

        if args.mode == "eval-math":
            ckpt_path = (
                find_latest_checkpoint(args.output_root)
                if args.checkpoint is None
                else args.checkpoint
            )
            model = load_checkpoint(ckpt_path, device)
            eval_report = evaluate_math_proficiency(model, device=device, seed=args.seed)
            print(
                json.dumps(
                    {
                        "architecture_version": ARCHITECTURE_VERSION,
                        "mode": "eval-math",
                        "device": str(device),
                        "threads": active_threads,
                        "checkpoint": str(ckpt_path.resolve()),
                        "results": eval_report,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.mode == "eval-higher-math":
            ckpt_path = (
                find_latest_checkpoint(args.output_root)
                if args.checkpoint is None
                else args.checkpoint
            )
            model = load_checkpoint(ckpt_path, device)
            eval_report = evaluate_higher_math_proficiency(model, device=device)
            print(
                json.dumps(
                    {
                        "architecture_version": ARCHITECTURE_VERSION,
                        "mode": "eval-higher-math",
                        "device": str(device),
                        "threads": active_threads,
                        "checkpoint": str(ckpt_path.resolve()),
                        "results": eval_report,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.mode == "eval-all-math":
            ckpt_path = (
                find_latest_checkpoint(args.output_root)
                if args.checkpoint is None
                else args.checkpoint
            )
            model = load_checkpoint(ckpt_path, device)
            f_res = evaluate_math_proficiency(model, device=device, seed=args.seed)
            h_res = evaluate_higher_math_proficiency(model, device=device)
            tot_passed = int(f_res["passed"]) + int(h_res["passed"])
            tot_cases = int(f_res["total"]) + int(h_res["total"])
            overall_pct = round(100.0 * tot_passed / max(1, tot_cases), 1)
            print(
                json.dumps(
                    {
                        "architecture_version": ARCHITECTURE_VERSION,
                        "mode": "eval-all-math",
                        "device": str(device),
                        "threads": active_threads,
                        "checkpoint": str(ckpt_path.resolve()),
                        "results": {
                            "combined_overall_score": f"{tot_passed}/{tot_cases} ({overall_pct}%)",
                            "foundation_score": f_res["overall_score"],
                            "foundation_100_pct": f_res["foundation_100_pct"],
                            "higher_math_score": h_res["overall_score"],
                            "foundation_breakdown": f_res["breakdown"],
                            "higher_math_breakdown": h_res["breakdown"],
                            "failed_samples": (
                                f_res["failed_samples"][:4] + h_res["failed_samples"][:4]
                            ),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.mode == "chat":
            ckpt_path = (
                find_latest_checkpoint(args.output_root)
                if args.checkpoint is None
                else args.checkpoint
            )
            model = load_checkpoint(ckpt_path, device)
            run_interactive_chat(
                model=model,
                ckpt_path=ckpt_path,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            return 0

        if args.mode == "generate":
            ckpt_path = (
                find_latest_checkpoint(args.output_root)
                if args.checkpoint is None
                else args.checkpoint
            )
            model = load_checkpoint(ckpt_path, device)
            tokenizer = ByteLevelTokenizer()
            prompt_ids = torch.tensor(
                [tokenizer.encode(args.prompt, add_bos=True, add_eos=False)],
                dtype=torch.long,
                device=device,
            )
            started = time.perf_counter()
            out_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=EOS_TOKEN_ID,
            )
            elapsed = max(time.perf_counter() - started, 1e-6)
            new_tokens = int(out_ids.shape[1] - prompt_ids.shape[1])
            generated_text = tokenizer.decode(out_ids[0].tolist(), skip_special=True)
            print(
                json.dumps(
                    {
                        "architecture_version": ARCHITECTURE_VERSION,
                        "mode": "generate",
                        "device": str(device),
                        "threads": active_threads,
                        "checkpoint": str(ckpt_path.resolve()),
                        "prompt": args.prompt,
                        "generated_text": generated_text,
                        "new_tokens": new_tokens,
                        "tokens_per_sec": round(new_tokens / elapsed, 1),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        self_test_summary = run_self_tests(device)
        tokenizer = ByteLevelTokenizer()

        if args.data_path is not None:
            train_ds, val_ds, dataset_stats = load_jsonl_datasets(
                data_path=args.data_path,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                val_ratio=args.val_ratio,
                seed=args.seed,
                max_docs=args.max_docs,
                math_replay_ratio=args.math_replay_ratio,
            )
            dataset_stats["source"] = str(args.data_path.expanduser().resolve())
        else:
            train_texts, val_texts = _builtin_stage2_math_splits(seed=args.seed)
            train_docs = [
                tokenizer.encode(txt, add_bos=True, add_eos=True) for txt in train_texts
            ]
            val_docs = [
                tokenizer.encode(txt, add_bos=True, add_eos=True) for txt in val_texts
            ]
            train_ds = _pack_isolated_documents(train_docs, seq_len=args.seq_len)
            val_ds = _pack_isolated_documents(val_docs, seq_len=args.seq_len)
            dataset_stats = {
                "source": "builtin_stage2_math_curriculum",
                "train_documents": len(train_docs),
                "val_documents": len(val_docs),
                "train_blocks": len(train_ds),
                "val_blocks": len(val_ds),
                "seq_len": args.seq_len,
            }

        teacher_paths: Optional[list[Path]] = None
        if args.teacher_checkpoints:
            teacher_paths = [
                Path(item.strip())
                for item in args.teacher_checkpoints.split(",")
                if item.strip()
            ]

        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_layers=args.layers,
            state_size=args.state_size,
            memory_heads=args.heads,
            conv_kernel_size=args.conv_kernel,
            ssm_chunk_size=args.ssm_chunk_size,
            ffn_expansion=args.ffn_expansion,
            dropout=args.dropout,
            use_associative_memory=True,
        )

        model, train_metrics = pretrain_language_model(
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            dataset_stats=dataset_stats,
            max_steps=args.steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_batches=args.eval_batches,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            device=device,
            log_interval=args.log_interval,
            init_checkpoint=args.init_checkpoint,
            expand_checkpoint=args.expand_checkpoint,
            teacher_checkpoints=teacher_paths,
            distill_alpha=args.distill_alpha,
            amp_dtype=amp_dtype,
        )

        output: dict[str, Any] = {
            "architecture_version": ARCHITECTURE_VERSION,
            "device": str(device),
            "threads": active_threads,
            "mode": "pretrain",
            "self_tests_status": self_test_summary["overall_status"],
            "status": "PASS",
            "results": train_metrics,
        }

        if not args.no_save:
            ckpt_path, report_path, ckpt_hash = save_checkpoint_with_report(
                model=model,
                output_root=args.output_root,
                seed=args.seed,
                run_metadata=output,
            )
            output["checkpoint_path"] = str(ckpt_path)
            output["report_path"] = str(report_path)
            output["checkpoint_sha256"] = ckpt_hash

        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
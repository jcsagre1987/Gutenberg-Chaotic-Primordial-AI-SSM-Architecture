# Gutenberg Chaotic Primordial (`v0.3.0` / `v0.3.1`)

**Gutenberg Chaotic Primordial** is a standalone, single-file PyTorch implementation of a hybrid **Selective Diagonal State-Space Model (SSM) + Dual-Stream Delta-Rule Associative Memory** language model. Designed for transparent research, reproducible CPU/GPU training, and $O(1)$-per-token stateful streaming inference, it requires zero external tokenizer files and zero custom CUDA kernels.

* **Author:** Juan Carlo Sagre
* **License:** Apache-2.0
* **Available Engine Scripts:**
  * `gutenberg_chaotic_primordial_v0_3_0.py` — **Core Architecture Engine:** Complete hybrid SSM + Associative Memory model, from-scratch pretraining, Net2Net checkpoint expansion, multi-teacher distillation, math benchmarks, and $O(1)$ streaming inference.
  * `gutenberg_chaotic_primordial_v0_3_1.py` — **Core Engine + Native LoRA Adapters:** A full backward-compatible superset of `v0.3.0` that adds frozen-base Low-Rank Adaptation (`--lora-rank`), modular `Adapters\AdapterXX.pt` serialization, hot-swappable inference/evaluation (`--lora-adapter`), and lossless weight merging (`--mode merge-lora`). Both scripts share the exact `0.3.x` checkpoint format.

## Table of Contents

1. [Architecture Highlights](#1-architecture-highlights)
2. [Windows PowerShell Setup & Quick Verification](#2-windows-powershell-setup--quick-verification)
3. [Workspace Layout & Automatic Checkpoint Management](#3-workspace-layout--automatic-checkpoint-management)
4. [Using Your Own Datasets & Directories](#4-using-your-own-datasets--directories)
5. [PowerShell Training, Resuming, Expanding & Distillation Workflows (`v0.3.0` & `v0.3.1`)](#5-powershell-training-resuming-expanding--distillation-workflows-v030--v031)
6. [LoRA Adapter Fine-Tuning, Swapping & Lossless Merging (`v0.3.1`)](#6-lora-adapter-fine-tuning-swapping--lossless-merging-v031)
7. [Running Math & Reasoning Benchmarks in PowerShell](#7-running-math--reasoning-benchmarks-in-powershell)
8. [Text Generation & Interactive Terminal Chat in PowerShell](#8-text-generation--interactive-terminal-chat-in-powershell)
9. [Model Scaling Presets & Full CLI Reference](#9-model-scaling-presets--full-cli-reference)
10. [PowerShell Tips & Troubleshooting](#10-powershell-tips--troubleshooting)

## 1. Architecture Highlights

* **Lossless UTF-8 Byte-Level Tokenizer (`vocab_size = 260`):** Encodes any UTF-8 string directly into raw bytes (`4..259`) plus four reserved control tokens (`PAD=0`, `BOS=1`, `EOS=2`, `DELIM=3`). There are no out-of-vocabulary (`<UNK>`) tokens and no external vocabulary files to manage.
* **Selective Diagonal SSM (`SelectiveDiagonalSSM`):** Combines a causal depthwise 1D convolution (`conv_kernel_size = 3`), input-dependent step sizes ($\Delta_t$), multi-timescale log-spaced decay initialization, exact Zero-Order Hold (ZOH) discretization, and an $O(\log C)$ parallel prefix scan (`ssm_chunk_size = 64`), followed by a pre-normalized SwiGLU feed-forward block.
* **Dual-Stream Delta-Rule Associative Memory (`DeltaRuleAssociativeMemory`):** Maintains fast-weight associative matrices ($d_{\text{head}} \times d_{\text{head}}$ per head) with a 1-step key shift, learnable bigram phrase mixing, $L_2$-normalized queries and keys, gated delta-rule updates ($\beta_t, \gamma_t$), chunkwise triangular-solve parallel scanning, and a learnable sigmoid residual gate per layer.
* **Exact $O(1)$ Streaming Inference:** Full-sequence parallel training and token-by-token autoregressive generation are mathematically equivalent, keeping inference memory constant regardless of sequence length.
* **Modular Low-Rank Adaptation (`v0.3.1`):** Injects trainable low-rank matrices ($W + \frac{\alpha}{r} B A$) across targeted linear projections while keeping base checkpoint weights frozen, allowing rapid domain specialization without catastrophic forgetting.

## 2. Windows PowerShell Setup & Quick Verification

### Requirements

* **OS & Shell:** Windows 10 / 11 with **Windows PowerShell 5.1** or **PowerShell 7+**
* **Python:** `3.10` or newer (`3.11` recommended)
* **Required Package:** `torch` (CPU or CUDA build)
* **Optional Packages:** `pyarrow` or `datasets` (only needed if you plan to train directly on `.parquet` or Hugging Face `.arrow` dataset files; `.jsonl`, `.json`, `.txt`, and `.md` work out of the box with standard Python)

Open **Windows PowerShell**, navigate to the folder containing the engine scripts, and install dependencies:

```powershell
cd "C:\Path\To\Gutenberg Chaotic Primordial"
python -m pip install torch

# Optional: enable .parquet and Hugging Face .arrow dataset loading
python -m pip install pyarrow
```

### Verify Your Environment

Before starting a training run, execute the built-in self-test suite in PowerShell to verify chunkwise SSM continuation, associative memory state equivalence, $O(1)$ streaming parity, and finite gradients:

```powershell
# Verify v0.3.0 core architecture engine
python gutenberg_chaotic_primordial_v0_3_0.py --mode self-test

# Verify v0.3.1 engine (includes LoRA gradient isolation & lossless merge parity tests)
python gutenberg_chaotic_primordial_v0_3_1.py --mode self-test
```

A healthy installation outputs `"overall_status": "PASS"`.

## 3. Workspace Layout & Automatic Checkpoint Management

By default, the scripts create and manage `Models\`, `reports\`, and (in `v0.3.1`) `Adapters\` inside the directory where the script lives. If you keep your code and experiment outputs in separate folders, pass `--output-root "D:\MyWorkspace"`:

```text
<output-root>\
├── gutenberg_chaotic_primordial_v0_3_0.py
├── gutenberg_chaotic_primordial_v0_3_1.py
├── Models\
│   ├── Test01.pt          # Sequential atomic base/merged model checkpoints
│   └── Test02.pt
├── Adapters\              # Created by v0.3.1 when training with --lora-rank > 0
│   ├── Adapter01.pt       # Compact sequential LoRA adapter weights
│   └── Adapter02.pt
└── reports\
    ├── Test01.json        # Reproducibility metadata, metrics & SHA-256 hashes
    ├── Test02.json
    └── Adapter01.json     # LoRA adapter training metrics & base checkpoint link
```

* **Sequential, Non-Destructive Saves:** Every completed training run writes the next available `TestXX.pt` (or `AdapterXX.pt` in `v0.3.1`) and corresponding `.json` report via atomic file replacement (`fsync` + rename) so existing checkpoints and adapters are never overwritten.
* **Automatic Latest-Checkpoint Discovery:** In inference or evaluation modes (`generate`, `chat`, `eval-math`, `eval-higher-math`, `eval-all-math`), omitting `--checkpoint` automatically selects the highest-numbered `TestXX.pt` inside `<output-root>\Models\`.
* **Cross-Version Checkpoint Compatibility:** Checkpoints created by `v0.3.0` load directly into `v0.3.1` (and merged checkpoints exported by `v0.3.1` load cleanly back into `v0.3.0`).

## 4. Using Your Own Datasets & Directories

You do not need to move your datasets into the repository folder or convert them into a rigid format. In PowerShell, point `--data-path` to **any single file** or **any root directory** on your drive (always wrap paths containing spaces in double quotes `"..."`). The loader recursively scans subdirectories (`rglob`) and automatically skips Hugging Face metadata files (`dataset_info.json`, `state.json`).

### Supported File Formats

| Extension | How It Is Parsed | Extra Dependency |
| :--- | :--- | :--- |
| `.jsonl`, `.ndjson` | Line-delimited JSON objects (malformed lines are safely skipped) | None |
| `.json` | Top-level JSON array `[...]`, single object, or dict containing `"data"`, `"rows"`, `"records"`, `"examples"`, or `"train"` | None |
| `.txt`, `.md` | Split into documents on blank lines (`\n\n`), or line-by-line if single-spaced | None |
| `.parquet` | Apache Parquet tables read row-by-row | `pyarrow` |
| `.arrow` | Hugging Face Arrow IPC stream or file shards (point directly at a cached HF dataset folder) | `pyarrow` or `datasets` |

### Supported Record Schemas (Automatic Detection)

When reading `.jsonl`, `.json`, `.parquet`, or `.arrow` records, the loader automatically inspects each dictionary and formats it into clean training text:

1. **Multi-Turn Chat (`messages`):**
   ```json
   {"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there."}]}
   ```
   *Formatted as:* `user: Hello!\nassistant: Hi there.`

2. **Instruction Tuning (`instruction` + `output` or `response`, optional `input`):**
   ```json
   {"instruction": "Summarize the text.", "input": "Water boils at 100C.", "output": "Water boils at 100 degrees Celsius."}
   ```
   *Formatted as:* `Instruction: ...\nInput: ...\nResponse: ...`

3. **Question & Answer (`question` + `answer`):**
   ```json
   {"question": "What planet is closest to the Sun?", "answer": "Mercury."}
   ```
   *Formatted as:* `Question: ...\nAnswer: ...`

4. **Vocabulary / Dictionary (`word` + `definition` or `meaning`, optional `example`):**
   ```json
   {"word": "ephemeral", "definition": "lasting for a very short time", "example": "ephemeral spring blossoms"}
   ```
   *Formatted as:* `Word: ephemeral | Meaning: lasting for a very short time | Example: ephemeral spring blossoms`

5. **Plain Prose / Stories (`text`, `story`, `content`, `document`, `body`, or `passage`):**
   ```json
   {"story": "Once upon a time, a small clockwork bird learned to sing."}
   ```
   *Extracted directly as raw text.*

6. **Prompt / Completion (`prompt` + `response` or `completion`) & `input` / `output`:**
   Automatically normalized into `User:` / `Assistant:` or `Input:` / `Output:` pairs.

### Memory-Safe Streaming, Document Isolation & Math Replay

* **Constant-RAM Reservoir Sampling (`--max-docs`):** Large datasets (hundreds of thousands of documents) are streamed using uniform reservoir sampling (`--max-docs 25000` by default) so RAM usage stays bounded. Set `--max-docs 0` to load every record in the directory.
* **Cross-Document State Isolation:** Each document is encoded as `[BOS] ... [EOS]` followed by `4` `PAD` tokens (`PAD_TOKEN_ID = 0`, ignored by cross-entropy loss) before packing into `[N, seq_len + 1]` blocks, preventing recurrent SSM and associative memory states from bleeding across unrelated documents.
* **Catastrophic Forgetting Prevention (`--math-replay-ratio`):** When training on external prose or instruction datasets via `--data-path`, the loader interleaves built-in Stage-1 and Stage-2 math traces (`--math-replay-ratio 0.20` = 20% math by default) to preserve arithmetic and symbolic reasoning. Set `--math-replay-ratio 0.0` if you want pure language training.

## 5. PowerShell Training, Resuming, Expanding & Distillation Workflows (`v0.3.0` & `v0.3.1`)

> **Script Choice Note:** All commands in this section use `gutenberg_chaotic_primordial_v0_3_0.py` and work identically if you substitute `gutenberg_chaotic_primordial_v0_3_1.py`.
>
> **PowerShell Syntax Note:** In Windows PowerShell, multi-line commands use the **backtick character (`` ` ``)** at the end of a line (not `\`), and directory paths use Windows backslashes (`Models\Test01.pt`).

### A. Train from Scratch on the Built-In Math Curriculum

Omitting `--data-path`, `--init-checkpoint`, and `--expand-checkpoint` initializes a fresh model and trains on the built-in Stage-1 & Stage-2 mathematics curriculum (8-bit binary conversion, carry addition, borrow subtraction, multiplication, pre-algebra, algebra, geometry, trigonometry, and calculus):

```powershell
# Default Nano model (4.88M parameters: d_model=256, layers=4, ffn_expansion=2.0) -> saves Models\Test01.pt
python gutenberg_chaotic_primordial_v0_3_0.py --steps 1200 --lr 4e-4

# Custom Small model (10.06M parameters: d_model=320, layers=5, ffn_expansion=2.2) -> saves Models\Test02.pt
python gutenberg_chaotic_primordial_v0_3_0.py --d-model 320 --layers 5 --ffn-expansion 2.2 --steps 1200 --lr 4e-4
```

### B. Train on Your Own External Dataset Directory

Point `--data-path` to any folder or file on your machine (single-line or multi-line with PowerShell backticks):

```powershell
# Single-line PowerShell command
python gutenberg_chaotic_primordial_v0_3_0.py --data-path "C:\Training Datasets\Tinystories" --max-docs 25000 --math-replay-ratio 0.20 --steps 1500 --lr 4e-4

# Multi-line PowerShell command (using backtick ` line continuation)
python gutenberg_chaotic_primordial_v0_3_0.py `
  --data-path "D:\Datasets\Alpaca" `
  --d-model 320 `
  --layers 5 `
  --ffn-expansion 2.2 `
  --max-docs 25000 `
  --math-replay-ratio 0.20 `
  --steps 1500 `
  --lr 4e-4
```

### C. Continue Training an Existing Checkpoint (`--init-checkpoint`)

Resume training from a saved `v0.3.x` checkpoint at the **same architecture dimensions**. Check `reports\TestXX.json` first if you are unsure which dimensions a checkpoint was created with:

```powershell
# Inspect a checkpoint's saved architecture dimensions in PowerShell
Get-Content "reports\Test01.json" | ConvertFrom-Json | Select-Object -ExpandProperty model_config

# 1) Resume a default Nano checkpoint (4.88M: d_model=256, layers=4, ffn_expansion=2.0)
python gutenberg_chaotic_primordial_v0_3_0.py --init-checkpoint "Models\Test01.pt" --seed 2026 --steps 500 --lr 8e-5

# 2) Resume a custom Small checkpoint (10.06M: d_model=320, layers=5, ffn_expansion=2.2)
python gutenberg_chaotic_primordial_v0_3_0.py --init-checkpoint "Models\Test02.pt" --d-model 320 --layers 5 --ffn-expansion 2.2 --seed 2026 --steps 500 --lr 8e-5
```

> **Important:** Full-model `--init-checkpoint` requires `--d-model`, `--layers`, `--heads`, and `--ffn-expansion` to match the source checkpoint exactly. To grow a smaller checkpoint into larger dimensions, use `--expand-checkpoint` below.

### D. Function-Preserving Model Expansion (`--expand-checkpoint`)

Scale a smaller checkpoint into a wider and/or deeper architecture using Net2Net-style weight preservation. Existing channels retain their learned weights (with LayerNorm variance compensation and near-identity initialization for newly added layers), while extra channels receive small symmetry-breaking noise:

```powershell
# 1) Expand default Nano (4.88M: d_model=256, layers=4, ffn=2.0) -> Small (10.06M: d_model=320, layers=5, ffn=2.2)
python gutenberg_chaotic_primordial_v0_3_0.py --expand-checkpoint "Models\Test01.pt" --d-model 320 --layers 5 --ffn-expansion 2.2 --steps 300 --lr 8e-5

# 2) Expand Small (10.06M: d_model=320, layers=5, ffn=2.2) -> Medium (15.08M: d_model=384, layers=5, ffn=2.35)
python gutenberg_chaotic_primordial_v0_3_0.py --expand-checkpoint "Models\Test02.pt" --d-model 384 --layers 5 --ffn-expansion 2.35 --steps 300 --lr 8e-5
```

**Expansion Compatibility Rules:**

* Target `--d-model` $\ge$ source `d_model`, target `--layers` $\ge$ source `num_layers`, and target `--state-size` $\ge$ source `state_size`.
* Target `--heads` must equal the source `memory_heads`, and `--d-model` must be divisible by `--heads`.

### E. Multi-Teacher Knowledge Distillation (`--teacher-checkpoints`)

Anchor a student checkpoint against one or more frozen teacher checkpoints via temperature-scaled KL-divergence distillation ($T = 1.5$) blended with cross-entropy loss (`--distill-alpha`):

```powershell
python gutenberg_chaotic_primordial_v0_3_0.py --init-checkpoint "Models\Test02.pt" --teacher-checkpoints "Models\Test01.pt,Models\Test02.pt" --distill-alpha 0.25 --d-model 320 --layers 5 --ffn-expansion 2.2 --seed 2026 --steps 450 --lr 8e-5
```

## 6. LoRA Adapter Fine-Tuning, Swapping & Lossless Merging (`v0.3.1`)

When using **`gutenberg_chaotic_primordial_v0_3_1.py`**, you can freeze any trained checkpoint (`--init-checkpoint`) and train lightweight Low-Rank Adaptation (`LoRA`) adapters without modifying the base weights. Note that in `v0.3.1`, LoRA automatically inherits the architecture dimensions stored inside `--init-checkpoint` unless you explicitly override them.

### A. Train a New LoRA Adapter (`--lora-rank`)

Setting `--lora-rank > 0` freezes all base parameters in `--init-checkpoint` and saves only the compact low-rank weights to `Adapters\Adapter01.pt` and `reports\Adapter01.json`:

```powershell
# Train a rank-16 LoRA adapter on external data while keeping Models\Test01.pt frozen
python gutenberg_chaotic_primordial_v0_3_1.py `
  --init-checkpoint "Models\Test01.pt" `
  --lora-rank 16 `
  --lora-alpha 32 `
  --data-path "C:\Training Datasets\Tinystories" `
  --max-docs 1200 `
  --math-replay-ratio 0.10 `
  --steps 900 `
  --lr 3e-4 `
  --threads 2
```

* `--lora-targets` controls which projections inside `model.layers` receive adapters: `all` (default), `ffn` (SwiGLU `gate_proj`, `up_proj`, `down_proj`), `ssm` (SSM & associative memory projections), or a comma-separated list of module names.

### B. Continue Training an Existing LoRA Adapter (`--lora-adapter`)

To run a second training pass starting from an existing adapter, pass both `--lora-rank` and `--lora-adapter` (the output saves sequentially to the next `Adapters\AdapterXX.pt`):

```powershell
python gutenberg_chaotic_primordial_v0_3_1.py `
  --init-checkpoint "Models\Test01.pt" `
  --lora-rank 16 `
  --lora-adapter "Adapters\Adapter01.pt" `
  --data-path "C:\Training Datasets\Alpaca" `
  --max-docs 800 `
  --math-replay-ratio 0.10 `
  --seed 2028 `
  --steps 900 `
  --lr 1.5e-4 `
  --threads 2
```

### C. Hot-Swap Adapters During Evaluation, Generation & Chat

Attach any saved adapter to its matching base checkpoint at runtime using `--lora-adapter`:

```powershell
# Benchmark math retention with Adapter01.pt attached
python gutenberg_chaotic_primordial_v0_3_1.py --mode eval-all-math --checkpoint "Models\Test01.pt" --lora-adapter "Adapters\Adapter01.pt"

# Generate text with Adapter01.pt attached
python gutenberg_chaotic_primordial_v0_3_1.py --mode generate --checkpoint "Models\Test01.pt" --lora-adapter "Adapters\Adapter01.pt" --prompt "Once upon a time" --temperature 0.65 --top-k 24

# Launch interactive chat with Adapter01.pt attached
python gutenberg_chaotic_primordial_v0_3_1.py --mode chat --checkpoint "Models\Test01.pt" --lora-adapter "Adapters\Adapter01.pt" --temperature 0.65 --top-k 24
```

### D. Lossless LoRA Weight Merging (`--mode merge-lora`)

Fold a trained adapter ($W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} B A$) directly back into the base linear layers to produce a standalone `Models\TestXX.pt` checkpoint with zero runtime LoRA overhead (compatible with both `v0.3.0` and `v0.3.1`):

```powershell
# Merge an existing adapter into a new standalone Models\TestXX.pt checkpoint
python gutenberg_chaotic_primordial_v0_3_1.py --mode merge-lora --checkpoint "Models\Test01.pt" --lora-adapter "Adapters\Adapter01.pt"
```

*(You can also pass `--merge-lora` during LoRA training to save both `Adapters\AdapterXX.pt` and a merged `Models\TestXX.pt` in one command.)*

## 7. Running Math & Reasoning Benchmarks in PowerShell

All evaluation modes run deterministic greedy decoding (`temperature = 0.0`) on unseen problem instances and output a structured JSON scorecard in PowerShell (works with both `v0.3.0` and `v0.3.1`):

```powershell
# Combined 65-problem suite (40 Foundation + 25 Higher Math) — Recommended
python gutenberg_chaotic_primordial_v0_3_0.py --mode eval-all-math --checkpoint "Models\Test01.pt"

# 40 unseen Foundation Math problems (10 add, 10 sub, 10 mul, 10 base-10 to base-2)
python gutenberg_chaotic_primordial_v0_3_0.py --mode eval-math --checkpoint "Models\Test01.pt"

# 25 unseen Higher Math problems (5 pre-algebra, 5 algebra, 5 geometry, 5 trig, 5 calculus)
python gutenberg_chaotic_primordial_v0_3_0.py --mode eval-higher-math --checkpoint "Models\Test01.pt"

# Omit --checkpoint to automatically evaluate the newest TestXX.pt in Models\
python gutenberg_chaotic_primordial_v0_3_0.py --mode eval-all-math
```

## 8. Text Generation & Interactive Terminal Chat in PowerShell

### Single-Prompt Generation (`--mode generate`)

Use `--temperature 0.0` for deterministic step-by-step math scratchpads, or `--temperature 0.5`–`0.7` with `--top-k 24`–`32` for creative prose:

```powershell
# Deterministic algebra evaluation
python gutenberg_chaotic_primordial_v0_3_0.py --mode generate --checkpoint "Models\Test01.pt" --prompt "Alg: Solve 4x + 7 = 31 ->" --temperature 0.0

# Story continuation
python gutenberg_chaotic_primordial_v0_3_0.py --mode generate --checkpoint "Models\Test02.pt" --prompt "Once upon a time" --temperature 0.6 --top-k 32 --max-new-tokens 160

# Instruction prompt using PowerShell newline escape (`n)
python gutenberg_chaotic_primordial_v0_3_0.py --mode generate --checkpoint "Models\Test02.pt" --prompt "Instruction: Explain why the sky looks blue.`nResponse:" --temperature 0.6 --top-k 24
```

### Interactive Terminal REPL (`--mode chat`)

Launches a live interactive prompt loop inside PowerShell using $O(1)$ stateful streaming (type `exit` or `quit` to exit):

```powershell
python gutenberg_chaotic_primordial_v0_3_0.py --mode chat --checkpoint "Models\Test01.pt" --temperature 0.0
```

## 9. Model Scaling Presets & Full CLI Reference

### Recommended Model Presets

| Preset | `--d-model` | `--layers` | `--heads` | `--ffn-expansion` | Total Base Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nano** (Default) | `256` | `4` | `8` | `2.0` | **4,875,972** (~4.88M) |
| **Small** | `320` | `5` | `8` | `2.2` | **10,055,285** (~10.06M) |
| **Medium** | `384` | `5` | `8` | `2.35` | **15,084,533** (~15.08M) |
| **Large** | `448` | `5` | `8` | `2.25` | **19,883,381** (~19.88M) |

### Core Command-Line Arguments (`v0.3.0` & `v0.3.1`)

#### Execution & Workspace

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--mode` | `pretrain` | `v0.3.0`: `pretrain`, `generate`, `chat`, `eval-math`, `eval-higher-math`, `eval-all-math`, `self-test` (`v0.3.1` also adds `merge-lora`) |
| `--output-root` | Script dir | Root directory where `Models\`, `reports\`, and `Adapters\` are read and saved |
| `--checkpoint` | Latest | Checkpoint `.pt` path for `generate`, `chat`, `eval-*`, and `merge-lora` modes |
| `--no-save` | `False` | Run `pretrain` or `merge-lora` without writing `.pt` or `.json` artifacts to disk |

#### Datasets & Curriculum

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--data-path` | `None` | File or directory path (`.jsonl`, `.json`, `.txt`, `.md`, `.parquet`, `.arrow`); omit for built-in math |
| `--max-docs` | `25000` | Reservoir sample cap for external records (`0` loads all valid records) |
| `--math-replay-ratio` | `0.20` | Fraction of Stage-1/2 math interleaved into external `--data-path` training (`0.0` to disable) |
| `--val-ratio` | `0.05` | Fraction of sampled documents held out for validation loss/perplexity |
| `--seq-len` | `128` | Context block length in byte tokens |

#### Checkpoints, Expansion & Distillation

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--init-checkpoint` | `None` | Existing `v0.3.x` checkpoint `.pt` to resume training from (or freeze as base when `--lora-rank > 0` in `v0.3.1`) |
| `--expand-checkpoint` | `None` | Smaller `v0.3.x` checkpoint `.pt` to expand into larger `--d-model` / `--layers` |
| `--teacher-checkpoints` | `None` | Comma-separated teacher checkpoint paths for KL-divergence distillation |
| `--distill-alpha` | `0.35` | Weight of teacher KL distillation loss when `--teacher-checkpoints` is active |

#### Model Architecture

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--d-model` | `256` | Hidden channel dimension (must be divisible by `--heads`) |
| `--layers` | `4` | Number of interleaved SSM + Associative Memory hybrid layers |
| `--state-size` | `16` | Diagonal SSM state slots per channel |
| `--heads` | `8` | Number of associative memory heads |
| `--conv-kernel` | `3` | Causal depthwise 1D convolution kernel size |
| `--ssm-chunk-size` | `64` | Parallel scan chunk size for the selective SSM |
| `--ffn-expansion` | `2.0` | SwiGLU feed-forward hidden expansion multiplier |
| `--dropout` | `0.05` | Residual dropout probability (`[0, 1)`) |

#### Optimization & Reproducibility

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--steps` | `1200` | Total optimizer steps |
| `--batch-size` | `8` | Sequences per micro-batch |
| `--grad-accum` | `1` | Gradient accumulation steps (`tokens_per_step = batch_size * grad_accum * seq_len`) |
| `--lr` | `4e-4` | Peak AdamW learning rate with linear warmup and cosine decay to `0.1 * lr` |
| `--warmup-steps` | `40` | Linear learning-rate warmup steps |
| `--weight-decay` | `0.1` | AdamW weight decay (automatically disabled for 1D biases, norms, decays, and gates) |
| `--eval-batches` | `4` | Validation batches evaluated per logging interval |
| `--log-interval` | `50` | Progress logging interval in steps |
| `--seed` | `1701` | Deterministic RNG seed for data sampling, initialization, and evaluation |

#### Generation & Hardware

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--prompt` | `"Once upon a time"` | Input prompt for `--mode generate` and post-training sample generation |
| `--max-new-tokens` | `160` | Maximum tokens to generate |
| `--temperature` | `0.0` | Sampling temperature (`0.0` = greedy argmax) |
| `--top-k` | `32` | Top-k token filtering when `--temperature > 0` |
| `--threads` | `0` | CPU intra-op threads (`0` = auto-detect physical CPU cores; use `--threads 2` for low-resource background runs) |
| `--device` | `auto` | Compute device: `auto`, `cpu`, or `cuda` |
| `--dtype` | `float32` | Compute precision: `float32`, `bfloat16`, or `auto` (`float32` recommended on CPU) |

### Additional LoRA Arguments (`v0.3.1` Only)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--lora-rank` | `0` | LoRA rank ($r > 0$ freezes `--init-checkpoint` and trains low-rank adapters; `0` = standard full-model training) |
| `--lora-alpha` | `32.0` | LoRA scaling factor ($\alpha / r$) |
| `--lora-dropout` | `0.05` | Dropout probability applied before the LoRA $A$ projection |
| `--lora-targets` | `all` | Target linear modules: `all`, `ffn`, `ssm`, or comma-separated layer names |
| `--lora-adapter` | `None` | Path to a saved `AdapterXX.pt` file for `generate`, `chat`, `eval-*`, `merge-lora`, or continued LoRA tuning |
| `--save-adapter` | `None` | Optional custom output path for the trained adapter `.pt` (defaults to `Adapters\AdapterXX.pt`) |
| `--merge-lora` | `False` | When training with `--lora-rank > 0`, also save a merged standalone `Models\TestXX.pt` checkpoint |

## 10. PowerShell Tips & Troubleshooting

* **Quote Paths With Spaces:** Always wrap paths like `"C:\Training Datasets\Tinystories"` or `"Models\Test01.pt"` in double quotes so PowerShell passes them as a single argument.
* **Low-Resource Background CPU Training (`--threads 2`):** On multi-core CPUs, passing `--threads 2` caps PyTorch to two physical cores so your system remains responsive and cool during long runs with minimal cache-sync overhead.
* **Inspecting JSON Reports Directly in PowerShell:** You can inspect any saved run report's config or scores right in your terminal:
  ```powershell
  Get-Content "reports\Test01.json" | ConvertFrom-Json | Select-Object -ExpandProperty model_config
  ```
* **`ValueError: Insufficient tokens (...) for seq_len=...`:** Your dataset file is shorter than `seq_len + 1` bytes. Add more text or lower `--seq-len` (e.g., `--seq-len 64`).
* **`ValueError: CLI model architecture flags do not match --init-checkpoint config`:** Check `reports\TestXX.json` under `"model_config"` and pass the exact same `--d-model`, `--layers`, `--heads`, and `--ffn-expansion` flags, or switch to `--expand-checkpoint` if you intend to scale the model up.
* **`ValueError: LoRA training (--lora-rank > 0) requires --init-checkpoint`:** In `v0.3.1`, LoRA always needs an existing base checkpoint via `--init-checkpoint "Models\TestXX.pt"`.
* **`.parquet` or `.arrow` files are not loading:** Install `pyarrow` (`python -m pip install pyarrow`) so the streaming reader can unpack columnar files.
* **Maximizing CPU Throughput:** Leave `--dtype float32` on CPU (PyTorch CPU `bfloat16` emulation is slower than `float32` on most x86 processors).
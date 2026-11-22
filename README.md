# Gutenberg Chaotic Primordial

**Version:** 0.3.0  
**Author:** Juan Carlo Sagre  
**License:** Apache License 2.0  
**Implementation:** Standalone Python script using PyTorch

Gutenberg Chaotic Primordial is an experimental, self-contained language-model training and inference engine. Version 0.3.0 combines a **Selective Diagonal State-Space Model (SSM)** with a **dual-stream, delta-rule associative memory**. It uses lossless UTF-8 byte-level tokenization and supports recurrent, stateful streaming inference.

The project is designed to be portable: users can keep their own datasets wherever they are stored and pass the location to the script at runtime. **You do not need to edit the Python source to change dataset directories.**

> This repository provides a model architecture and experimental training pipeline. It does not claim production-grade language understanding or guarantee a particular result from a user's dataset. Training quality depends on the data, model configuration, compute resources, and training settings.

---

## 1. Architecture at a glance

The model is a hybrid sequence architecture built from the following components:

- **Lossless UTF-8 byte-level tokenizer** — vocabulary size 260: four special tokens plus all 256 byte values. This avoids a learned word/subword vocabulary and has no out-of-vocabulary byte values.
- **Causal depthwise 1D convolution** — provides local causal mixing and carries its convolution state across chunks.
- **Selective Diagonal SSM** — input-selective state updates with exact zero-order-hold discretization and chunkwise parallel scanning.
- **Dual-stream Delta-Rule Associative Memory** — multi-head fast-weight memory with gated delta-rule updates, shifted-key/phrase context, and recurrent memory state.
- **SwiGLU feed-forward network** — channel mixing in the residual blocks.
- **Stateful streaming path** — recurrent states are passed from one input chunk or token to the next, rather than retaining the entire prior token sequence. This provides constant-size recurrent state with respect to the length of the stream for a fixed model configuration.

Conceptual layer flow:

```text
UTF-8 bytes
    │
    ▼
Token Embedding
    │
    ▼
Causal Convolution + Selective Diagonal SSM
    │
    ▼
Dual-Stream Delta-Rule Associative Memory
    │
    ▼
SwiGLU / Residual Processing
    │
    ▼
Normalization + Language-Model Head
    │
    ▼
Next-byte logits / generated text
```

The implementation uses chunkwise scans for the SSM and delta-rule memory paths. It includes internal checks for chunk continuation, streaming equivalence, and finite gradients.

## 2. Main capabilities

- Train a new model from the built-in Stage 1 and Stage 2 mathematics curriculum.
- Train from a user-supplied dataset file or a directory containing supported dataset files.
- Continue training from a compatible v0.3.x checkpoint.
- Expand a smaller checkpoint to a wider/deeper model using the implemented Net2Net-style expansion routine.
- Consolidate training with one or more frozen teacher checkpoints through KL-divergence knowledge distillation.
- Run foundation-math and higher-math benchmark suites.
- Generate text from a prompt or start an interactive terminal chat session.
- Save sequential checkpoints, JSON training reports, and a SHA-256 checkpoint digest.
- Select CPU or CUDA execution, control CPU threads, and select supported compute precision.

## 3. Requirements

- **Python:** 3.10 or newer (the accompanying usage guide records testing with Python 3.11).
- **Core dependency:** PyTorch (`torch`). CPU is supported; CUDA can be selected when available and correctly configured.
- **Optional dataset dependency:**
  - `pyarrow` is required for Parquet (`.parquet`) files.
  - `pyarrow` or Hugging Face `datasets` can be used to read Arrow (`.arrow`) files.
  - `.jsonl`, `.ndjson`, `.json`, `.txt`, and `.md` are handled using Python's standard library and need no additional dataset package.

Install PyTorch using the installation command appropriate for your operating system and CPU/CUDA environment. For example, a CPU-only installation can be obtained from the official PyTorch installation selector: <https://pytorch.org/get-started/locally/>.

Check the command-line options available in the copy of the script you are using:

```bash
python gutenberg_chaotic_primordial_v0_3_0.py --help
```

## 4. First run and self-test

Run the built-in verification before training:

```bash
python gutenberg_chaotic_primordial_v0_3_0.py --mode self-test --device cpu --threads 2
```

The self-test checks:

1. Full-sequence output against chunked SSM-block continuation.
2. Full-sequence output against chunked delta-rule-memory continuation.
3. Full forward inference against token-by-token streaming inference, plus a backward pass and finite-gradient checks.

A successful run prints a JSON report with `"overall_status": "PASS"`.

The normal pretraining flow also runs these self-tests before it proceeds to load the training data and start optimization.

## 5. Portable dataset loading — use your own directories

### No hard-coded dataset directory

Use `--data-path` to provide the dataset location when launching the script. The value can be:

- A path to one supported dataset file; or
- A path to a directory. The loader recursively searches that directory and its subdirectories for supported file extensions.

The location can be relative to the current working directory or an absolute path. Put paths containing spaces inside quotes. The examples below are templates: replace the path with the location on **your own** computer.

**Windows PowerShell example:**

```powershell
python .\gutenberg_chaotic_primordial_v0_3_0.py `
  --data-path "D:\Training Data\My Custom Dataset" `
  --max-docs 25000 `
  --math-replay-ratio 0.20 `
  --steps 1500 `
  --lr 4e-4 `
  --device cpu `
  --threads 2
```

**Linux / macOS example:**

```bash
python ./gutenberg_chaotic_primordial_v0_3_0.py \
  --data-path "/home/your-user/datasets/my_custom_dataset" \
  --max-docs 25000 \
  --math-replay-ratio 0.20 \
  --steps 1500 \
  --lr 4e-4 \
  --device cpu \
  --threads 2
```

**Single-file example:**

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --data-path "/path/to/your/data/training.jsonl" \
  --steps 1200
```

The script resolves and validates the path at runtime. If the path does not exist, or no supported dataset files are found, it reports an error containing the resolved location. No source-code path edits are required.

### Supported file formats

| Extension | Loader behavior | Additional package |
|---|---|---|
| `.jsonl`, `.ndjson` | Reads one JSON record per non-empty line | None |
| `.json` | Reads a JSON list, supported record-list wrapper, or a single record | None |
| `.txt`, `.md` | Splits into documents on blank lines; if there are no blank-line separators, uses lines | None |
| `.parquet` | Reads Parquet rows | `pyarrow` |
| `.arrow` | Reads Arrow IPC or Hugging Face Arrow dataset files | `pyarrow` or `datasets` |

For a directory input, supported files are found recursively. The loader ignores common Hugging Face metadata files named `dataset_info.json` and `state.json`.

### Recognized record structures

For structured records, the loader extracts text from these common fields or combinations:

| Record structure | Extracted training text |
|---|---|
| A plain string | The string itself |
| `{"text": "..."}` (also `story`, `content`, `document`, `body`, or `passage`) | The selected text field |
| `{"messages": [{"role": "user", "content": "..."}, ...]}` | Role-labelled message content |
| `{"instruction": "...", "input": "...", "output": "..."}` | Instruction, optional input, and response |
| `{"instruction": "...", "response": "..."}` | Instruction and response |
| `{"prompt": "...", "response": "..."}` or `completion` | User prompt and assistant response |
| `{"input": "...", "output": "..."}` | Input and output |
| `{"question": "...", "answer": "..."}` | Question and answer |
| `{"word": "...", "definition": "..."}` or `meaning` | Word and meaning, with optional example |

For `.json` files, a top-level list is accepted. A top-level object can contain a list under `data`, `rows`, `records`, `examples`, or `train`; otherwise it is treated as one record. Records that do not match a supported structure or contain too little text are skipped.

### Dataset sampling and math replay

| Option | Default | Meaning |
|---|---:|---|
| `--max-docs` | `25000` | Maximum number of valid records sampled from the discovered files. Use `0` to load all valid records. |
| `--val-ratio` | `0.05` | Fraction of sampled documents set aside for validation. |
| `--math-replay-ratio` | `0.20` | Fraction of built-in Stage 1/2 mathematics examples interleaved with external-dataset training. Set to `0` to disable math replay. |
| `--seq-len` | `128` | Token sequence length used when packing the training data. |
| `--seed` | `1701` | Seed used for reproducible sampling and training setup. |

The loader tokenizes documents and packs them into sequence blocks. It inserts padding gaps between documents to prevent recurrent state from carrying directly from one document into the next within the packed stream.

When `--data-path` is omitted, the script uses its built-in mathematics curriculum instead of searching for an external dataset. The built-in material includes binary conversion, carry addition, borrow subtraction, multiplication, pre-algebra, algebra, geometry, trigonometry, and calculus.

## 6. Training examples

### Train from the built-in mathematics curriculum

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --steps 1200 \
  --lr 4e-4 \
  --device cpu \
  --threads 2
```

### Train from your external dataset

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --data-path "/your/custom/dataset/location" \
  --max-docs 25000 \
  --math-replay-ratio 0.20 \
  --steps 1500 \
  --lr 4e-4 \
  --device cpu \
  --threads 2
```

### Run without saving a checkpoint

Use `--no-save` for a trial run that trains and reports results without writing the `TestXX.pt` checkpoint or its `TestXX.json` report:

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --data-path "/your/custom/dataset/location" \
  --steps 20 \
  --no-save
```

### Continue training from a checkpoint

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --init-checkpoint "/path/to/Models/Test01.pt" \
  --steps 450 \
  --lr 8e-5
```

When using `--init-checkpoint`, the model configuration supplied to the new run must exactly match the saved checkpoint configuration. This includes the architecture settings represented by `--d-model`, `--layers`, `--state-size`, `--heads`, `--conv-kernel`, `--ssm-chunk-size`, `--ffn-expansion`, and `--dropout`. Use the original settings when continuing an existing model.

### Expand a checkpoint

The expansion routine can increase model width, depth, and state size while retaining copied weights from the source model. The target must not be smaller than the source in `d_model`, layer count, or state size. The target head count must match the source head count, and `d_model` must be divisible by the number of heads.

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --expand-checkpoint "/path/to/Models/Test01.pt" \
  --d-model 320 \
  --layers 5 \
  --heads 8 \
  --ffn-expansion 2.2 \
  --steps 250 \
  --lr 8e-5
```

### Multi-teacher knowledge distillation

One or more teacher checkpoint paths can be supplied as a comma-separated list. Teacher models are loaded in evaluation mode and frozen while the student is trained. The implementation averages teacher probability distributions and uses a KL-divergence distillation term weighted by `--distill-alpha`.

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --init-checkpoint "/path/to/student/Models/Test01.pt" \
  --teacher-checkpoints "/path/to/teacher-a.pt,/path/to/teacher-b.pt" \
  --distill-alpha 0.20 \
  --steps 450 \
  --lr 8e-5
```

## 7. Model-size presets

The accompanying v0.3.0 usage guide lists these reference configurations:

| Preset | `--d-model` | `--layers` | `--heads` | `--ffn-expansion` | Approx. parameters |
|---|---:|---:|---:|---:|---:|
| Nano (default) | 256 | 4 | 8 | 2.0 | 4,875,972 (~4.88M) |
| Small | 320 | 5 | 8 | 2.2 | 10,055,285 (~10.06M) |
| Medium | 384 | 5 | 8 | 2.35 | 15,084,533 (~15.08M) |

These are model-size references, not hardware-performance guarantees. Actual memory use and training speed depend on sequence length, batch size, gradient accumulation, precision, and the available CPU/GPU.

## 8. Evaluation, generation, and chat

### Math evaluation

The combined benchmark runs both suites in one pass:

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --mode eval-all-math \
  --checkpoint "/path/to/Models/Test01.pt"
```

| Evaluation mode | Cases | Coverage |
|---|---:|---|
| `eval-math` | 40 | Addition, subtraction, multiplication, and base-10 to base-2 conversion |
| `eval-higher-math` | 25 | Pre-algebra, algebra, geometry, trigonometry, and calculus |
| `eval-all-math` | 65 | Both suites combined |

The current evaluator checks whether an expected answer marker appears in generated output. Treat these scores as results from this project's built-in benchmark—not as a complete assessment of mathematical reasoning or general intelligence.

If `--checkpoint` is omitted for an evaluation or inference mode, the program loads the highest-numbered `TestXX.pt` file under the selected output root's `Models` directory.

### Generate text

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --mode generate \
  --checkpoint "/path/to/Models/Test01.pt" \
  --prompt "Alg: Solve 4x + 7 = 31 ->" \
  --temperature 0.0
```

Use `--temperature 0.0` for deterministic greedy generation. For more varied text, the usage guide suggests experimenting with approximately `0.5` to `0.7`.

### Interactive terminal chat

```bash
python gutenberg_chaotic_primordial_v0_3_0.py \
  --mode chat \
  --checkpoint "/path/to/Models/Test01.pt" \
  --temperature 0.6
```

Type `exit` or `quit` to close the interactive session.

## 9. Checkpoints, reports, and portable output paths

By default, output is placed beside the Python script. The script creates these directories automatically:

```text
<output-root>/
├── Models/
│   ├── Test01.pt
│   ├── Test02.pt
│   └── ...
└── reports/
    ├── Test01.json
    ├── Test02.json
    └── ...
```

Each training run receives the next available sequential identifier. Existing checkpoints are not overwritten. The JSON report contains run metadata, model configuration, dataset statistics, environment details, and the SHA-256 digest of the checkpoint.

To store outputs somewhere else—for example, on a different drive—set `--output-root`:

```powershell
python .\gutenberg_chaotic_primordial_v0_3_0.py `
  --data-path "D:\Training Data\My Corpus" `
  --output-root "E:\GCP Experiments\Run 01" `
  --steps 1500
```

The dataset directory and output directory are independent settings. You can store the source code, dataset, and checkpoints in three different locations.

## 10. Useful runtime options

| Option | Default | Description |
|---|---|---|
| `--device` | `auto` | Select `auto`, `cpu`, or `cuda`. |
| `--threads` | `0` | CPU thread count; `0` auto-detects physical cores. |
| `--dtype` | `float32` | Select `float32`, `bfloat16`, or `auto`; float32 is the documented CPU recommendation. |
| `--batch-size` | `8` | Training batch size. |
| `--grad-accum` | `1` | Number of gradient-accumulation steps. |
| `--steps` | `1200` | Number of optimizer steps. |
| `--lr` | `4e-4` | Learning rate. |
| `--checkpoint` | unset | Explicit checkpoint path for evaluation or inference. |
| `--output-root` | Script directory | Root directory for `Models/` and `reports/`. |
| `--no-save` | off | Skip checkpoint and report creation for a trial run. |

Run `--help` for the complete list of options.

## 11. Current validation record

The v0.3.0 script's CPU self-test was executed during preparation of this README using `--device cpu --threads 2`. It returned `overall_status: PASS` for all three test groups:

- SSM block chunk-continuation maximum absolute output error: approximately `2.38e-7`.
- Delta-rule memory chunk-continuation maximum absolute output error: approximately `1.19e-7`.
- Full-forward versus token-streaming maximum absolute logit error: approximately `1.67e-6`; the backward check reported finite gradients for 87 trainable tensors.

This is a focused architecture/self-test result. It is not evidence that every dataset, model size, operating system, accelerator, or training configuration has been validated. Run the self-test in your own environment before beginning a substantial training job.

## 12. Responsible use and dataset ownership

Users are responsible for ensuring that they have the rights and permissions necessary to use their chosen training datasets, checkpoints, and other inputs. The repository's license applies to the project code; it does not automatically grant rights to third-party datasets or model weights.

---

## License

Copyright 2026 Juan Carlo Sagre.

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at <https://www.apache.org/licenses/LICENSE-2.0> or in the accompanying [`LICENSE`](LICENSE) file. The full license text is reproduced below.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

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
```

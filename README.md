# 🎸 Open-Fret: MIDI to Guitar Tablature Transformer

[![🤗 Hugging Face: open-fret-tab](https://img.shields.io/badge/🤗_Hugging_Face-open--fret--tab-yellow)](https://huggingface.co/mylonasc/open-fret-tab)
[![Dataset: vldsavelyev/guitar_tab](https://img.shields.io/badge/🤗_Dataset-vldsavelyev%2Fguitar__tab-orange)](https://huggingface.co/datasets/vldsavelyev/guitar_tab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12 tested](https://img.shields.io/badge/python-3.12%20tested-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)

**Open-Fret** is a research MIDI-to-guitar-tablature pipeline using a small T5 model inspired by the **Fretting-Transformer** architecture. The current trained model is a development baseline — downloadable from [Hugging Face](https://huggingface.co/mylonasc/open-fret-tab) with full training info and dataset provenance — not a published benchmark. See the [acceptance and issue ledger](docs/issue_triage.md) for verified scope and remaining work.

---

## ✨ Key Features

- **🎯 Anchored decoding**: Preserves supported input attacks and onset timing while selecting pitch-equivalent fingerings; unsupported/unassignable inputs fail explicitly.
- **⚡ Small T5 model**: Supports CPU and CUDA execution; throughput depends on input and hardware.
- **🎸 Fingering constraints**: Enforces distinct strings within each onset. Sustained-string occupancy and general human playability are not guaranteed.
- **🔧 Multi-Config**: Support for custom tunings and capo positions (work in progress).
- **📦 End-to-End Pipeline**: Complete scripts for data downloading, processing, vocabulary generation, and training.

---

## 🌐 Live site & demo

[**📖 Documentation website →**](https://mylonasc.github.io/open-fret/) —
motivation, dataset pipeline, loss definitions, HPO insights, training info,
and playable examples with real model predictions.

[**▶ Synthetic demo →**](https://mylonasc.github.io/open-fret/examples/synthetic/)

A static page rendering a playable, seekable AlphaTab score next to a
MIDI piano roll, with a predicted/actual fingering toggle. It uses a hand-authored
synthetic fixture; no corpus audio or reference annotations from a corpus are
published there (model weights are released separately on
[Hugging Face](https://huggingface.co/mylonasc/open-fret-tab)), and no
server-side inference runs. Build and hosting details are in
[docs/pages_demo.md](docs/pages_demo.md).

---

## 🚀 Getting Started

For the tested CPU setup, real end-to-end fixture and full-song inference
contract, see [engineering acceptance](docs/engineering_acceptance.md).

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/mylonasc/open-fret.git
cd open-fret
pip install -r requirements.txt
```

### 2. Dataset Preparation

Open-Fret uses the **vldsavelyev/guitar_tab** AlphaTex corpus. The modular
preprocessing lives in `open_fret/` (`alphatex`, `sequences`, `splits`,
`postprocess`, `render`, `training`); `scripts/` are thin CLI wrappers.

```bash
npm ci --prefix tools/alphatex_parser  # pinned extended AlphaTab parser
# Real CPU fixture: no corpus download, no production artifacts
python scripts/run_pipeline.py --fixture --workspace /path/to/new-fixture
python -m pytest tests/ -q
# Full corpus: separate versioned workspace, no automatic training
python scripts/run_pipeline.py --workspace dataset/processed/new-protocol-v2 --prepare-only
```

The pipeline splits song families/full-source fingerprints **before** deterministic
training-only capo augmentation. Primary validation/test remain unaugmented.
TAB vocabulary comes from training or predefined fretboard tokens; structural
input tokens have a documented all-split policy. Existing data is never silently
overwritten or treated as valid merely because its directory exists.

### 3. Training

Train the `T5-Tiny-Tab` model from scratch (fresh by default):

```bash
# Fresh run (default)
python scripts/train_model.py

# Resume latest checkpoint in --output-dir, or fail visibly if none
python scripts/train_model.py --resume

# Resume an explicit checkpoint
python scripts/train_model.py --resume models/tiny-tab-v1/checkpoint-XXXX
```

Training auto-selects CUDA/fp16 when available, otherwise CPU/full precision.
Memory needs depend on configuration; no fixed VRAM budget is guaranteed.
Resume requires model, trainer, optimizer, scheduler and RNG checkpoint files;
an absent/incomplete checkpoint fails instead of silently starting fresh.
Use `--init-from` separately for weights-only initialization into a new run.

### 4. Inference

Convert your MIDI files to tokens, and then to tablature:

```bash
python scripts/inference.py --midi path/to/your/song.mid --model path/to/local/run/final
# Optional: --out predictions.tokens
# Sampling is legacy-only: --decoding-mode legacy --stochastic
```

### 5. Hyperparameter tuning (Optuna)

```bash
python scripts/hpo_optuna.py --trials 20 --storage sqlite:///hpo.db
# Best config -> models/hpo/best_config.json; retrain via scripts/train_model.py
```

### 6. Experimental pitch-aware losses

Training and Optuna support separate opt-in `--pitch-loss-alpha 0.3`,
`--equivalence-smoothing 0.1`, `--event-type-alpha 0.3` and
`--conditional-pitch` experiments; ordinary CE remains the default.
**Rebuild processed data/cache first:** inline AlphaTex durations previously
failed to carry forward, and unsupported syntax was silently skipped. New loss
experiments reject obsolete/unversioned caches. Rebuild the vocabulary with
`--structural-lists` covering every split so unseen `TUNING_*`/`TS_*`
conditioning never becomes `[UNK]`; the cache builder now fails loudly instead of
silently substituting it. The cache also masks same-onset/same-string reference
groups that cannot be fingered. See
[the loss integration guide](docs/domain_aware_losses.md) for safe rebuild paths,
formulas, matched CE controls, resume behavior and verification commands. These
features have not yet demonstrated a model-quality gain.
The user-requested [fresh combined-loss full run](docs/full_combined_run.md)
uses CE weight 0.3, smoothing 0.1, pitch weight 1 and event-type weight 0.3,
with automatic HTML history.

---

## 🧠 Architecture: Fretting-Transformer Standard
Open-Fret strictly follows the base architecture defined in the research paper:

- **Type**: Encoder-Decoder (Seq2Seq T5)
- **Layers**: 3 Encoder / 3 Decoder
- **Dimensions**: `d_model = 128`, `d_ff = 1024`
- **Attention**: 4 Heads
- **Conditioning**: Prefix-based conditioning for `CAPO` and `TUNING`.
- **Tokenizer**: WordLevel Tokenizer (Treats musical events as atomic units).

### Training Protocol
- **Effective Batch Size**: 64 (via Gradient Accumulation)
- **Learning Rate**: 5e-4 with **Cosine Decay**.
- **Optimizer**: Adafactor (Paper standard).
- **Augmentation**: Deterministic source-level capo derivatives, training only after splitting; no automatic tuning-shift augmentation.

---

## 📊 Datasets

**The current model uses one documented real training corpus:
[`vldsavelyev/guitar_tab`](https://huggingface.co/datasets/vldsavelyev/guitar_tab).**
It contains GuitarPro tablatures converted to AlphaTex. Upstream reports 47,299
source files in its default configuration and 188,297 track records in its `all`
configuration; these are different units, not two datasets. Parser acceptance,
windowing and augmentation determine the processed example counts.

- **Current run:** `full-curated-3ep-20261109` uses the
  `loss-v4-curated-20261109` cache: 843,886 training windows and 105,394 validation
  windows. These are not counts of independent songs. Its recorded initialization
  is from scratch (`init_from: null`). See the [run documentation](docs/full_combined_run.md).
- **Processing versions:** `loss-v3` and `loss-v4-curated` are processing/cache
  versions of the same corpus, not additional data sources.
- **Not used here:** DadaGP and SynthTab are not ingested by the current training
  pipeline. Earlier README references mixed research-paper context with this
  implementation's actual inputs. This project does not reproduce the
  [Fretting-Transformer paper's](https://arxiv.org/abs/2506.14223) data protocol,
  and no cross-dataset quality advantage has been established.

**Historical clarification:** the downloader already selected `vldsavelyev/guitar_tab`
in the initial commit (`8b792f7`), despite the original README claiming
DadaGP/SynthTab. Commit `6636893` revised the dataset description but did not
change the downloader. The available Git history does not establish an actual
switch from those datasets or document a controlled comparison justifying this
corpus choice.

Dataset/model publication rights remain unresolved; see the
[licensing review](docs/dataset_licensing.md). Public availability is not a license.

---

## 🗺️ Roadmap
- [x] **Anchored attack decoding**: Pitch-legal choices with distinct strings per onset; sustained occupancy remains open.
- [ ] **Complete technique fidelity**: Parser/token support does not imply all effects survive generation and playback.
- [ ] **Audio-to-Tab API**: Full end-to-end transcription from MP3/WAV.
- [ ] **ONNX Quantization**: Faster CPU inference for web deployment.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

Original project code is licensed under the [MIT License](LICENSE). Third-party
components retain their own terms. **The training dataset and trained weights
are not automatically covered by this code license.** No express dataset license
was found in the reviewed `vldsavelyev/guitar_tab` card/repository metadata;
public/commercial model-release rights are not established. See the
[dataset licensing review](docs/dataset_licensing.md) before publishing weights.

## 🙏 Acknowledgments

- **Primary Research**: [Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription](https://arxiv.org/abs/2506.14223) by Anna Hamberger, Sebastian Murgul, Jochen Schmidt, and Michael Heizmann.
- Built with ❤️ using [HuggingFace Transformers](https://huggingface.co/transformers/).

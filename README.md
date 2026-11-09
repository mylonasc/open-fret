# 🎸 Open-Fret: MIDI to Guitar Tablature Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12 tested](https://img.shields.io/badge/python-3.12%20tested-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)

**Open-Fret** is a research MIDI-to-guitar-tablature pipeline using a small T5 model inspired by the **Fretting-Transformer** architecture. The current trained model is a development baseline, not a published benchmark or downloadable model release. See the [acceptance and issue ledger](docs/issue_triage.md) for verified scope and remaining work.

---

## ✨ Key Features

- **🎯 Anchored decoding**: Preserves supported input attacks and onset timing while selecting pitch-equivalent fingerings; unsupported/unassignable inputs fail explicitly.
- **⚡ Small T5 model**: Supports CPU and CUDA execution; throughput depends on input and hardware.
- **🎸 Fingering constraints**: Enforces distinct strings within each onset. Sustained-string occupancy and general human playability are not guaranteed.
- **🔧 Multi-Config**: Support for custom tunings and capo positions (work in progress).
- **📦 End-to-End Pipeline**: Complete scripts for data downloading, processing, vocabulary generation, and training.

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
The project utilizes the following data sources for training and evaluation:
- **[vladsavelyev/guitar_tab](https://huggingface.co/datasets/vldsavelyev/guitar_tab)**: A massive corpus of ~188k GuitarPro tracks converted to AlphaTex format. This serves as our primary training database for large-scale generalization.
- **DadaGP/SynthTab Alignment**: While the [Fretting-Transformer paper](https://arxiv.org/abs/2506.14223) focuses on a curated subset (Acoustic, Beginner, and Jazz), this implementation leverages the broader `vladsavelyev` corpus to provide a more robust and versatile model.

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Primary Research**: [Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription](https://arxiv.org/abs/2506.14223) by Anna Hamberger, Sebastian Murgul, Jochen Schmidt, and Michael Heizmann.
- Built with ❤️ using [HuggingFace Transformers](https://huggingface.co/transformers/).

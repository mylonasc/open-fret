# 🎸 Open-Fret: MIDI to Guitar Tablature Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)

**Open-Fret** is a high-performance, research-driven MIDI to Guitar Tablature transcription engine. Based on the **Fretting-Transformer** architecture, it leverages a custom-tailored T5 Transformer model to translate symbolic music (MIDI) into ergonomic, playable, and accurate guitar tabs.

---

## ✨ Key Features

- **🎯 High Accuracy**: Translates MIDI notes into precise string and fret combinations.
- **⚡ CPU Optimized**: Specifically designed `T5-Tiny-Tab` architecture for fast inference on standard hardware.
- **🎸 Ergonomics First**: Learns playable fingerings from massive datasets of real GuitarPro files.
- **🔧 Multi-Config**: Support for custom tunings and capo positions (work in progress).
- **📦 End-to-End Pipeline**: Complete scripts for data downloading, processing, vocabulary generation, and training.

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/open-fret.git
cd open-fret
pip install -r requirements.txt
```

### 2. Dataset Preparation

Open-Fret uses the **vldsavelyev/guitar_tab** AlphaTex corpus. The modular
preprocessing lives in `open_fret/` (`alphatex`, `sequences`, `splits`,
`postprocess`, `render`, `training`); `scripts/` are thin CLI wrappers.

```bash
# 0. Quick fixture check (no download, no GPU)
/tmp/opencode/venv/bin/python scripts/smoke_test.py
/tmp/opencode/venv/bin/python -m pytest tests/ -q

# 1. Download the dataset (if applicable/accessible)
python scripts/00_download.py

# 2. Process AlphaTex into aligned MIDI/tab windows
npm ci --prefix tools/alphatex_parser  # pinned extended AlphaTab parser
python scripts/01_process.py

# 3. Split by source song (no leakage across train/val/test)
python scripts/03_split.py

# 4. Generate the vocabulary (prefer training-only for final runs)
python scripts/01.5_generate_vocab.py --train-list dataset/processed/train_files.txt

# 5. Tokenize the sequences for the model
python scripts/02_tokenize.py

# 6. Cache aligned train/validation/test tensors
python scripts/04_cache_dataset.py
```

> Vocab built on all windows leaks held-out tokens; use `--train-list` for
> final runs.

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

*Note: Optimized to train on GPUs with as little as 8GB VRAM using the Adafactor optimizer.*

### 4. Inference

Convert your MIDI files to tokens, and then to tablature:

```bash
python scripts/inference.py --midi path/to/your/song.mid
# Optional: --out predictions.tokens --stochastic for sampling
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
- **Augmentation**: Random Capo (0-7) and Tuning shifts across training samples.

---

## 📊 Datasets
The project utilizes the following data sources for training and evaluation:
- **[vladsavelyev/guitar_tab](https://huggingface.co/datasets/vldsavelyev/guitar_tab)**: A massive corpus of ~188k GuitarPro tracks converted to AlphaTex format. This serves as our primary training database for large-scale generalization.
- **DadaGP/SynthTab Alignment**: While the [Fretting-Transformer paper](https://arxiv.org/abs/2506.14223) focuses on a curated subset (Acoustic, Beginner, and Jazz), this implementation leverages the broader `vladsavelyev` corpus to provide a more robust and versatile model.

---

## 🗺️ Roadmap
- [x] **Neighbor Search Implementation**: 100% pitch matching between MIDI and Tab (Section 3.5).
- [x] **Technique Layer**: Full support for Bends, Slides, and Vibrato in the token stream.
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

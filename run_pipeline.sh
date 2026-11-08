#!/bin/bash

# ==============================================================================
# Fretting-Transformer: End-to-End Pipeline
# ------------------------------------------------------------------------------
# This script executes the full pipeline from data download to model training.
# Ensure you have the requirements installed: pip install -r requirements.txt
# Order matters: split BEFORE vocab so the vocabulary is built from training
# windows only (no held-out leakage).
# ==============================================================================

set -e # Exit on error

echo "🚀 Starting Fretting-Transformer Pipeline..."

# 1. Download Dataset
if [ ! -f "dataset/raw/data.zip" ]; then
    echo "📥 Step 1: Downloading vldsavelyev/guitar_tab dataset..."
    python scripts/00_download.py
else
    echo "⏭️ Step 1: Dataset already downloaded. Skipping."
fi

# 2. Process & Augment (musically aligned windows)
if [ ! -d "dataset/processed/individual" ]; then
    echo "⚙️ Step 2: Processing AlphaTex files and applying augmentations..."
    python scripts/01_process.py
else
    echo "⏭️ Step 2: Data already processed. Skipping."
fi

# 3. Split Dataset (by source song - no leakage across splits)
if [ ! -f "dataset/processed/train_files.txt" ]; then
    echo "✂️ Step 3: Splitting into Train/Val/Test..."
    python scripts/03_split.py
else
    echo "⏭️ Step 3: Dataset split already exists. Skipping."
fi

# 4. Generate Vocabulary (training windows only)
if [ ! -f "dataset/processed/unique_tokens.json" ]; then
    echo "📖 Step 4: Generating technique-aware vocabulary..."
    python scripts/01.5_generate_vocab.py --train-list dataset/processed/train_files.txt
else
    echo "📖 Step 4: Vocabulary already exists. Skipping."
fi

# 5. Tokenization
if [ ! -d "dataset/processed/tab_tokenizer" ]; then
    echo "🔢 Step 5: Tokenizing the dataset for T5..."
    python scripts/02_tokenize.py
else
    echo "⏭️ Step 5: Tokenizer already exists. Skipping."
fi

# 6. Caching (Performance Step)
if [ ! -d "dataset/processed/cached_dataset" ]; then
    echo "📦 Step 6: Caching dataset for high-speed training..."
    python scripts/04_cache_dataset.py
else
    echo "⏭️ Step 6: Cached dataset already exists. Skipping."
fi

# 7. Training (fresh by default; add --resume to continue, or
#    --init-from t5-small to fine-tune a pretrained T5 endpoint)
echo "🎸 Step 7: Initializing Model Training..."
python scripts/train_model.py "$@"

echo "✅ Pipeline Complete! Check the 'models/' directory for checkpoints."

# Transformer From Scratch (PyTorch) — "Attention Is All You Need"

A from-scratch PyTorch implementation of the original Transformer architecture (Vaswani et al., *Attention Is All You Need*, 2017), trained for English→Italian translation. Built while following a video walkthrough — see **Credits** below.

## Credits

This implementation follows [Umar Jamil's YouTube tutorial](https://youtu.be/ISNdQcPhsts), based on his [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer) repo. The encoder/decoder/attention code here was written by hand while working through that video — it is a learning/portfolio project, not an original architecture design. Please credit the original author if you build on this, and check the source repo directly for its terms before reusing the code elsewhere (as of writing, it carries no explicit open-source license).

## What's Implemented

- Full encoder-decoder Transformer from scratch — embeddings, sinusoidal positional encoding, multi-head self/cross-attention, feed-forward blocks, layer norm, residual connections — no `nn.Transformer` shortcuts
- Custom tokenizer training (word-level, via HuggingFace `tokenizers`) on the `opus_books` en–it dataset
- Training loop with TensorBoard logging (loss, CER, WER, BLEU) and checkpointing/resume
- Greedy decoding for inference
- Attention visualization notebook — renders per-layer, per-head attention heatmaps (encoder self-attention, decoder self-attention, encoder–decoder cross-attention) with Altair

## Project Structure

```
.
├── model.py               # Transformer architecture (embeddings, attention, encoder/decoder blocks)
├── dataset.py              # BilingualDataset — tokenization, padding, masks
├── config.py               # Hyperparameters and checkpoint path helpers
├── train.py                 # Training loop, validation, greedy decoding
├── attention_visual.ipynb   # Attention heatmap visualization
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python train.py
```
This downloads `opus_books` (en–it), trains word-level tokenizers on first run, and checkpoints to `opus_books_weights/` after every epoch. Progress and validation metrics (BLEU, WER, CER) log to TensorBoard under `runs/`.

**Visualize attention:**
Open `attention_visual.ipynb`, load a trained checkpoint, and run the cells to render attention heatmaps for the encoder, decoder, and cross-attention layers.

> **Note:** if you hit `ModuleNotFoundError: No module named 'torchtext'` in `train.py`, it's a leftover, unused import (`import torchtext.datasets as datasets`) — the actual dataset loading uses HuggingFace's `datasets` library instead. Just delete that one line.

## Config

Key hyperparameters (`config.py`): `d_model=512`, 6 encoder/decoder layers, 8 attention heads, `seq_len=350`, batch size 8, Adam optimizer with `lr=1e-4`, 20 epochs.

## Results


## Reference

Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

# Hibiri

Hibiri is a real-time and multilingual speech translation model.
It translates from French, Spanish, Portuguese and German to English: accurately, with low latency, high audio quality, and voice transfer.

https://github.com/user-attachments/assets/d533ec45-8d5e-4e41-886a-0b2d198be6f3

[🤗 Hugging Face Model Card](https://huggingface.co/kyutai/hibiri-3b-pytorch-bf16) |
[⚙️ Tech report](https://kyutai.org/blog/2026-02-12-hibiri) |
[📄 Paper](https://arxiv.org/abs/2602.11072) |
[🎧 More samples](https://huggingface.co/spaces/kyutai/hibiri-samples)

## Requirements

Hibiri is a 3B-parameter model and requires an NVIDIA GPU to run: 8 GB VRAM should work, 12 GB is safe.

## Run the server

Hibiri comes with a server you can run to interact with Hibiri in real time. To run it, just use:

```python
uvx -p 3.13 hibiri serve [--gradio-tunnel]
```

Then go to the URL displayed to try out Hibiri.
The `--gradio-tunnel` flag will forward the server to a public URL that you can access from anywhere.

If you don't have `uv`, you must first install hibiri with `pip install hibiri` and then run the server with `hibiri serve [--gradio-tunnel]`.

## Run inference

If you'd like to run Hibiri on existing audio files, run:

```python
uvx -p 3.13 hibiri generate [--file /path/to/my/audio.wav --file /path/to/another/audio.mp3]
```

Batch inference is supported, meaning you can run the model on multiple audio files at the same time.

## Local development

We recomment using `uv`, run anything with `uv run` in this repository. For example

```bash
uv run some_file.py
or 
uv run hibiri serve
```
if you use pip, use `pip install -e .` before executing python commands.


# SoviaMate

**An open research effort toward end-to-end spoken dialogue systems — starting with the audio codec foundation.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#current-status)

SoviaMate is a long-term research project aiming to build an **end-to-end spoken dialogue system (SDS)**: a single model that listens, reasons, and speaks naturally, with controllable voice, robust to real-world noise, and integrable with large language models.

The first released component is **SoviaMate-Codec**, a neural audio codec designed from the ground up for LLM integration. Future releases will add a speech-to-speech LLM, dialogue management, and full-pipeline streaming.

> ⚠️ **Current scope**: SoviaMate is in active research. Today the repository ships the codec architecture and training pipeline. The full dialogue system is the goal, not the current deliverable. We are looking for collaborators and compute — see [Collaborate](#collaborate--research-partnerships).

---

## Why another codec?

Existing neural audio codecs (EnCodec, SoundStream, DAC) optimize for perceptual quality but lack the properties needed to drive a downstream speech LLM: measurable semantic preservation, noise robustness by design, and content–speaker decoupling. SoviaMate-Codec is built around four architectural choices that target exactly those properties.

### Four design choices

1. **ASR decoder *before* quantization** — A lightweight ASR head reads the encoder's continuous features and is trained jointly with the codec. Its gradient forces the encoder to bake linguistic information into its representation. Semantic fidelity becomes directly measurable (WER), not assumed.
2. **Continuous features for LLM input** — Discrete tokens are used for transmission/storage; the downstream LLM consumes the *pre-quantization* continuous features, avoiding quantization-induced information loss while keeping the codec's low-bitrate transmission path intact.
3. **Speech enhancement as a training paradigm** — The codec is trained noisy-in → clean-out, so the encoder learns to discard noise rather than encode it. Real-world robustness comes from the objective, not from post-hoc adaptation.
4. **Post-quantization speaker adapter** — Voice identity is injected *after* quantization via a hybrid AdaLN + cross-attention adapter conditioned on a 3–5 s reference. This decouples "what is said" from "who says it", enables zero-shot voice swapping, and frees the quantizer's capacity for content.

### Architecture at a glance

```
Audio Input
   │
   ▼
Encoder ──► [Continuous Features] ────┐
   │             │                    │
   │             └──► ASR Decoder     └──► LLM Input (continuous)
   │                  (text output)
   ▼
Quantizer ──► [Discrete Tokens] ──► Bitstream (transmission)
   │
   ▼
Speaker Adapter ◄── Speaker Prompt (3–5 s)
   │
   ▼
Audio Decoder ──► Clean Speech
```

A more detailed architecture write-up will accompany the forthcoming technical report.

---

## Current status

| Component | Status |
|---|---|
| Codec architecture (encoder / quantizer / decoder / ASR head / speaker adapter) | ✅ Implemented |
| Multi-objective training pipeline (audio + adversarial + text losses) | ✅ Implemented |
| Speech enhancement training (noisy → clean) | ✅ Implemented |
| Streaming inference | ✅ Supported in architecture |
| Benchmarking against EnCodec / SoundStream / DAC | 🔄 In progress |
| Pretrained checkpoint release | 🔄 In progress |
| Technical report / paper | 🔄 In progress |
| LLM integration adapters | ⏳ Planned |
| End-to-end spoken dialogue system | ⏳ Long-term goal |

Honest disclaimer: this is alpha research code. APIs will change, results are preliminary, and many evaluation numbers are not in yet.

---

## Getting started

### Prerequisites
- Python **3.12**
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU recommended for training (single-GPU inference is feasible)

### Installation
```sh
git clone https://github.com/samson-ailabs/SoviaMate.git
cd SoviaMate
uv sync --frozen
```

### Pretrained codec checkpoint
```sh
hf download samson-ailabs/SoviaMate-Codec --local-dir models/codec
```

### Training
The example training config is `configs/training/audio_codec.yaml`. Required fields are marked `???` and must be supplied:

```sh
uv run python train.py --config-name audio_codec \
    task.data.trainset.filepaths=/path/to/trainset.jsonl \
    task.data.valset.filepaths=/path/to/valset.jsonl \
    task.model.speaker_adapter.sv_checkpoint=/path/to/campplus.bin \
    loggers.tb.name=my_run \
    trainer.devices=1
```

You can also copy the file and edit it directly, or compose your own config on top of it via Hydra.

### Evaluation
```sh
uv run python scripts/eval_audio_codec.py --help
```

---

## Roadmap

The repository will evolve in three releases:

- **v0.1 — Codec foundation** *(current)*
  Stable, benchmarked SoviaMate-Codec with ASR-constrained encoding, zero-shot speaker adaptation, and enhancement-trained robustness.
- **v0.2 — LLM integration**
  Input/output adapters that bridge the codec's continuous features with a speech-aware LLM. Streaming speech-to-speech inference.
- **v1.0 — End-to-end SDS**
  Dialogue management, multi-turn context, emotion/prosody control. The full vision of SoviaMate.

---

## Collaborate / research partnerships

Building a credible end-to-end spoken dialogue system from scratch needs more than code — it needs compute, datasets, and people. **We are actively looking for:**

- **Academic & industry collaborators** with expertise in speech codecs, speech LLMs, ASR/TTS, or dialogue systems.
- **Compute grants & sponsorships** for large-scale codec and LLM training (e.g., academic compute programs, cloud research credits, GPU partnerships).
- **Dataset partners** — multilingual conversational speech, real-world noisy recordings, expressive/emotional speech corpora.
- **Engineers and researchers** who want to own a piece of the stack (codec internals, LLM adapters, streaming runtime, evaluation harness).

If any of that fits you or your organization, please reach out: **[samson.ailabs@gmail.com](mailto:samson.ailabs@gmail.com)** with subject line `SoviaMate collaboration`. For code-level discussion, open a GitHub issue or discussion.

---

## Contributing

Code contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding style, and a list of good first contributions. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Citing this work

A technical report is in preparation. In the meantime, please cite the repository:

```bibtex
@misc{soviamate2026,
  author       = {Son Dang Dinh (Samson)},
  title        = {SoviaMate: Toward End-to-End Spoken Dialogue Systems},
  year         = {2026},
  howpublished = {\url{https://github.com/samson-ailabs/SoviaMate}},
}
```

A `CITATION.cff` is provided for GitHub's "Cite this repository" button.

---

## License & responsible use

SoviaMate is released under the [Apache License 2.0](LICENSE). It is intended for open research and beneficial applications of conversational AI.

The architecture supports zero-shot voice cloning. **It must not be used for impersonation, fraud, non-consensual voice synthesis, or any deceptive or harmful purpose.** Outputs may contain biases or inaccuracies inherited from training data; the authors accept no liability for downstream use. By using SoviaMate you agree to these terms and to applicable law in your jurisdiction.

---

## Acknowledgments

SoviaMate builds on a large body of public research in neural codecs, self-supervised speech models, ASR/TTS, and speech LLMs. The forthcoming technical report will include a full bibliography. Thanks to the open-source PyTorch, Lightning, Hydra, SentencePiece, and HuggingFace communities — this project would not be possible without them.

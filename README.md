# TrueTone: Audio Forgery Detection

Classifies an audio file as **Original**, **AI Generated** or **Combined**. (Combined means audios that both AI and original parts in it.)
> **Status: paused (November 2026)**
>
> - The model worked well on new audio in 2025. It gives 99% accuracy on test data and works well on the audios outside the dataset as well.
> - In 2026 it stopped working on newer audio.
> - Likely cause: the arms race between voice generation and voice detection. See [After the underlying patterns of audio creation started to change in the audio generation tools, the model stopped working on new audios.](#why-it-stopped-working).
> - I stopped development to focus on LLM applications. The code stays here as a record of what I built and what I learned.

## Live demo

https://bushrasaleem491-truetone-urdu.hf.space/

This demo runs [on version 1 using the previous dataset] and fail on new audio from newer voice tools.

## Dataset Link
https://www.kaggle.com/datasets/bushrasaleem020/features-of-truth-of-voice-in-urdu

## Overview

TrueTone was my Final Year Project for BS Information Technology at the University of Gujrat. It is an end-to-end pipeline for audio forgery detection in Urdu and English: dataset, preprocessing, augmentation, feature extraction, model training and deployment.

## Versions

| Version | What it is | Result |
|---|---|---|
| v1 | Final Year Project. Three classes. Dense neural network on 34 audio features. | 99% on 170 held-out files |
| v2 | Retrained as a binary classifier on clean synthetic speech from Azure Neural TTS. | 99% on its test set |

## Pipeline (v1)

**1. Dataset**
- Manually curated 1,530+ Urdu audio files.
- Split into Training (1,070), Validation (290) and Testing (170) files.
- Three categories: Original, AI Generated, Combined.

**2. Preprocessing**
- Converted all audio to WAV, 16 kHz, mono.
- Volume normalization with librosa.

**3. Data augmentation**
- Generated 10,710+ augmented files using 7 techniques: pitch shift, speed up (1.2x), slow down (0.8x), white noise, random volume change, reverb, band-pass filtering.
- Augmentation techniques were applied to all the training, testing and validation data.

**4. Feature extraction (34 features per file)**
- MFCC: 13
- Chroma: 12
- Spectral contrast: 7
- Zero crossing rate: 1
- RMSE: 1

**5. Model training**
- Architecture: dense neural network (256, 128, output)
- Optimizer: Adam
- Loss: sparse categorical crossentropy
- Epochs: 100

**6. Deployment**
- Converted to TensorFlow Lite for lightweight inference.
- WAs first deployed on Railway.
- Then deployed on Hugging Face Spaces with a Gradio interface.

## What went wrong in my own work

- The dataset I built had mislabeled classes.
- Some of my training audio clips came through WhatsApp. The compression erased the sound patterns the model needed.
- My inference code gave different answers for the same file, because of how I averaged features.

After I fixed the data and retrained on clean synthetic speech (v2), the model reached 99% on its test set. In 2025 it also worked on audio that was not in my dataset.

## Why it stopped working

In 2026, new audio started failing, and I kept looking for a bug in my own code. Then I searched and found that other people report the same pattern: an arms race between voice generation and voice detection. Voice tools keep changing their underlying patterns, so a detector trained on older patterns can fail on newer voice tools.

For example, Mao et al. (2025) report that detectors do well against older voice generators but much worse against newer, unseen ones: [Benchmarking Fake Voice Detection in the Fake Voice Generation Arms Race](https://arxiv.org/html/2510.06544v1).

I cannot prove this is exactly what broke my model. But the pattern matches what researchers describe.

## Limitations

- Small dataset: 1,530+ original files.
- The v1 test set is small (170 files), so its score is uncertain.
- The speakers were same in the training, testing and validation data.
- Does not work reliably on newer audio (2026).
- Not maintained.

## What I would do next

- Test on audio from newer voice tools every few months.
- Keep a fixed set of new-audio test clips to catch decay early.
- Retrain with audio from newer generators.

## What I learned

- Check your labels first. A model can learn the wrong thing very well.
- A model is never finished. When the world changes, a good score can go stale. Keep testing on new samples.
- Know when to stop. I chose to put my time into LLM applications.

## Tech stack

Python, TensorFlow, TensorFlow Lite, librosa, scikit-learn, pydub, scipy, Gradio, Hugging Face Spaces

## Author

Bushra Saleem

- GitHub: https://github.com/Bushra-491
- Hugging Face: https://huggingface.co/BushraSaleem491
- LinkedIn: https://www.linkedin.com/in/bushra-saleem-89ab03371/

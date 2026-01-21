# Test-Time Adaptation for Speech Emotion Recognition

[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-blue.svg)](https://2026.ieeeicassp.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of our paper accepted to **ICASSP 2026**: *Test-Time Adaptation Methods for Speech Emotion Recognition*.

## Overview

Speech emotion recognition (SER) systems often suffer from performance degradation when encountering domain shifts during deployment. This work provides a comprehensive evaluation of test-time adaptation (TTA) methods for SER, addressing three critical adaptation scenarios:

- **Task 1: Intra-corpus personalization** - Adapting to unseen speakers within the same dataset
- **Task 2: Acted to natural emotion adaptation** - Adapting from scripted to improvised speech
- **Task 3: Cross-Corpus generalization** - Adapting across different emotional speech datasets

We evaluate 11 state-of-the-art TTA methods on speech emotion recognition using pre-trained Wav2Vec2 models, providing insights into their effectiveness across different adaptation scenarios.

## Installation

### Requirements

- Python 3.8 or higher (tested with Python 3.11)
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotionTTA.git
cd emotionTTA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The datasets will be automatically downloaded from HuggingFace when you run the experiments for the first time.

## Usage

### Task 1: Speaker-Independent Emotion Recognition

#### IEMOCAP (Sessions 1-4 → Session 5)

```bash
python Task1/IEMOCAP.py --tta_method tent --batch_size 32
```

#### RAVDESS (Actors 1-16 → Actors 21-24)

```bash
python Task1/RAVDESS.py --tta_method tent --batch_size 32
```

### Task 2: Within-Corpus Domain Shift (Scripted → Improvised)

```bash
# Test on improvised speech after training on scripted speech
python Task2/IEMOCAP-task2.py --tta_method tent --batch_size 32
```

### Task 3: Cross-Corpus Adaptation

#### Test on IEMOCAP (trained on RAVDESS)

```bash
python Task3/crosscorpus-task3.py --test_dataset_name iemocap --tta_method tent --batch_size 32
```

#### Test on RAVDESS (trained on IEMOCAP)

```bash
python Task3/crosscorpus-task3.py --test_dataset_name ravdess --tta_method tent --batch_size 32
```

## Datasets

### IEMOCAP
- **Source**: [HuggingFace - Zahra99/IEMOCAP_Audio](https://huggingface.co/datasets/Zahra99/IEMOCAP_Audio)
- **Emotions**: Angry (ang), Happy (hap), Neutral (neu), Sad (sad)
- **Sessions**: 5 sessions, each from different actors
- **Automatically downloaded** when running experiments

### RAVDESS
- **Source**: [HuggingFace - xbgoose/ravdess](https://huggingface.co/datasets/xbgoose/ravdess)
- **Emotions**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Actors**: 24 professional actors
- **Automatically downloaded** when running experiments

## Pre-trained Models

The experiments use pre-trained Wav2Vec2 models fine-tuned on emotional speech:

### Models Available on HuggingFace

All models are automatically downloaded from HuggingFace Hub when running experiments:

- **Task 1 (IEMOCAP)**: `hongdage/wav2vec2-base-finetuned-iemocap`
- **Task 1 (RAVDESS)**: `LincolnD/wav2vec2-base-finetuned-ravdess-personalization`
- **Task 2 (IEMOCAP)**: `LincolnD/wav2vec2-base-finetuned-iemocap-acted-to-natural-emotion`
- **Task 3 (RAVDESS→IEMOCAP)**: `hongdage/back_iemocap`
- **Task 3 (IEMOCAP→RAVDESS)**: `hongdage/wav2vec2-base-ravdess22`

## Expected Output

After running an experiment, you will see:

```
=== FINAL RESULTS ===
Method: TENT
Accuracy: 0.XXXX
Macro F1: 0.XXXX
Evaluation completed!
```

Results are also saved to the console with detailed per-class metrics.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{yourname2026tta,
  title={Test-Time Adaptation Methods for Speech Emotion Recognition},
  author={Your Name and Co-authors},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [jiahengdong215@gmail.com](mailto:jiahengdong215@gmail.com).

---

**Note**: This is research code. For production use, additional engineering and testing are recommended.


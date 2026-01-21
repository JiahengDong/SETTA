# Guide: Uploading Models to HuggingFace

This guide explains how to upload your local model checkpoints to HuggingFace Hub so others can easily use them.

## Models to Upload

You have two local model checkpoints that should be uploaded:

1. **RAVDESS Task1 Model**: `wav2vec2-base-finetuned-ravdess-task1-val/`
   - Used in: `Task1/RAVDESS.py`
   - Current path: Local directory

2. **IEMOCAP Task2 Model**: `wav2vec2-base-finetuned-iemocap-task2/`
   - Used in: `Task2/IEMOCAP-task2.py`
   - Current path: Local directory

## Prerequisites

1. **HuggingFace Account**: Create one at [huggingface.co](https://huggingface.co/join)

2. **Install HuggingFace CLI**:
```bash
pip install huggingface_hub
```

3. **Login to HuggingFace**:
```bash
huggingface-cli login
```
This will prompt you for your token (get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

## Method 1: Using Python Script (Recommended)

Create and run this upload script:

```python
# upload_models.py
from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Your HuggingFace username
HF_USERNAME = "your-username"  # Change this to your HuggingFace username

# Model 1: RAVDESS Task1
model1_local_path = "./wav2vec2-base-finetuned-ravdess-task1-val"
model1_repo_name = f"{HF_USERNAME}/wav2vec2-base-finetuned-ravdess-task1"

# Model 2: IEMOCAP Task2
model2_local_path = "./wav2vec2-base-finetuned-iemocap-task2"
model2_repo_name = f"{HF_USERNAME}/wav2vec2-base-finetuned-iemocap-task2"

def upload_model(local_path, repo_id, model_description):
    """Upload a model to HuggingFace Hub."""
    api = HfApi()
    
    print(f"Creating repository: {repo_id}")
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    print(f"Uploading model from {local_path}...")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {model_description}"
    )
    
    print(f"✓ Successfully uploaded to: https://huggingface.co/{repo_id}")

# Upload both models
print("=" * 60)
print("Uploading Models to HuggingFace Hub")
print("=" * 60)

print("\n[1/2] Uploading RAVDESS Task1 Model...")
upload_model(
    model1_local_path, 
    model1_repo_name,
    "Wav2Vec2 fine-tuned on RAVDESS for Task1"
)

print("\n[2/2] Uploading IEMOCAP Task2 Model...")
upload_model(
    model2_local_path,
    model2_repo_name,
    "Wav2Vec2 fine-tuned on IEMOCAP for Task2"
)

print("\n" + "=" * 60)
print("All models uploaded successfully!")
print("=" * 60)
print("\nNext steps:")
print(f"1. Update Task1/RAVDESS.py with: {model1_repo_name}")
print(f"2. Update Task2/IEMOCAP-task2.py with: {model2_repo_name}")
print("\nView your models at:")
print(f"  - https://huggingface.co/{model1_repo_name}")
print(f"  - https://huggingface.co/{model2_repo_name}")
```

**Run the script:**
```bash
# First, update HF_USERNAME in the script
# Then run:
python upload_models.py
```

## Method 2: Using HuggingFace CLI

```bash
# Upload RAVDESS Task1 Model
huggingface-cli upload your-username/wav2vec2-base-finetuned-ravdess-task1 \
    ./wav2vec2-base-finetuned-ravdess-task1-val \
    --repo-type model

# Upload IEMOCAP Task2 Model
huggingface-cli upload your-username/wav2vec2-base-finetuned-iemocap-task2 \
    ./wav2vec2-base-finetuned-iemocap-task2 \
    --repo-type model
```

## Method 3: Using Web Interface (Manual)

1. Go to [huggingface.co/new](https://huggingface.co/new)
2. Create a new model repository
3. Clone the repository locally:
   ```bash
   git clone https://huggingface.co/your-username/model-name
   ```
4. Copy your model files into the cloned directory
5. Commit and push:
   ```bash
   cd model-name
   git add .
   git commit -m "Upload model"
   git push
   ```

## After Uploading: Update Your Code

Once uploaded, you need to update the model paths in your code:

### Update Task1/RAVDESS.py

**Before:**
```python
model_checkpoint = "/home/jiahengd/emotionTTA-main/wav2vec2-base-finetuned-ravdess-task1-val"
```

**After:**
```python
model_checkpoint = "your-username/wav2vec2-base-finetuned-ravdess-task1"
```

### Update Task2/IEMOCAP-task2.py

**Before:**
```python
model_checkpoint = "/home/jiahengd/emotionTTA-main/wav2vec2-base-finetuned-iemocap-task2"
```

**After:**
```python
model_checkpoint = "your-username/wav2vec2-base-finetuned-iemocap-task2"
```

## Adding Model Cards (Recommended)

Create a `README.md` file in each model repository to document your model:

```markdown
---
language: en
tags:
- audio
- speech-emotion-recognition
- wav2vec2
- RAVDESS
license: mit
datasets:
- xbgoose/ravdess
---

# Wav2Vec2 Fine-tuned on RAVDESS for Emotion Recognition

This model is a fine-tuned version of Facebook's Wav2Vec2 on the RAVDESS dataset for speech emotion recognition.

## Model Description

- **Base Model**: facebook/wav2vec2-base
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Task**: Speech Emotion Recognition
- **Emotions**: 8 classes (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Use Case**: Task 1 - Speaker-independent emotion recognition

## Training Details

- **Training Data**: RAVDESS Actors 1-16
- **Validation Data**: RAVDESS Actors 17-20
- **Test Data**: RAVDESS Actors 21-24

## Usage

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model_checkpoint = "your-username/wav2vec2-base-finetuned-ravdess-task1"
processor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
model = AutoModelForAudioClassification.from_pretrained(model_checkpoint)
```

## Citation

If you use this model, please cite our ICASSP 2026 paper:

```bibtex
@inproceedings{yourname2026tta,
  title={Test-Time Adaptation Methods for Speech Emotion Recognition},
  author={Your Name},
  booktitle={ICASSP},
  year={2026}
}
```

## Related Models

- [IEMOCAP Task2 Model](https://huggingface.co/your-username/wav2vec2-base-finetuned-iemocap-task2)

## License

MIT License
```

## Benefits of Using HuggingFace Hub

1. **Easy Sharing**: Anyone can use your model with just the model ID
2. **Automatic Caching**: Models are cached locally after first download
3. **Version Control**: Built-in versioning for model updates
4. **Model Cards**: Document your model with README files
5. **No Large Files in Git**: Keep your code repository clean
6. **DOI**: HuggingFace provides DOIs for citation

## Testing After Upload

Test that your models work correctly:

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Test RAVDESS model
model_id = "your-username/wav2vec2-base-finetuned-ravdess-task1"
processor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)
print(f"✓ RAVDESS model loaded: {model.config.num_labels} classes")

# Test IEMOCAP Task2 model
model_id = "your-username/wav2vec2-base-finetuned-iemocap-task2"
processor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)
print(f"✓ IEMOCAP Task2 model loaded: {model.config.num_labels} classes")
```

## Updating README.md

After uploading, update your main README.md to reflect the new model locations:

```markdown
## Pre-trained Models

All models are available on HuggingFace Hub:

- **Task 1 (IEMOCAP)**: `hongdage/wav2vec2-base-finetuned-iemocap`
- **Task 1 (RAVDESS)**: `your-username/wav2vec2-base-finetuned-ravdess-task1`
- **Task 2 (Domain Shift)**: `your-username/wav2vec2-base-finetuned-iemocap-task2`
- **Task 3 (RAVDESS→IEMOCAP)**: `hongdage/back_iemocap`
- **Task 3 (IEMOCAP→RAVDESS)**: `hongdage/wav2vec2-base-ravdess22`

All models are automatically downloaded when running experiments.
```

## Troubleshooting

### Large File Error
If you get errors about files > 5GB:
```bash
# Install git-lfs
git lfs install
# Track large files
git lfs track "*.safetensors"
git lfs track "*.bin"
```

### Authentication Issues
Make sure you're logged in:
```bash
huggingface-cli whoami
```

### Private vs Public
By default, models are public. To make private:
```python
create_repo(repo_id=repo_id, repo_type="model", private=True)
```

## Summary Checklist

- [ ] Create HuggingFace account
- [ ] Install huggingface_hub
- [ ] Login with `huggingface-cli login`
- [ ] Upload RAVDESS Task1 model
- [ ] Upload IEMOCAP Task2 model
- [ ] Add model cards (README.md) to each repository
- [ ] Update `Task1/RAVDESS.py` with new model path
- [ ] Update `Task2/IEMOCAP-task2.py` with new model path
- [ ] Test that models load correctly
- [ ] Update main README.md with new model links
- [ ] Optional: Remove local model directories from git

---

Need help? Check the [HuggingFace Hub documentation](https://huggingface.co/docs/hub/index)


#!/usr/bin/env python3
"""
IEMOCAP Emotion Recognition with Test-Time Adaptation (TTA)
Converted from Jupyter notebook for Spartan compatibility
"""

import os
import random
import numpy as np
import torch
import evaluate
from datasets import load_dataset, Audio, DatasetDict, ClassLabel
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from TTAs import tent
from TTAs import sar
from TTAs.sam import SAM
import math
import copy
# Configuration
model_checkpoint = "hongdage/wav2vec2-base-finetuned-iemocap"
# Configuration
iemocap_emotions = ["ang", "hap", "neu", "sad"]
max_duration = 8.0  # seconds


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True)
    
def load_and_prepare_dataset():
    """Load and prepare the IEMOCAP dataset"""
    print("Loading IEMOCAP dataset...")
    dataset = load_dataset("Zahra99/IEMOCAP_Audio")

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets([
    dataset["session1"],
    dataset["session2"],
    dataset["session3"],
    dataset["session4"],
    ])

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = dataset["session5"]
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    
    dataset1 = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    return dataset1

def setup_labels(dataset1):
    """Setup label mappings"""
    labels = dataset1["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    
    print("Label mappings created:")
    print(f"Dataset has {len(labels)} classes: {labels}")
    print(f"Label range: 0 to {len(labels)-1}")
    print("Sample label:", id2label["7"] if "7" in id2label else id2label[str(len(labels)-1)])
    
    return labels, label2id, id2label

def show_random_samples(dataset1, id2label, num_samples=5):
    """Display random samples from the dataset"""
    print(f"\nShowing {num_samples} random samples:")
    
    for _ in range(num_samples):
        rand_idx = random.randint(0, len(dataset1["train"])-1)
        example = dataset1["train"][rand_idx]
        audio = example["audio"]
        
        print(f'Label: {id2label[str(example["label"])]}')
        print(f'Shape: {audio["array"].shape}, sampling rate: {audio["sampling_rate"]}')
        print()

def create_preprocess_function(processor):
    """Create preprocessing function for audio data"""
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=processor.sampling_rate,
            padding=True,
            max_length=int(processor.sampling_rate * max_duration),
            truncation=True,
        )
        return inputs
    return preprocess_function

# Removed compute_metrics function - using sklearn metrics directly now

def load_pretrained_model():
    """Load pre-trained model and processor"""
    print("Loading pre-trained model...")
    
    processor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(model_checkpoint)
    
    print(f"Loaded model: {model_checkpoint}")
    print(f"Model config: {model.config}")
    print(f"Model expects {model.config.num_labels} classes")
    
    return processor, model

def run_tta_evaluation(model, processor, test_all_dataset, tta_method="tent", source_dataset=None):
    """Setup TTA and run evaluation using DataLoader (no Trainer)"""
    print(f"Setting up and running TTA evaluation with method: {tta_method}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create preprocessing function
    def preprocess_function_tta(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=processor.sampling_rate,
            padding=True,
            max_length=int(processor.sampling_rate * max_duration),
            truncation=True,
        )
        return inputs

    # Setup TTA model based on method
    if tta_method == "source":
        print("Using source model without TTA")
        tta_model = model
    elif tta_method == "tent":
        from TTAs import tent
        print("Setting up TENT TTA...")
        
        # TTA for model only
        net = tent.configure_model(model)
        params, param_names = tent.collect_params(net)
        optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        tta_model = tent.Tent(net, optimizer)
        
    elif tta_method == "ebats":
        import ebats
        print("Setting up EBATS TTA...")
        
        # TTA for model only
        net = ebats.configure_model(model)
        tta_model = ebats.EBATS(net, num_prompts=1, source_dataset=source_dataset)
    elif tta_method == "sar":
        from TTAs import sar
        print("Setting up SAR TTA...")
        
        # TTA for model only
        net = sar.configure_model(model)
        params, param_names = sar.collect_params(net)
        
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(params, base_optimizer, lr=1e-5, weight_decay=0., foreach=False)
        tta_model = sar.SAR(net, optimizer, margin_e0=0.4*math.log(4), reset_constant_em=0.2)
    elif tta_method == "eata":
        from TTAs import eata
        print("Setting up EATA TTA...")
        
        sub_net = eata.configure_model(model)
        params, param_names = eata.collect_params(sub_net)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        
        # Compute fisher matrices using test dataset (target domain)
        print("Computing fisher matrices using test dataset...")
        # Prepare test dataset
        encoded_test = test_all_dataset.map(preprocess_function_tta, remove_columns=["audio"], batched=True)
        encoded_test = encoded_test.rename_column("label", "labels")
        encoded_test.set_format(type="torch", columns=["input_values", "labels"])
        fisher_loader = DataLoader(encoded_test, batch_size=1, shuffle=False)  # Keep order for test data
        
        # Setup loss function
        train_loss_fn = torch.nn.CrossEntropyLoss().to(device)
        sub_net = sub_net.to(device)
        
        # Compute fisher matrices
        with torch.enable_grad():  # Enable gradients for fisher computation
            for iter_, batch in enumerate(tqdm(fisher_loader, desc="Computing fisher matrices"), start=1):
                input_values = batch["input_values"].to(device)
                
                # Get model outputs
                outputs = sub_net(input_values)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                
                # Use predicted labels (following original implementation)
                _, targets = outputs.max(1)
                
                # Compute loss and gradients
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                
                # Update fisher matrices
                for name, param in sub_net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                
                ewc_optimizer.zero_grad()
        
        print("Fisher matrices computation completed")
        del ewc_optimizer

        optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        tta_model = eata.EATA(sub_net, optimizer, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=0.4*math.log(4), d_margin=0.04, fishers=fishers)
    elif tta_method == "suta":
        from TTAs import suta
        print("Setting up SUTA TTA...")
        
        # TTA for model only
        net = suta.configure_model(model)
        params, param_names = suta.collect_params(net, train_feature=True, train_all=False, train_LN=True)
        
        optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        tta_model = suta.SUTA(net, optimizer, steps=10, episodic=True, em_coef=0.3, reweight=True, temp=1.5, repeat_inference=True)
    
    elif tta_method == "cotta":
        from TTAs import cotta
        print("Setting up CoTTA TTA...")
        
        # TTA for model only
        net = cotta.configure_model(model)
        params, param_names = cotta.collect_params(net)
        
        optimizer = torch.optim.AdamW(params, lr=2e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        tta_model = cotta.CoTTA(net, optimizer, steps=1, mt_alpha=0.99, rst_m=0.1, ap=0.9)
    elif tta_method == "cea":
        from TTAs import cea
        print("Setting up CEA TTA...")
        
        # TTA for model only
        net = cea.configure_model(model)
        params1, param_names1 = cea.collect_params(net, True, False, True)
        params2, param_names2 = cea.collect_params(net, False, False, True)
        params = [params1, params2]
        lrs = [1e-5, 1e-4]
        steps1 = 10
        steps2 = 10
        optimizer = cea.setup_optimizer(params, lrs)
        tta_model = cea.CEA(net, optimizer, model_base="wav2vec2", steps1=steps1, steps2=steps2, episodic=True, 
                            tc_coef=0.4, em_coef=0.3, temp=1.5)
    elif tta_method == "foa":
        from TTAs import foa
        print("Setting up FOA TTA...")
        
        if source_dataset is None:
            print("Warning: FOA requires source dataset for computing in-domain statistics")
        
        # Setup FOA with source dataset
        net = foa.configure_model(model)
        params, param_names = foa.collect_params(net)
        tta_model = foa.FOA(net, processor=processor, num_prompts=3, fitness_lambda=0.2, source_dataset=source_dataset, shift_vector_coef=0.5, cma_candidate_num=29)

    elif tta_method == "t3a":
        from TTAs import t3a
        print("Setting up T3A TTA...")
        
        net = t3a.configure_model(model)
        tta_model = t3a.T3A(net, num_classes=4, filter_K=64)
    
    elif tta_method == "lame":
        from TTAs import lame
        print("Setting up LAME TTA...")
        
        net = lame.configure_model(model)
        tta_model = lame.LAME(net, knn=5, sigma=1.0, affinity='kNN', force_symmetry=True)
    
    elif tta_method == "dsuta":
        from TTAs import dsuta
        print("Setting up DSUTA TTA...")
        net = dsuta.configure_model(model)
        fast_model = copy.deepcopy(net)
        slow_model = copy.deepcopy(net)
        param_fast, param_names_fast = dsuta.collect_params(fast_model, train_feature=True, train_all=False, train_LN=True)
        param_slow, param_names_slow = dsuta.collect_params(slow_model, train_feature=True, train_all=False, train_LN=True)
        fast_optimizer = torch.optim.AdamW(param_fast, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        slow_optimizer = torch.optim.AdamW(param_slow, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        tta_model = dsuta.DSUTA(fast_model, slow_model, fast_optimizer, slow_optimizer, update_freq=20, memory_size=20, adaptation_steps=10, temperature=1.5, entropy_weight=0.3, device='cuda', class_num=4)
    elif tta_method == "awmc":
        from TTAs import awmc
        print("Setting up AWMC TTA...")
        
        net = awmc.configure_model(model)
        anchor = copy.deepcopy(net)
        system = copy.deepcopy(net)
        leader = copy.deepcopy(net)
        param_anchor, param_names_anchor = awmc.collect_params(anchor, train_feature=False, train_all=False, train_LN=True, bitfit=True)
        param_leader, param_names_leader = awmc.collect_params(leader, train_feature=False, train_all=False, train_LN=True, bitfit=True)
        param_system, param_names_system = awmc.collect_params(system, train_feature=False, train_all=False, train_LN=True, bitfit=True)
        anchor_optimizer = torch.optim.AdamW(param_anchor, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        leader_optimizer = torch.optim.AdamW(param_leader, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        system_optimizer = torch.optim.AdamW(param_system, lr=1e-5, betas=(0.9, 0.999), weight_decay=0., foreach=False)
        anchor_opt_param_names = param_names_anchor
        alpha = 0.999
        steps = 10

        tta_model = awmc.AWMC(anchor, system, leader, device = 'cuda', anchor_optimizer = anchor_optimizer, leader_optimizer = leader_optimizer, system_optimizer = system_optimizer, anchor_opt_param_names = anchor_opt_param_names, alpha = alpha, steps = steps)

    else:
        raise ValueError(f"Unsupported TTA method: {tta_method}. Supported: ['source', 'tent', 'sar', 'suta', 'cotta', 'foa' ,'cea', 't3a', 'lame', 'eata', 'dsuta', 'awmc', 'ebats']")
    
    # Move model to device
    tta_model = tta_model.to(device)
    
    # Prepare test dataset
    print("Preprocessing test dataset...")
    encoded_dataset = test_all_dataset.map(preprocess_function_tta, remove_columns=["audio"], batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_values", "labels"])
    
    # Create DataLoader
    dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size, shuffle=False)  # Using command line batch size
    
    # Run evaluation
    print(f"Running evaluation with {tta_method}...")
    y_pred_list = []
    targets_list = []
    
    tta_model.eval()
    with torch.no_grad() if tta_method == "source" else torch.enable_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating with {tta_method}"):
            input_values = batch["input_values"].to(device)
            targets = batch["labels"].to(device)
            
            # Forward pass
            if tta_method == "source":
                outputs = tta_model(input_values)
                logits = outputs.logits
            else:
                # TTA methods return logits directly or wrapped outputs
                outputs = tta_model(input_values)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            y_pred_list.append(predictions.cpu())
            targets_list.append(targets.cpu())
    
    # Calculate metrics
    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    y_true = torch.cat(targets_list, dim=0).numpy()
    accuracy = accuracy_score(y_true, y_pred)
    detailed_metrics = classification_report(y_true, y_pred, digits=4, output_dict=True)
    
    print(f"{tta_method.upper()} evaluation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {detailed_metrics['macro avg']['f1-score']:.4f}")
    
    return {
        'eval_accuracy': accuracy,
        'eval_macro_f1': detailed_metrics['macro avg']['f1-score'],
        'detailed_metrics': detailed_metrics
    }



if __name__ == "__main__":
    import argparse
    
    # Set random seeds for reproducibility
    seed_everything(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="IEMOCAP Emotion Recognition with Test-Time Adaptation")
    parser.add_argument(
        "--tta_method", 
        type=str, 
        default="tent",
        choices=["source", "tent", "sar", "suta", "cotta", "foa" ,'cea', 't3a', 'lame', 'eata', 'dsuta', 'awmc', 'ebats'],
        help="TTA method to use: 'source' (no TTA), 'tent', 'sar', 'suta', 'cotta', 'foa', 'cea', 't3a', 'lame', 'eata', 'dsuta' or 'awmc'"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for TTA testing (default: 32)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting IEMOCAP emotion recognition with TTA method: {args.tta_method}")
    
    # Load and prepare dataset
    dataset0 = load_and_prepare_dataset()
    dataset1, test_all_dataset = dataset0, dataset0["test"]
    labels, label2id, id2label = setup_labels(dataset1)
    
    # Show some samples
    show_random_samples(dataset1, id2label)
    
    # Load pre-trained model and processor
    processor, model = load_pretrained_model()
    
    # Check label compatibility
    print(f"\nLabel Compatibility Check:")
    print(f"Dataset labels: {len(labels)} classes")
    print(f"Model expects: {model.config.num_labels} classes")
    
    if len(labels) != model.config.num_labels:
        print("⚠️  MISMATCH DETECTED!")
        print("This will cause the CUDA assertion error.")
        print("Solutions:")
        print("1. Use a model trained on IEMOCAP")
        print("2. Remap labels to match model's expected classes")
        exit(1)
    
    # Prepare source dataset if needed for FOA
    source_dataset = None
    if args.tta_method == "foa" or args.tta_method == "ebats":
        print("Preparing source dataset for FOA...")
        # Create preprocessing function for source data
        def preprocess_function_source(examples):
            audio_arrays = [x["array"] for x in examples["audio"]]
            inputs = processor(
                audio_arrays,
                sampling_rate=processor.sampling_rate,
                padding=True,
                max_length=int(processor.sampling_rate * max_duration),
                truncation=True,
            )
            return inputs
        
        # Use training data as source dataset (actors 1-16)
        source_raw = dataset1["train"]  # This is already filtered to actors 1-16
        encoded_source = source_raw.map(preprocess_function_source, remove_columns=["audio"], batched=True)
        encoded_source = encoded_source.rename_column("label", "labels")
        encoded_source.set_format(type="torch", columns=["input_values", "labels"])
        source_dataset = encoded_source
        print(f"Source dataset prepared with {len(source_dataset)} samples")

    # Setup and run evaluation with specified method
    tta_results = run_tta_evaluation(model, processor, test_all_dataset, tta_method=args.tta_method, source_dataset=source_dataset)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Method: {args.tta_method.upper()}")
    print(f"Accuracy: {tta_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Macro F1: {tta_results.get('eval_macro_f1', 'N/A'):.4f}")
    print("Evaluation completed!")

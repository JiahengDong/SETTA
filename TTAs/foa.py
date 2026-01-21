import os 
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import random
from torch import nn
import cma
import math
import numpy as np

def set_seed(seed=42):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

class FOA(nn.Module):
    def __init__(self, model, num_prompts, processor, fitness_lambda=0.2, source_dataset=None, device='cuda', shift_vector_coef=1.0, cma_candidate_num=27):
        """Initialize FOA with model and adaptation parameters"""
        super(FOA, self).__init__()
        self.device = device
        self.model = model
        self.model.requires_grad_(False)
        
        # For Wav2Vec2-based emotion models, use wav2vec2 config
        if hasattr(self.model, 'wav2vec2'):
            self.prompt_dim = self.model.wav2vec2.config.hidden_size
            self.config = self.model.wav2vec2.config
        elif hasattr(self.model, 'config'):
            self.prompt_dim = self.model.config.hidden_size
            self.config = self.model.config
        else:
            # Fallback
            self.prompt_dim = 768
            self.config = None
            
        self.num_prompts = num_prompts
        self.layer_norm = nn.LayerNorm(self.prompt_dim).cuda()
        self.project_dim = nn.Linear(self.prompt_dim, self.prompt_dim).cuda()
        self.best_loss = np.inf
        self.hist_stat = None
        self.processor = processor
        self.encoder_source_hidden = []
        self.encoder_target_hidden = []
        self.train_info = self.compute_in_domain_statistics(source_dataset)
        self.fitness_lambda = fitness_lambda
        self.shift_vector_coef = shift_vector_coef
        self.cma_candidate_num = cma_candidate_num
        # Xavier uniform initialization
        fan_in = self.prompt_dim
        fan_out = self.prompt_dim
        val = math.sqrt(6. / float(fan_in + fan_out))
        self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
        nn.init.uniform_(self.prompts.data, -val, val)
        self.best_prompts = self.prompts
        # Ensure prompt_dim matches the hidden size of Wav2Vec2
        #assert self.model.wav2vec2.config.conv_dim[-1] == self.prompt_dim, "Prompt dimension must match the extracted feature embedding of the Wav2Vec2 model."
        self.es = self.init_cma()

    def init_promts(self):
        # Assuming fan_in could be the size of the input features (e.g., 80 for log-mel spectrograms)
        # fan_out is the prompt_dim, related to the transformer hidden size
        fan_in = self.model.wav2vec2.config.conv_dim[-1] 
        fan_out = self.model.wav2vec2.config.hidden_size
    
        # Xavier uniform initialization adapted for Wav2Vec 2.0
        val = math.sqrt(6. / float(fan_in + fan_out))
        self.prompts = nn.Parameter(torch.zeros(1, self.num_prompts, self.prompt_dim))
        nn.init.uniform_(self.prompts.data, -val, val)
        self.best_prompts = self.prompts

    def init_cma(self):
        """CMA-ES initialization"""
        dim = self.prompts.numel()
        popsize = self.cma_candidate_num # which is equal to 4 + 3 * np.log(dim) when #prompts=3
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(dim * [0], 0.1, inopts=cma_opts)
        self.popsize = es.popsize
        return es

    def compute_in_domain_statistics(self, source_dataset):
        """Compute source domain statistics from encoded emotion dataset"""
        if source_dataset is None:
            print("Warning: No source dataset provided for computing in-domain statistics")
            # Return dummy statistics - will be replaced with proper source data
            dummy_stats = torch.zeros(self.prompt_dim * 13).cuda()  # Assuming 13 layers like original FOA
            return [dummy_stats, torch.ones_like(dummy_stats)]
        
        accumulated_hidden_states = None
        model = self.model.wav2vec2.to(self.device)  # Ensure model is on correct device
        num_batches = 0

        # Create DataLoader for the source dataset
        from torch.utils.data import DataLoader
        dataloader = DataLoader(source_dataset, batch_size=1, shuffle=False)  # Use small batch size
        
        for i, batch in enumerate(dataloader):
            if i > 32:  # Limit to 32 batches for efficiency
                break
            num_batches += 1
            
            # Extract input_values from the batch (already preprocessed)
            input_values = batch["input_values"].cuda()

            with torch.no_grad():
                # Extract final hidden states
                outputs = model(input_values, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Shape: (N, B, L, D)
                last_hidden = outputs.last_hidden_state
                processed_last_hidden = last_hidden.mean(dim=1)  # Pool over sequence length
                processed_hidden_states = torch.cat([hidden_state.mean(dim=1) for hidden_state in hidden_states], dim=1) #shape (B, N*D)
            
            # Accumulate the sum of hidden states across batches
            if accumulated_hidden_states is None:
                accumulated_hidden_states = processed_hidden_states
            else:
                accumulated_hidden_states += processed_hidden_states
            
            # Store individual samples for analysis (optional)
            for sample_idx in range(processed_last_hidden.shape[0]):
                self.encoder_source_hidden.append(processed_last_hidden[sample_idx].cpu().numpy())
        
        if num_batches == 0:
            print("Warning: No batches processed from source dataset")
            dummy_stats = torch.zeros(self.prompt_dim * 13).cuda()
            return [dummy_stats, torch.ones_like(dummy_stats)]
        
        # Concatenate all hidden states across batches
        all_hidden_states = accumulated_hidden_states / num_batches  # Shape: (B, N*D)

        # Calculate batch statistics
        batch_std, batch_mean = torch.std_mean(all_hidden_states, dim=0, unbiased=False)
        return [batch_mean, batch_std]


    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean

    def get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[0][-768:] - self.hist_stat

    def prompt_injection(self, features):
        """
        Inject prompts into the feature sequence.
        """
        batch_size = features.size(0)
        expanded_prompts = self.prompts.expand(batch_size, -1, -1)
        features_with_prompts = torch.cat((expanded_prompts, features), dim=1)
        return features_with_prompts

    def __call__(self, x):
        """Override the call method to ensure forward_and_adapt is called first"""
        # First run adaptation
        outputs = self.forward(x)
        
        return outputs

    def _forward_impl(self, input_values):
        """Internal implementation of forward pass with prompt injection"""
        # Ensure input_values is on the same device as the model
        input_values = input_values.to(next(self.model.parameters()).device)
        
        self.model.wav2vec2.config.output_hidden_states = True
        # Feature extraction
        features = self.model.wav2vec2.feature_extractor(input_values)
        features = features.transpose(1, 2)
        
        hidden, features = self.model.wav2vec2.feature_projection(features)
        hidden_states = self.prompt_injection(hidden)
        # Transformer encoder - uses the same number of layers as the original Wav2Vec2 model
        with torch.no_grad():
           transformer_output = self.model.wav2vec2.encoder(hidden_states, output_hidden_states=True)
           all_hidden_states = transformer_output.hidden_states
           final_hidden_state = transformer_output.last_hidden_state

        # For emotion classification, we need to pool the sequence and use classifier
        if hasattr(self.model, 'classifier'):
            # Pool over sequence dimension for classification
            pooled_output = final_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            pooled_output = self.model.projector(pooled_output)
            logits = self.model.classifier(pooled_output)
            return logits, final_hidden_state, all_hidden_states
        elif hasattr(self.model, 'lm_head'):
            # Fallback for CTC-style models (speech recognition)
            logits = self.model.lm_head(final_hidden_state)
            return logits, final_hidden_state, all_hidden_states
        else:
            # Create emotion classifier if none exists
            if not hasattr(self, 'emotion_classifier'):
                self.emotion_classifier = nn.Linear(final_hidden_state.size(-1), 8).cuda()
            pooled_output = final_hidden_state.mean(dim=1)
            pooled_output = self.model.projector(pooled_output)
            logits = self.emotion_classifier(pooled_output)
            return logits, final_hidden_state, all_hidden_states


    def forward(self, x):
        """Forward pass with FOA adaptation
        
        This is the main forward method that performs adaptation using CMA-ES
        and returns the best outputs.
        """
        shift_vector = self.get_shift_vector()

        self.best_loss, best_outputs, batch_means = np.inf, None, []
        final_hidden_state = None

        # Sampling from CMA-ES and evaluate the new solutions
        # Note that we also compare the current solutions with the previous best one
        prompts, losses = self.es.ask() + [self.best_prompts.flatten().cpu()], []
        for j, prompt in enumerate(prompts):
            self.prompts = torch.nn.Parameter(torch.tensor(prompt, dtype=torch.float).reshape_as(self.prompts).cuda())
            self.prompts.requires_grad_(False)
            outputs, loss, batch_mean, processed_final_hidden = forward_and_get_loss(x, self, shift_vector, self.fitness_lambda)
            batch_means.append(batch_mean[-768:].unsqueeze(0))
            del batch_mean

            if self.best_loss > loss.item():
                self.best_prompts = self.prompts
                self.best_loss = loss.item()
                best_outputs = outputs
                outputs = None
                final_hidden_state = processed_final_hidden
            losses.append(loss.item())
            del outputs

            #print(f'Solution:[{j+1}/{len(prompts)}], Loss: {loss.item()}')

        # CMA-ES updates, Eqn. (6)
        prompts = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in prompts]
        self.es.tell(prompts, losses)
            
        # Update overall test statistics, Eqn. (9)
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        
        if final_hidden_state is not None:
            self.encoder_target_hidden.append(final_hidden_state.squeeze(0).cpu().numpy())
        return best_outputs

def softmax_entropy(x, dim=-1):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def forward_and_get_loss(x, model, shift_vector, fitness_lambda):
    """
    Calculate the fitness value based on activation statistics and entropy, incorporating historical statistics and activation shifting.
    :param output: Model output for the batch.
    :param hist_stat: Running mean and std of historical statistics.
    :param train_info: Mean and Standard deviation of the source domain's activation statistics.
    :return: Fitness value to guide CMA-ES.
    """
    #check_outputs = model.model(x).logits
    #print(torch.argmax(check_outputs, dim=-1))

    outputs, final_hidden_state, all_hidden_states = model._forward_impl(x)
    processed_hidden_states = torch.cat([hidden_state.mean(dim=1) for hidden_state in all_hidden_states], dim=1)
    processed_final_hidden = final_hidden_state.mean(dim=1)
    source_mean = model.train_info[0]
    source_std = model.train_info[1]
    source_mean = source_mean.cuda()
    source_std = source_std.cuda()

    # Calculate batch statistics
    criterion_mse = nn.MSELoss(reduction='none').cuda()
    batch_std, batch_mean = torch.std_mean(processed_hidden_states, dim=0, unbiased=False)
    std_mse, mean_mse = criterion_mse(batch_std, source_std), criterion_mse(batch_mean, source_mean)

    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum())/32
    #print("discrepency loss", discrepancy_loss)

    entropy_loss = softmax_entropy(outputs, dim=1).mean()  # Average over batch
    #print("entropy_loss (classification):", entropy_loss)
        
    """entropy loss for Eqn. (5)"""
    loss = discrepancy_loss + entropy_loss
        
    """activation shifting, Eqn. (7)"""
    if shift_vector is not None:
        # For classification, apply shift to pooled features
        pooled_shifted = processed_final_hidden + model.shift_vector_coef * shift_vector
        if hasattr(model.model, 'classifier'):
            pooled_shifted = model.model.projector(pooled_shifted)
            outputs = model.model.classifier(pooled_shifted)
        elif hasattr(model, 'emotion_classifier'):
            pooled_shifted = model.emotion_classifier(pooled_shifted)
            outputs = model.emotion_classifier(pooled_shifted)

    return outputs, loss, batch_mean, processed_final_hidden

def configure_model(model):
    """Configure model for use with FOA."""
    # FOA doesn't modify the model parameters directly, just adds prompts
    model.requires_grad_(False)
    return model

def collect_params(model):
    """FOA doesn't optimize model parameters, so return empty lists."""
    return [], []


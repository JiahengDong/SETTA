import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


class Buffer:
    """Buffer to store and manage adaptation data"""
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.data = []

    def update(self, x):
        self.data.append(x)
        if len(self.data) > self.max_size:
            self.data.pop(0)
    
    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)

class DSUTA(nn.Module):
    """
    Dual-Stream Test-Time Adaptation (DSUTA) for emotion recognition
    Maintains both fast and slow adaptation streams
    """
    def __init__(self, 
                 fast_model,
                 slow_model,
                 fast_optimizer,
                 slow_optimizer,
                 update_freq=5,
                 memory_size=5,
                 adaptation_steps=10,
                 temperature=2.5,
                 entropy_weight=0.3,
                 reweight=True,
                 device='cuda',
                 class_num=8):
        super().__init__()
        self.device = device
        
        # Initialize models
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.fast_optimizer = fast_optimizer
        self.slow_optimizer = slow_optimizer

        # Save initial state
        self.initial_model_state = copy.deepcopy(fast_model.state_dict())
        self.initial_optimizer_state = copy.deepcopy(fast_optimizer.state_dict())
        self.initial_slow_model_state = copy.deepcopy(slow_model.state_dict())
        self.initial_slow_optimizer_state = copy.deepcopy(slow_optimizer.state_dict())
        
        # Adaptation hyperparameters
        self.update_freq = update_freq
        self.adaptation_steps = adaptation_steps
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.class_num = class_num
        self.reweight = reweight
        
        # Memory buffer for slow adaptation
        self.memory = Buffer(max_size=memory_size)
        
        # Tracking variables
        self.timestep = 0
        self.adapt_count = 0
        self.expected_batch_size = None  # Will be set from first batch

    def entropy_loss(self, x, dim=1):
        # Entropy of softmax distribution from logits
        # For emotion detection: x is typically (batch_size, num_classes)
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

    def mcc_loss(self, x, reweight=False, dim=1):
        # For emotion detection: x is typically (batch_size, num_classes)
        # MCC loss encourages diverse predictions across the batch
        p = x.softmax(dim) # (batch_size, num_classes)
        
        if reweight:
            # Reweight samples by their prediction entropy
            target_entropy_weight = self.entropy_loss(x, dim=dim).detach() # (batch_size,)
            target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
            target_entropy_weight = x.shape[0] * target_entropy_weight / torch.sum(target_entropy_weight)
            # Weight each sample's contribution to covariance
            weighted_p = p * target_entropy_weight.unsqueeze(1) # (batch_size, num_classes)
            cov_matrix_t = weighted_p.transpose(1, 0).mm(p) # (num_classes, batch_size) * (batch_size, num_classes) -> (num_classes, num_classes)
        else:    
            cov_matrix_t = p.transpose(1, 0).mm(p) # (num_classes, batch_size) * (batch_size, num_classes) -> (num_classes, num_classes)

        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1, keepdim=True)
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / self.class_num
    
        return mcc_loss

    def adapt_fast_model(self, x):
        """Adapt fast model on current batch"""
        self.fast_model.eval()
        total_loss = 0
        
        for _ in range(self.adaptation_steps):
            loss = 0
            
            # Forward pass
            logits = self.fast_model(x).logits
            
            # Calculate losses
            entropy = self.entropy_loss(logits/self.temperature).mean()
            consistency = self.mcc_loss(logits/self.temperature, self.reweight)
            
            # Combine losses
            loss += self.entropy_weight * entropy + (1 - self.entropy_weight) * consistency
            
            # Backward pass
            loss.backward()
            self.fast_optimizer.step()
            self.fast_model.zero_grad()
            total_loss += loss.item()
            
        return total_loss

    def adapt_slow_model(self):
        """Adapt slow model on memory buffer"""
        if len(self.memory) == 0:
            return 0.0
            
        total_loss = 0
        self.slow_model.zero_grad()
        # Stack memory data
        memory_data = torch.stack(self.memory.data).to(self.device)
        
        denom_scale = len(memory_data) // 1
        for x in memory_data:
            logits = self.slow_model(x).logits
            entropy = self.entropy_loss(logits/self.temperature).mean()
            consistency = self.mcc_loss(logits/self.temperature, self.reweight)
            loss = self.entropy_weight * entropy + (1 - self.entropy_weight) * consistency
            loss = loss / denom_scale
            loss.backward()
            total_loss += loss.item()
        
        self.slow_optimizer.step()
        self.slow_model.zero_grad()

        return total_loss   

    def forward(self, x):
        """
        Forward pass with adaptation
        Args:
            x: Input tensor
        Returns:
            Predictions from the adapted model
        """
        x = x.to(self.device)
        
        # Initialize models if first step
        self.fast_model.load_state_dict(self.initial_model_state)
        self.fast_optimizer.load_state_dict(self.initial_optimizer_state)
        
        # Fast adaptation
        self.adapt_fast_model(x)
        
        # Get predictions from fast model
        with torch.no_grad():
            predictions = self.fast_model(x).logits
        
        # Update memory and slow model (only for full-size batches)
        # Set expected batch size from first batch
        if self.expected_batch_size is None:
            self.expected_batch_size = x.size(0)
        
        # Only store batches that match the expected size (skip incomplete last batches)
        if x.size(0) == self.expected_batch_size:
            self.memory.update(x.detach().cpu())
        
        if (self.timestep + 1) % self.update_freq == 0:
            # Adapt slow model and update fast model's initial state
            self.slow_model.load_state_dict(self.initial_slow_model_state)
            self.slow_optimizer.load_state_dict(self.initial_slow_optimizer_state)
            self.slow_model.eval()
            self.adapt_slow_model()
            self.initial_slow_model_state = copy.deepcopy(self.slow_model.state_dict())
            self.initial_slow_optimizer_state = copy.deepcopy(self.slow_optimizer.state_dict())
            self.memory.clear()
            self.initial_model_state = copy.deepcopy(self.initial_slow_model_state)
            self.initial_optimizer_state = copy.deepcopy(self.initial_slow_optimizer_state)
            
        self.timestep += 1
        self.adapt_count += 1
        
        return predictions

    def get_adapt_count(self):
        """Return total adaptation count"""
        return self.adapt_count

def collect_params(model, bias_only=False, train_feature=False, train_all=False, train_LN=True):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    trainable = []
    if bias_only:
        trainable = ['bias']
    else: 
        trainable = ['weight', 'bias']

    
    for nm, m in model.named_modules():
        print(nm)
        if train_LN: 
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if train_feature:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                    for np, p in m.named_parameters():
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_all: 
            for np, p in m.named_parameters():
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")
            

    return params, names

def configure_model(model):
    """Configure model for use with tent."""
    model.eval()
    model.requires_grad_(False)
    return model

def batchify(data, batch_size, shuffle=False):
    """
    Batch generator for list data.
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    if shuffle:  # Shuffle at the start of epoch
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch_data = [data[idx] for idx in batch_idx]
        yield batch_data
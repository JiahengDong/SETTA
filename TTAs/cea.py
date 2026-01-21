import os 
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class CEA(nn.Module):
    """CEA adapts a model by entropy minimization and temporal coherence during testing.
    Once adapted, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, model_base, steps1=10, steps2=0, episodic=False, 
                 tc_coef=0.3, em_coef=0.3, temp=2.5):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.model_base = model_base
        self.steps1 = steps1
        self.steps2 = steps2
        self.total_steps = steps1 + steps2
        assert self.total_steps > 0, "CEA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        # CEA specific parameters
        self.tc_coef = tc_coef
        self.em_coef = em_coef
        self.temp = temp
        self.current_step = 0

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
            self.current_step = 0

        for step in range(self.total_steps):
            outputs = forward_and_adapt(
                x, self.model, self.model_base, self.optimizer, 
                tc_coef=self.tc_coef,
                em_coef=self.em_coef,
                step=step,
                step1=self.steps1,
                temp=self.temp
            )
            self.current_step = step

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, 
            self.model_state, self.optimizer_state
        )
        self.current_step = 0


def setup_optimizer(params=[], lrs=[], weight_decay=0., opt_name='AdamW'):
    opt = getattr(torch.optim, opt_name)
    optimizer = [opt(p, lr=lr, weight_decay=weight_decay, foreach=False) for p, lr in zip(params, lrs)]
    print(f'[INFO]    optimizer: {opt}')

    return optimizer


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = [deepcopy(optimizer[i].state_dict()) for i in range(len(optimizer))]

    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    for i in range(len(optimizer)):
        optimizer[i].load_state_dict(optimizer_state[i])
   
    return model, optimizer

    
def configure_model(model):
    model.requires_grad_(False)
    return model


def collect_params(model, train_feature=False, train_all=False, train_LN=True, bias_only=False):
    """ Collect trainable parameters """

    params = []
    names = []
    trainable = ['weight', 'bias']
    if bias_only:
        trainable = ['bias']

    for nm, m in model.named_modules():
        if train_LN: 
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                for np, p in m.named_parameters():
                    if f"{nm}.{np}" in names:
                            continue
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_feature:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                    for np, p in m.named_parameters():
                        if f"{nm}.{np}" in names:
                            continue
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_all: 
            for np, p in m.named_parameters():
                if f"{nm}.{np}" in names:
                    continue
                if np in trainable:  
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")
            
    return params, names





def softmax_entropy(x, dim=1):
    # Entropy of softmax distribution from logits
    # For emotion detection: x is typically (batch_size, num_classes)
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def mcc_loss(x, reweight=False, dim=1, class_num=8):
    # For emotion detection: x is typically (batch_size, num_classes)
    # MCC loss encourages diverse predictions across the batch
    p = x.softmax(dim) # (batch_size, num_classes)
    
    if reweight:
        # Reweight samples by their prediction entropy
        target_entropy_weight = softmax_entropy(x, dim=dim).detach() # (batch_size,)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = x.shape[0] * target_entropy_weight / torch.sum(target_entropy_weight)
        # Weight each sample's contribution to covariance
        weighted_p = p * target_entropy_weight.unsqueeze(1) # (batch_size, num_classes)
        cov_matrix_t = weighted_p.transpose(1, 0).mm(p) # (num_classes, batch_size) * (batch_size, num_classes) -> (num_classes, num_classes)
    else:    
        cov_matrix_t = p.transpose(1, 0).mm(p) # (num_classes, batch_size) * (batch_size, num_classes) -> (num_classes, num_classes)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1, keepdim=True)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
   
    return mcc_loss


def tc_reg_loss(x):
    # Temporal coherence regularization for emotion detection
    # For emotion detection, we encourage consistency in feature representations
    # x is feature tensor from feature extractor, typically (batch_size, seq_len, hidden_dim)
    
    if x.dim() == 3:
        batch_size, seq_len, hidden_dim = x.shape
        if seq_len <= 1:
            return torch.tensor(0.0, device=x.device)
        
        # Compute pairwise similarities between adjacent time steps
        x_current = x[:, :-1, :]  # (batch_size, seq_len-1, hidden_dim)
        x_next = x[:, 1:, :]      # (batch_size, seq_len-1, hidden_dim)
        
        # L2 distance between adjacent time steps (encourage temporal smoothness)
        tc_loss = torch.norm(x_current - x_next, p=2, dim=-1).mean()
        
    else:
        # If features are not sequential, return zero loss
        tc_loss = torch.tensor(0.0, device=x.device)

    return tc_loss


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, model_base, optimizer, tc_coef=0.3, em_coef=0.3, step=0, step1=0, temp=1.):
    """Forward and adapt model for emotion detection."""

    loss = 0

    # Get model outputs
    if hasattr(model(x), 'logits'):
        outputs = model(x).logits  # For transformers models
    else:
        outputs = model(x)  # For direct output models

    if step < step1:
        # Phase 1: Focus on entropy minimization and MCC loss
        if em_coef > 0: 
            # Entropy minimization - encourage confident predictions
            e_loss = softmax_entropy(outputs / temp, dim=1)
            # Weight by inverse entropy (more weight to uncertain predictions)
            weight = 1 / (1 + torch.exp(-e_loss))
            e_loss = (weight * e_loss).mean()
            loss += e_loss * em_coef
            
        if 1 - em_coef > 0: 
            # MCC loss - encourage diverse predictions across batch
            c_loss = mcc_loss(outputs / temp, reweight=True, dim=1, class_num=outputs.shape[-1])
            loss += c_loss * (1 - em_coef)
    
        model.zero_grad()
        loss.backward()
        optimizer[0].step()

    else:
        # Phase 2: Add temporal coherence regularization
        # Extract features for temporal coherence
        feats = None
        if 'wav2vec2' in model_base:
            print("Using wav2vec2 feature extractor, step2 adaptation")
            if hasattr(model, 'wav2vec2') and hasattr(model.wav2vec2, 'feature_extractor'):
                feats = model.wav2vec2.feature_extractor(x)
        elif 'hubert' in model_base:
            if hasattr(model, 'hubert') and hasattr(model.hubert, 'feature_extractor'):
                feats = model.hubert.feature_extractor(x)
        elif 'wavlm' in model_base:
            if hasattr(model, 'wavlm') and hasattr(model.wavlm, 'feature_extractor'):
                feats = model.wavlm.feature_extractor(x)

        if em_coef > 0:     
            # Entropy minimization
            e_loss = softmax_entropy(outputs / temp, dim=1).mean()
            loss += e_loss * em_coef
            
        if 1 - em_coef > 0: 
            # MCC loss
            c_loss = mcc_loss(outputs / temp, reweight=True, dim=1, class_num=outputs.shape[-1])
            loss += c_loss * (1 - em_coef)
        
        # Temporal coherence loss (only if features are available)
        if feats is not None and tc_coef > 0:
            tc_loss = tc_reg_loss(feats)
            loss += tc_coef * tc_loss

        model.zero_grad()
        loss.backward()
        optimizer[1].step()

    # Final inference
    with torch.no_grad():
        if hasattr(model(x), 'logits'):
            outputs = model(x).logits
        else:
            outputs = model(x)
            
    return outputs
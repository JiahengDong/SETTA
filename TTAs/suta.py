import os 
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch import nn
from jiwer import wer
import torch.nn.functional as F
from copy import deepcopy


class SUTA(nn.Module):
    """SUTA adapts a model by entropy minimization and MCC loss during testing.
    Once adapted, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=10, episodic=False, 
                 em_coef=0.9, reweight=False, temp=2.5, repeat_inference=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SUTA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        # SUTA specific parameters
        self.em_coef = em_coef
        self.reweight = reweight
        self.temp = temp
        self.repeat_inference = repeat_inference

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(
                x, self.model, self.optimizer, 
                em_coef=self.em_coef,
                reweight=self.reweight,
                temp=self.temp,
                repeat_inference=self.repeat_inference
            )

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, 
            self.model_state, self.optimizer_state
        )


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0.):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    return optimizer


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


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    

def configure_model(model):
    """Configure model for use with tent."""
    model.eval()
    model.requires_grad_(False)
    return model

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, em_coef=0.9, reweight=False, temp=1., repeat_inference=True):
    """Forward and adapt model on batch of data for emotion detection.

    Measure entropy of the model prediction, take gradients, and update params.
    
    For emotion detection: outputs are (batch_size, num_classes)
    """
    # forward
    if hasattr(model(x), 'logits'):
        outputs = model(x).logits  # For transformers models
    else:
        outputs = model(x)  # For direct output models
    
    # adapt
    loss = 0

    if em_coef > 0: 
        # Entropy minimization - encourage confident predictions
        e_loss = softmax_entropy(outputs / temp, dim=1).mean()
        #print(f"Entropy loss: {e_loss}")
        loss += e_loss * em_coef

    if 1 - em_coef > 0: 
        # MCC loss - encourage diverse predictions across batch
        c_loss = mcc_loss(outputs / temp, reweight, dim=1, class_num=outputs.shape[-1])
        #print(f"MCC loss: {c_loss}")
        loss += c_loss * (1 - em_coef)

    loss.backward()
    optimizer.step()
    model.zero_grad()

    # inference again
    if repeat_inference:
        with torch.no_grad():
            if hasattr(model(x), 'logits'):
                outputs = model(x).logits
            else:
                outputs = model(x)
    return outputs
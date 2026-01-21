import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class AWMC(nn.Module):
    """
    Dual-Stream Test-Time Adaptation (DSUTA) for emotion recognition
    Maintains both fast and slow adaptation streams
    """
    def __init__(self, 
                anchor,
                leader,
                system,
                device,
                anchor_optimizer,
                leader_optimizer,
                system_optimizer,
                anchor_opt_param_names,
                alpha,
                steps
                ):
        super().__init__()
        self.device = device
        
        # Initialize models
        self.anchor = anchor
        self.leader = leader
        self.system = system
        self.anchor_optimizer = anchor_optimizer
        self.leader_optimizer = leader_optimizer
        self.system_optimizer = system_optimizer
        self.system.eval()
        self.anchor.eval()
        self.leader.eval()
        for param in self.anchor.parameters():
            param.detach_()
        for param in self.leader.parameters():
            param.detach_()
        # Save initial state and move to correct device
        self.initial_system_model_state = {k: v.to(device) for k, v in copy.deepcopy(system.state_dict()).items()}
        self.initial_system_optimizer_state = copy.deepcopy(system_optimizer.state_dict())
        self.initial_leader_model_state = {k: v.to(device) for k, v in copy.deepcopy(leader.state_dict()).items()}
        self.initial_leader_optimizer_state = copy.deepcopy(leader_optimizer.state_dict())
        # Adaptation hyperparameters
        self.ema_task_vector = None
        self.alpha = alpha
        self.opt_param_names = anchor_opt_param_names
        self.steps = steps

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

    @torch.no_grad()
    def _update_leader(self):
        origin_model_state = self.initial_system_model_state  
        task_vector = self._get_task_vector(leader=False)

        if self.ema_task_vector is None:
            self.ema_task_vector = {}
            for name in self.opt_param_names:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in self.opt_param_names:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        # add back to origin model
        merged_model_state = {}
        for name in origin_model_state:
            if name in self.opt_param_names:
                merged_model_state[name] = origin_model_state[name] + self.ema_task_vector[name]
            else:
                merged_model_state[name] = origin_model_state[name]
        
        self.initial_leader_model_state = merged_model_state
        self.leader.load_state_dict(merged_model_state)

    @torch.no_grad()
    def _get_task_vector(self, leader=False):
        if leader:
            model_state = self.leader.state_dict()
        else:
            model_state = self.system.state_dict()
        origin_model_state = self.initial_system_model_state
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in self.opt_param_names}
        return task_vector
    
    def _update(self, x):
        anchor_pl_target = self.anchor(x).logits
        anchor_pl_target = anchor_pl_target.softmax(dim=1)
        for _ in range(self.steps):
            leader_pl_target = self.leader(x).logits
            leader_pl_target = leader_pl_target.softmax(dim=1)
            self._ctc_adapt_auto(
                wavs=[x, x],
                labels=[anchor_pl_target, leader_pl_target],
                batch_size=1
            )
            self._update_leader()
    
    def _ctc_adapt_auto(self, wavs, labels, batch_size):
        self.system.zero_grad()
        denom_scale = len(wavs) // batch_size
        assert denom_scale > 0
        for x, label in zip(wavs, labels):
            loss = self._ctc_adapt_loss_only(x, label)
            loss = loss / denom_scale
            loss.backward()
        
        self.system_optimizer.step()
        self.system.zero_grad()
    
    def _ctc_adapt_loss_only(self, x, label):
        #since no ctc loss was used for fine-tuning, and this is a emotion detection task, we use cross-entropy loss instead
        output = self.system(x).logits
        loss = F.cross_entropy(output, label)
        return loss

    def forward(self, x):
        """
        Forward pass with adaptation
        Args:
            x: Input tensor
        Returns:
            Predictions from the adapted model
        """
        x = x.to(self.device)
        self._update(x)
        self.system.eval()
        predictions = self.system(x).logits
        predictions = predictions.softmax(dim=1)
        
        return predictions

def configure_model(model):
    """Configure model for use with tent."""
    model.eval()
    model.requires_grad_(False)
    return model

def collect_params(model, bias_only=False, train_feature=False, train_all=False, train_LN=True, bitfit=False):
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

        if bitfit:
            for np, p in model.named_parameters():
                if str(np).split('.')[1] == 'encoder' and "bias" in np:
                    p.requires_grad = True
                    params.append(p)
                    names.append(np)
        
        for nm, m in model.named_modules():
            # print(nm)
            if train_LN: 
                if isinstance(m, nn.LayerNorm):
                    for np, p in m.named_parameters():
                        if np in trainable:
                            if not p.requires_grad:
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
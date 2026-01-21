from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


def get_tta_transforms(gaussian_std: float=0.001, soft=False):
    """
    Create audio augmentation transforms for TTA, similar to image transforms
    
    Args:
        gaussian_std (float): Standard deviation for Gaussian noise
        soft (bool): Whether to use soft (less aggressive) augmentations
    
    Returns:
        A function that applies random audio augmentations
    """
    
    def apply_audio_transforms(audio_array):
        """Apply audio augmentations similar to image transforms"""
        import torch
        import numpy as np
        
        # Convert to tensor if numpy
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array).float()
        else:
            audio_tensor = audio_array.float()
        
        # Store original shape to restore later
        original_shape = audio_tensor.shape
        
        # Ensure we work with 2D tensor [batch_size, sequence_length]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim
        elif audio_tensor.dim() == 3 and audio_tensor.shape[1] == 1:
            # If shape is [batch, 1, sequence], squeeze the middle dimension
            audio_tensor = audio_tensor.squeeze(1)
        
        # 1. Add Gaussian noise (equivalent to GaussianNoise in image transforms)
        noise = torch.randn_like(audio_tensor) * gaussian_std
        audio_tensor = audio_tensor + noise
        
        ''' # 2. Volume scaling (equivalent to brightness/contrast in images)
        if soft:
            volume_factor = torch.rand(1).item() * (1.2 - 0.8) + 0.8  # Uniform between 0.8 and 1.2
        else:
            volume_factor = torch.rand(1).item() * (1.3 - 0.7) + 0.7  # Uniform between 0.7 and 1.3
        audio_tensor = audio_tensor * volume_factor
        
        # 3. Time shifting (equivalent to translation in images)
        shift_samples = int(0.1 * audio_tensor.shape[-1])  # 10% of length
        if soft:
            shift_amount = torch.randint(-shift_samples//2, shift_samples//2, (1,)).item()
        else:
            shift_amount = torch.randint(-shift_samples, shift_samples, (1,)).item()
        audio_tensor = torch.roll(audio_tensor, shifts=shift_amount, dims=-1)
        
        # 4. Time masking (equivalent to random crop/pad)
        mask_length = int(0.05 * audio_tensor.shape[-1]) if soft else int(0.1 * audio_tensor.shape[-1])
        mask_start = torch.randint(0, max(1, audio_tensor.shape[-1] - mask_length), (1,)).item()
        audio_tensor[..., mask_start:mask_start + mask_length] *= 0.1  # Attenuate instead of zero
        
        # 5. Polarity flip (equivalent to horizontal flip)
        if torch.rand(1).item() < 0.5:
            audio_tensor = -audio_tensor
        
        # 6. Clip to reasonable range (equivalent to Clip in image transforms)
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)'''
        
        # Restore original shape
        if len(original_shape) == 1:
            # Original was 1D, squeeze back to 1D
            audio_tensor = audio_tensor.squeeze(0)
        elif len(original_shape) == 2:
            # Original was 2D [batch, sequence], keep as is
            pass
        elif len(original_shape) == 3:
            # Original was 3D [batch, 1, sequence], add back the middle dimension
            audio_tensor = audio_tensor.unsqueeze(1)
        
        # Ensure we return the same type as input
        if isinstance(audio_array, np.ndarray):
            return audio_tensor.numpy()
        else:
            return audio_tensor
    
    return apply_audio_transforms

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.transform = get_tta_transforms()

        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        model_output = self.model(x)
        outputs = model_output.logits if hasattr(model_output, 'logits') else model_output
        # Teacher Prediction
        anchor_output = self.model_anchor(x)
        anchor_logits = anchor_output.logits if hasattr(anchor_output, 'logits') else anchor_output
        anchor_prob = torch.nn.functional.softmax(anchor_logits, dim=1).max(1)[0]
        
        standard_ema_output = self.model_ema(x)
        standard_ema = standard_ema_output.logits if hasattr(standard_ema_output, 'logits') else standard_ema_output
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        for i in range(N):
            ema_output = self.model_ema(self.transform(x))
            outputs_ = ema_output.logits if hasattr(ema_output, 'logits') else ema_output
            outputs_ = outputs_.detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().to(p.device) 
                        with torch.no_grad():
                            # Ensure the stored state is on the same device as the current parameter
                            stored_param = self.model_state[f"{nm}.{npp}"].to(p.device)
                            p.data = stored_param * mask + p * (1.-mask)
        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.eval()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
import torch
import torch.jit
import logging
from typing import List, Dict

import time
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy


class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):

        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf


class linear_affinity(AffinityMatrix):

    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())

class LAME(nn.Module):
    """
    LAME (Laplacian Regularization) adapted for emotion detection.
    
    Uses graph-based regularization to enforce smoothness in predictions
    based on feature similarity between audio samples.
    """  
    def __init__(self, model, knn=5, sigma=1.0, affinity='kNN', force_symmetry=True):
        """
        Args:
            model: Pre-trained audio emotion detection model
            knn: Number of nearest neighbors for graph construction
            sigma: RBF kernel bandwidth (if using RBF affinity)
            affinity: Type of affinity matrix ('kNN', 'rbf', 'linear')
            force_symmetry: Whether to symmetrize the affinity matrix
        """
        super().__init__()
        self.model = model

        self.knn = knn
        self.sigma = sigma
        self.affinity = eval(f'{affinity}_affinity')(sigma=self.sigma, knn=self.knn)
        self.force_symmetry = force_symmetry

    def forward(self, x):
        with torch.no_grad():
            # Get features and predictions from the audio model
            if hasattr(self.model, 'forward_features') and hasattr(self.model, 'forward_head'):
                # For models with separate feature extraction and head
                feats = self.model.forward_features(x)
                out = self.model.forward_head(feats)
            else:
                # For HuggingFace models, extract features differently
                if hasattr(self.model(x), 'logits'):
                    #print("Start forwarding by using HuggingFace model")
                    model_output = self.model(x, output_hidden_states=True)
                    out = model_output.logits  # [N, num_classes]
                    # Get features from last hidden state
                    if hasattr(model_output, 'hidden_states'):
                        feats = model_output.hidden_states[-1]  # [N, seq_len, hidden_dim]
                        feats = feats.mean(dim=1)  # Pool over sequence length: [N, hidden_dim]
                    else:
                        # Fallback: use a copy of model with classifier removed
                        featurizer = deepcopy(self.model)
                        if hasattr(featurizer, 'classifier'):
                            featurizer.classifier = nn.Identity()
                        feats = featurizer(x)
                        if len(feats.shape) > 2:
                            feats = feats.mean(dim=1)
            
            probas = out.softmax(dim=1)  # [N, num_classes]
            
            # --- Get unary terms and kernel ---
            unary = -torch.log(probas + 1e-10)  # [N, num_classes]
            
            # Process features for graph construction
            if len(feats.shape) > 2:
                # If feats has sequence dimension, pool it
                feats = feats.mean(dim=1)  # [N, hidden_dim]
            
            # Normalize features for better similarity computation
            feats = F.normalize(feats, p=2, dim=-1)  # [N, hidden_dim]
            
            # Build affinity matrix (graph)
            kernel = self.affinity(feats)  # [N, N]
            if self.force_symmetry:
                kernel = 1/2 * (kernel + kernel.t())

            # --- Perform Laplacian optimization ---
            Y = laplacian_optimization(unary, kernel)  # [N, num_classes]
            
            # Convert Y to logits by inverse softmax (log of probabilities)
            logits = torch.log(Y + 1e-10)  # Add small epsilon to avoid log(0)
            
            return logits
    
    def reset(self):
        """Reset method for compatibility with other TTA methods"""
        pass

def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):

    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            # print(f'Converged in {i} iterations')
            break
        else:
            oldE = E

    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    """Compute the energy function for Laplacian optimization"""
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E

def configure_model(model):
    """Configure model for use with LAME."""
    model.eval()
    # LAME doesn't modify the model parameters, just uses it for predictions
    model.requires_grad_(False)
    return model

def collect_params(model):
    """LAME doesn't optimize model parameters, so return empty lists."""
    return [], []
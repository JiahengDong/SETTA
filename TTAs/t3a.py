import torch
import torch.nn as nn
from copy import deepcopy

def get_audio_featurizer(model):
    """Extract feature extractor from audio classification model"""
    # For HuggingFace models, we need to get features before the classifier
    if hasattr(model, 'classifier'):
        # Store original classifier and replace with identity
        original_classifier = model.classifier
        model.classifier = nn.Identity()
        return model, original_classifier
    elif hasattr(model, 'head'):
        original_head = model.head
        model.head = nn.Identity()
        return model, original_head
    else:
        # For models where we can't easily separate, return as-is
        return model, None

class T3A(torch.nn.Module):
    """
    Test Time Template Adjustments (T3A) adapted for emotion detection
    
    T3A maintains a support set of features and labels, selecting the most confident
    examples for each class to build a prototype-based classifier.
    """
    def __init__(self, model, num_classes=8, filter_K=64):
        super().__init__()
        
        # For HuggingFace audio models, we need to handle the classifier differently
        if hasattr(model, 'classifier'):
            if hasattr(model, 'projector'):
                # Handle Wav2Vec2ForSequenceClassification which has projector + classifier
                self.projector = model.projector
                self.classifier = model.classifier
                # Create a copy for feature extraction
                self.featurizer = deepcopy(model)
                self.featurizer.classifier = nn.Identity()
                self.featurizer.projector = nn.Identity()
            else:
                self.classifier = model.classifier
                # Create a copy for feature extraction
                self.featurizer = deepcopy(model)
                self.featurizer.classifier = nn.Identity()
        elif hasattr(model, 'head'):
            self.classifier = model.head  
            self.featurizer = deepcopy(model)
            self.featurizer.head = nn.Identity()
        else:
            # If we can't separate, use the full model
            self.classifier = model
            self.featurizer = model

        # Initialize warmup supports using classifier weights as prototypes
        warmup_supports = self.classifier.weight.data  # (num_classes, hidden_dim)
        self.warmup_supports = warmup_supports
            
        # Get initial probabilities for these prototypes
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        print("supports", self.supports.shape, self.supports)
        print("labels", self.labels.shape, self.labels)
        print("ent", self.ent.shape, self.ent)
        self.filter_K = filter_K
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=True):
        # Extract features from audio input
        if hasattr(self.featurizer(x), 'last_hidden_state'):
            # For HuggingFace models that return BaseModelOutput
            z = self.featurizer(x).last_hidden_state.mean(dim=1)  # Pool over sequence length
        elif hasattr(self.featurizer(x), 'logits'):
            # If featurizer still outputs logits, we need to get hidden states
            # This shouldn't happen if we set up the featurizer correctly
            outputs = self.featurizer(x, output_hidden_states=True)
            z = outputs.hidden_states[-1].mean(dim=1)  # Use last hidden state, pool over time
        else:
            # Direct feature output
            z = self.featurizer(x)
            if len(z.shape) > 2:  # If there's a sequence dimension
                z = z.mean(dim=1)  # Pool over sequence length
        
        if adapt:
            # Online adaptation: add current example to support set
            if hasattr(self, 'projector'):
                # Project features before classification for Wav2Vec2
                z_proj = self.projector(z)
                p = self.classifier(z_proj)
            else:
                p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # Move supports to same device as input
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            
            # Add current features to support set
            if hasattr(self, 'projector'):
                # Project features before adding to support set
                z_proj = self.projector(z)
                self.supports = torch.cat([self.supports, z_proj])
            else:
                self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
            print("adaptation supports", self.supports.shape)
            print("adaptation labels", self.labels.shape)
            print("adaptation ent", self.ent.shape)
        # Select most confident supports for each class
        supports, labels = self.select_supports()
        
        # Build prototype-based classifier
        if hasattr(self, 'projector'):
            # Project features before classification for Wav2Vec2
            z = self.projector(z)  # Already projected supports
        
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ labels)  # (hidden_dim, num_classes)
        
        # Compute similarity-based predictions
        z_norm = torch.nn.functional.normalize(z, dim=1)
        weights_norm = torch.nn.functional.normalize(weights, dim=0)
        
        return z_norm @ weights_norm

    def select_supports(self):
        """Select the most confident examples for each emotion class"""
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        
        if filter_K == -1:
            # Use all supports
            return self.supports, self.labels

        indices = []
        device = self.supports.device
        indices1 = torch.arange(len(ent_s), device=device)
        
        for i in range(self.num_classes):
            # Find examples predicted as class i
            class_mask = (y_hat == i)
            if class_mask.sum() == 0:
                continue  # No examples for this class
            
            # Sort by entropy (ascending = most confident first)
            class_entropies = ent_s[class_mask]
            _, indices2 = torch.sort(class_entropies)
            
            # Select top-K most confident examples for this class
            class_indices = indices1[class_mask][indices2][:filter_K]
            indices.append(class_indices)
        
        if len(indices) > 0:
            indices = torch.cat(indices)
            
            # Update support set with selected examples
            self.supports = self.supports[indices]
            self.labels = self.labels[indices]
            self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        """Reset support set to initial warmup supports"""
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

def configure_model(model):
    """Configure model for use with T3A."""
    # T3A doesn't modify the model parameters, just uses it for adaptation
    model.requires_grad_(False)
    return model

def collect_params(model):
    """T3A doesn't optimize model parameters, so return empty lists."""
    return [], []

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
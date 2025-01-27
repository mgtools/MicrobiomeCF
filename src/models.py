import torch
import torch.nn as nn
import torch.nn.functional as F


# Define MaskedLinear
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mask = nn.Parameter(mask, requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if self.bias is not None:
            self.bias.register_hook(self._zero_bias_grad)
                
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.weight.data *= self.mask  # Apply mask to initial weights
        
    def forward(self, input):
        masked_weight = self.weight * self.mask  # Apply mask to weights
        return F.linear(input, masked_weight, self.bias)
    
    def _zero_bias_grad(self, grad):
        # Hook function to zero out the bias gradient
        return torch.zeros_like(grad)
    
    def __repr__(self):
        return (f"MaskedLinear("
                f"in_features={self.weight.shape[1]}, "
                f"out_features={self.weight.shape[0]}, "
                f"bias={self.bias is not None}, "
                f"mask_nonzero={self.mask.nonzero().size(0)})")

def previous_power_of_two(x):
    """
    Return the largest power of two less than or equal to x.
    """
    return 1 << (x - 1).bit_length() - 1

class PearsonCorrelationLoss(nn.Module):
    """
    Custom loss function for Pearson Correlation.
    """
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, pred, target):
        x = target
        y = pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm = x - mx
        ym = y - my
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)) + 1e-5
        r = r_num / r_den
        r = torch.clamp(r, min=-1.0, max=1.0)
        return r ** 2

class GAN(nn.Module):
    """
    GAN model with encoder, classifier, and disease_classifier.
    """
    def __init__(self, mask, num_layers=1, latent_dim = 64):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = self._build_encoder(mask, latent_dim, 1)
        self.classifier = self._build_classifier(self.latent_dim, num_layers)
        self.disease_classifier = self._build_classifier(self.latent_dim, num_layers)

    def _build_encoder(self, mask, latent_dim, num_layers):
        """Build the encoder network."""
        layers = []
        in_size = mask.shape[1]
        out_size = mask.shape[0]
            
        # Create a MaskedLinear layer and add to layers
        layers.append(MaskedLinear(in_size, out_size, mask))
            
        # Add ReLU activation for all layers except the last one
        layers.append(nn.BatchNorm1d(out_size))
        layers.append(nn.ReLU())

        first_layer = previous_power_of_two(out_size)
        current_dim = first_layer
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(out_size, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                nn.ReLU(),
            ])
            current_dim = current_dim // 2
        layers.extend([
            nn.Linear(current_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        ])

        # Return the model as an nn.Sequential container
        return nn.Sequential(*layers)

    def _build_classifier(self, latent_dim, num_layers):
        """Build the classifier network."""
        layers = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                nn.Tanh(),
            ])
            current_dim = current_dim // 2
           
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

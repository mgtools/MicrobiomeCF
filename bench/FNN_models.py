import torch
import torch.nn as nn

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
    def __init__(self, input_size, latent_dim=64, num_layers=1):
        super(GAN, self).__init__()

        self.encoder = self._build_encoder(input_size, latent_dim, num_layers)
        self.classifier = self._build_classifier(latent_dim, num_layers)
        self.disease_classifier = self._build_classifier(latent_dim, num_layers)

    def _build_encoder(self, input_size, latent_dim, num_layers):
        """Build the encoder network."""
        layers = []
        first_layer = previous_power_of_two(input_size)
        layers.extend([
            nn.Linear(input_size, first_layer),
            nn.BatchNorm1d(first_layer),
            nn.ReLU()
        ])
        current_dim = first_layer
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                nn.ReLU()
            ])
            current_dim = current_dim // 2
        layers.extend([
            nn.Linear(current_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        ])
        return nn.Sequential(*layers)

    def _build_classifier(self, latent_dim, num_layers):
        """Build the classifier network."""
        layers = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                nn.Tanh()
            ])
            current_dim = current_dim // 2
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

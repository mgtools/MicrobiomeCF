import torch
import torch.nn as nn

def previous_power_of_two(x):
    """Return the largest power of two less than or equal to x."""
    return 1 << ((x - 1).bit_length() - 1)

def get_norm_layer(norm_type, num_features):
    """Return a normalization layer based on norm_type."""
    if norm_type == "batch":
        return nn.BatchNorm1d(num_features)
    elif norm_type == "layer":
        return nn.LayerNorm(num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

def get_activation(act):
    """Return an activation layer based on the given activation function name."""
    act = act.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        raise ValueError(f"Unsupported activation: {act}")
    

class KLDivergenceLoss(nn.Module):
    """
    Custom loss function based on KL divergence that drives the predicted probability toward 0.5.
    This makes the output as uninformative as possible (i.e., removes confounder signal).
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, pred, target):
        # Here, 'pred' should be in the range (0,1); if your network outputs logits, apply torch.sigmoid externally.
        eps = 1e-8  # to avoid log(0)
        p = torch.clamp(pred, min=eps, max=1.0 - eps)
        uniform_value = 0.5
        # Compute KL divergence for a Bernoulli distribution vs uniform [0.5, 0.5]
        kl = p * torch.log(p / uniform_value) + (1.0 - p) * torch.log((1.0 - p) / uniform_value)
        return torch.mean(kl)
    

class MSEUniformLoss(nn.Module):
    """
    Custom loss function based on Mean Squared Error that forces the predicted probability toward 0.5.
    For binary classification, 0.5 represents the uninformative (uniform) prediction.
    """
    def __init__(self):
        super(MSEUniformLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # 'pred' should be in the range (0,1); if necessary, apply torch.sigmoid externally.
        target_uniform = torch.full_like(pred, 0.5)
        return self.mse_loss(pred, target_uniform)



class PearsonCorrelationLoss(nn.Module):
    """
    Custom loss function based on Pearson correlation.
    Returns 0 loss for perfect correlation.
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
        # Loss is higher when correlation is high (perfect correlation gives 1 loss).
        return r ** 2

class GAN(nn.Module):
    """
    GAN model with a configurable encoder and two classifier branches.
    (Despite the name GAN, this model defines an encoder with two parallel classifier heads—
     one of which, disease_classifier, is used for the main prediction.)
    """
    def __init__(self, input_size, latent_dim, num_encoder_layers, num_classifier_layers,
                 dropout_rate, norm="batch", classifier_hidden_dims=None, activation="relu", last_activation = "relu"):
        super(GAN, self).__init__()
        self.activation = activation  # Save the chosen activation
        self.last_activation = last_activation
        self.encoder = self._build_encoder(input_size, latent_dim, num_encoder_layers,
                                           dropout_rate, norm)
        self.classifier = self._build_classifier(latent_dim, num_classifier_layers,
                                                 dropout_rate, norm, classifier_hidden_dims)
        self.disease_classifier = self._build_classifier(latent_dim, num_classifier_layers,
                                                         dropout_rate, norm, classifier_hidden_dims)

    def _build_encoder(self, input_size, latent_dim, num_layers, dropout_rate, norm):
        layers = []
        # Starting layer: use the largest power of two ≤ input_size.
        first_layer_dim = previous_power_of_two(input_size)
        layers.append(nn.Linear(input_size, first_layer_dim))
        layers.append(get_norm_layer(norm, first_layer_dim))
        layers.append(get_activation(self.activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        current_dim = first_layer_dim
        # Add extra encoder layers if requested.
        for _ in range(num_layers - 1):
            new_dim = current_dim // 2
            layers.append(nn.Linear(current_dim, new_dim))
            layers.append(get_norm_layer(norm, new_dim))
            layers.append(get_activation(self.activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = new_dim
        
        # Final projection to the latent space.
        layers.append(nn.Linear(current_dim, latent_dim))
        layers.append(get_norm_layer(norm, latent_dim))
        layers.append(get_activation(self.activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _build_classifier(self, latent_dim, num_layers, dropout_rate, norm, hidden_dims):
        layers = []
        current_dim = latent_dim
        
        # If hidden dimensions are provided, use them.
        if hidden_dims and len(hidden_dims) > 0:
            for hd in hidden_dims:
                layers.append(nn.Linear(current_dim, hd))
                layers.append(get_norm_layer(norm, hd))
                layers.append(get_activation(self.activation))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = hd
        else:
            # Otherwise, reduce dimension by half in each layer.
            for i in range(num_layers):
                new_dim = current_dim // 2
                layers.append(nn.Linear(current_dim, new_dim))
                layers.append(get_norm_layer(norm, new_dim))
                if i == num_layers-1:
                    layers.append(get_activation(self.activation))
                else:
                    layers.append(get_activation(self.last_activation))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = new_dim
        
        # Final output layer.
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

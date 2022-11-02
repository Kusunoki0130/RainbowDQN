import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """
    基本的网络结构
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DuelingNetwork(nn.Module):
    """
    Dueling 网络结构
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(DuelingNetwork, self).__init__()

        # self.layers = nn.Sequential(
        #     nn.Linear(in_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, out_dim)
        # )

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class NoisyLinear(nn.Module):
    """
    Noisy 层
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon
        )


class NoisyNetwork(nn.Module):
    """
    Noisy 网络结构
    """
    def __init__(self, in_dim: int, out_dim: int):
        super(NoisyNetwork, self).__init__()

        self.feature_layer = nn.Linear(in_dim, 128)
        self.noisy_layer1 = NoisyLinear(128, 128)
        self.noisy_layer2 = NoisyLinear(128, out_dim)

    def reset_noise(self):
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = F.relu(self.feature_layer(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)
        return out


class CategoricalNetwork(nn.Module):
    """
    Categorical 网络结构
    """

    def __init__(self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor):
        """Initialization."""
        super(CategoricalNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist


class NoisyDuelingCategoricalNetwork(nn.Module):
    """
    Rainbow 网络结构
    """

    def __init__(self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor):
        super(NoisyDuelingCategoricalNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(
            -1, 1, self.atom_size
        )
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist

    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class NoisyCategoricalNetwork(nn.Module):
    """
    no dueling
    """

    def __init__(self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor):
        super(NoisyCategoricalNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )
        self.noisy_layer_1 = NoisyLinear(128, 128)
        self.noisy_layer_2 = NoisyLinear(128, out_dim * atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        noisy_1 = F.relu(self.noisy_layer_1(feature))
        q_atoms = self.noisy_layer_2(noisy_1).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist

    def reset_noise(self):
        self.noisy_layer_1.reset_noise()
        self.noisy_layer_2.reset_noise()


class DuelingCategoricalNetwork(nn.Module):
    """
    no noisy
    """

    def __init__(self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor):
        super(DuelingCategoricalNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )

        self.advantage_layer = nn.Sequential (
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim * atom_size)
        )

        self.value_layer = nn.Sequential (
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        advantage = self.advantage_layer(feature).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(feature).view(
            -1, 1, self.atom_size
        )
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist


class NoisyDuelingNetwork(nn.Module):
    """
    no categorial
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(NoisyDuelingNetwork, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim)

        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)
        advantage_hidden = F.relu(self.advantage_hidden_layer(feature))
        advantage = self.advantage_layer(advantage_hidden)
        value_hidden = F.relu(self.value_hidden_layer(feature))
        value = self.value_layer(value_hidden)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

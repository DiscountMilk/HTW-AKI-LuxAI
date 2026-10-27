"""Netzarchitekturen.

`QNet` entspricht der Vorlesungsfolie 47 (MLP -> Q-Werte pro Aktion), erweitert um
eine optionale Dueling-Zerlegung Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)).
Die Zerlegung hilft hier, weil die meisten Steps "Lagebewertung" sind (V) und die
Aktionswahl oft kaum einen Unterschied macht (A).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 0 stay, 1 up, 2 right, 3 down, 4 left, 5 sap
N_ACTIONS = 6


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS,
                 hidden: int = 256, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
            self.adv = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, n_actions))
        else:
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if not self.dueling:
            return self.head(h)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)

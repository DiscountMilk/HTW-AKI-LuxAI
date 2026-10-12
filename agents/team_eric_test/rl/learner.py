"""
DQN-Learner: Netz + Aktion wählen + Lernen. Wie agents/DQN/main.py, nur ohne Feature-Code.

INPUT : s    = Feature-Vektor einer Einheit (aus rl/spaces.py)
        mask = bool pro Aktion, False = verboten (aus den Regeln)
OUTPUT: pick() -> Aktions-Index 0..n_actions-1

Immer get_learner() benutzen, nicht Learner(): Der Runner baut den Agent jedes Spiel neu,
der Learner (Netz + Erfahrungen) soll aber über alle Spiele weiterleben.
"""
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    """Feature-Vektor -> ein Q-Wert pro Aktion. Ein Netz für alle Einheiten."""
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class Learner:
    def __init__(self, obs_dim, n_actions, path=None, lr=1e-3, gamma=0.99, epsilon=0.1, batch_size=64):
        self.path = path
        self.gamma = gamma            # wie wichtig ist die Zukunft (0 = nur jetzt)
        self.epsilon = epsilon        # Anteil Zufallsaktionen (ausprobieren)
        self.batch_size = batch_size
        self.q = QNet(obs_dim, n_actions)
        if path and os.path.isfile(path):
            self.q.load_state_dict(torch.load(path))
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = deque(maxlen=100_000)   # Erfahrungen (s, a, r, s2, done), älteste fliegt raus

    def pick(self, s, mask):
        """Mit Wahrscheinlichkeit epsilon zufällig, sonst höchster Q-Wert. Nur erlaubte Aktionen."""
        if random.random() < self.epsilon:
            return int(random.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            q = self.q(torch.as_tensor(s, dtype=torch.float32))
        q[~torch.as_tensor(mask)] = -float("inf")
        return int(q.argmax())

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def train_step(self):
        """Lernziel: Q(s, a) = r + gamma * max Q(s2). Bei done gibt es keine Zukunft.
        Lernt aus zufälligen alten Erfahrungen statt nur aus dem letzten Schritt (stabiler)."""
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, done = (torch.as_tensor(np.array(col), dtype=torch.float32) for col in zip(*batch))

        q_pred = self.q(s).gather(1, a.long().unsqueeze(1)).squeeze(1)   # Q der gewählten Aktion
        with torch.no_grad():
            # ponytail: kein Target-Netz (wie DQN-Ordner); einbauen, falls Q-Werte explodieren
            target = r + self.gamma * (1 - done) * self.q(s2).max(dim=1).values

        loss = nn.functional.mse_loss(q_pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def eval_mode(self):
        self.epsilon = 0.0

    def save(self):
        if self.path:
            torch.save(self.q.state_dict(), self.path)


_LEARNERS = {}   # path -> Learner, lebt solange Python läuft (also über alle Spiele)


def get_learner(path, obs_dim, n_actions):
    if path not in _LEARNERS:
        _LEARNERS[path] = Learner(obs_dim, n_actions, path)
    return _LEARNERS[path]


if __name__ == "__main__":
    # Selbsttest: Aktion 2 gibt Reward 1, alle anderen 0. Danach muss das Netz 2 wählen.
    # .venv/bin/python -m agents.team_eric_test.rl.learner
    L = Learner(obs_dim=4, n_actions=5)
    for _ in range(500):
        a = random.randrange(5)
        L.add(np.random.rand(4), a, float(a == 2), np.zeros(4), 1.0)
        L.train_step()
    L.eval_mode()
    s, mask = np.random.rand(4), np.ones(5, dtype=bool)
    assert L.pick(s, mask) == 2
    mask[2] = False
    assert L.pick(s, mask) != 2
    print("learner.py OK")

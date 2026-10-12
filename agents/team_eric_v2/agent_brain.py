"""
Brain = das lernende Netz (DQN). Drei Teile, jeder für sich änderbar:

  1 Stellschrauben   : Lernrate, Zufallsanteil, ...  -> Konstanten oben
  2 QNet             : Architektur des Netzes         -> Schichten ändern
  3 Brain.features() : WAS das Netz sieht             -> Zeile hinzufügen, N_FEATURES anpassen
  4 Brain.reward()   : WOFÜR es belohnt wird          -> Term addieren

Schnittstelle für agent.py (bleibt immer gleich):
  features(unit, world) -> float32[N_FEATURES]
  reward(unit, world)   -> float
  pick(s, mask)         -> Aktions-Index 0..4 (= Lux-Aktion STAY..LEFT)
  add(s, a, r, s2)      -> Erfahrung merken
  train()               -> ein Lernschritt
  eval_mode(), save()

ACHTUNG: N_FEATURES oder QNet ändern = alte .pth passt nicht mehr -> löschen oder umbenennen.
"""
import copy
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from agents.team_eric_v2.world import MAX_UNIT_ENERGY, MOVES

# ----------------------------------------------------------------------
# 1 Stellschrauben
# ----------------------------------------------------------------------
N_FEATURES = 12          # muss zur Liste in Brain.features() passen
N_ACTIONS = len(MOVES)   # STAY, UP, RIGHT, DOWN, LEFT
LR = 1e-3                # Lernrate
GAMMA = 0.95             # wie wichtig ist die Zukunft (0 = nur jetzt)
EPSILON = 0.1            # Anteil Zufallsaktionen im Training
BATCH_SIZE = 64
BUFFER_SIZE = 100_000    # so viele Erfahrungen merken, älteste fliegt raus
TARGET_SYNC = 500        # alle N Lernschritte das Ziel-Netz nachziehen
MAX_TILE_ENERGY = 20


# ----------------------------------------------------------------------
# 2 Netz-Architektur
# ----------------------------------------------------------------------
class QNet(nn.Module):
    """Features einer Einheit -> ein Q-Wert pro Aktion (wie gut ist die Aktion?)."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_FEATURES, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x):
        return self.layers(x)


# ----------------------------------------------------------------------
# 3 Brain
# ----------------------------------------------------------------------
class Brain:
    def __init__(self, path=None):
        self.path = path
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.q = QNet()
        if path and os.path.isfile(path):
            self._load(path)
        self.target = copy.deepcopy(self.q)        # langsames Ziel-Netz, macht das Lernen stabiler
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.train_steps = 0
        self.stats = {"loss": [], "q": []}         # Messwerte fürs Training-Log, pop_stats() leert sie

    # --- Was das Netz sieht -------------------------------------------
    def features(self, unit, w):
        """Eine Zahl pro Zeile, etwa -1..1. Neue Zeile -> N_FEATURES anpassen."""
        x, y = unit.pos
        relic_dx, relic_dy = w.offset_to_nearest(unit.pos, w.relics)
        point_dx, point_dy = w.offset_to_nearest(unit.pos, w.point_tiles())
        cand_dx, cand_dy = w.offset_to_nearest(unit.pos, w.candidate_tiles())
        f = [
            unit.energy / MAX_UNIT_ENERGY,                  # eigene Energie
            w.match_step / w.cfg["max_steps_in_match"],     # Match-Fortschritt
            w.tile_energy[x, y] / MAX_TILE_ENERGY,          # Energie des eigenen Felds
            relic_dx, relic_dy,                             # Richtung nächste Relic
            point_dx, point_dy,                             # Richtung nächstes Punktefeld
            float(w.point_tile[x, y] == 1),                 # steht auf Punktefeld
            float(w.near_relic(unit.pos)),                  # im 5x5 um eine Relic
            cand_dx, cand_dy,                               # Richtung nächstes unerforschtes Relic-Feld
            (w.units_at(unit.pos) - 1) / 4,                 # andere eigene Einheiten auf meinem Feld
        ]
        assert len(f) == N_FEATURES, f"N_FEATURES auf {len(f)} setzen (und alte .pth löschen)"
        return np.array(f, dtype=np.float32)

    # --- Wofür es belohnt wird ---------------------------------------
    def reward(self, unit, w):
        """Wie gut war der letzte Schritt für diese Einheit? Neue Terme einfach addieren."""
        # TODO hier Implementieren
        x, y = unit.pos
        r = 0.0
        r += 1.0 * (w.point_tile[x, y] == 1)   # steht auf bekanntem Punktefeld -> eigener Verdienst
        r += 0.2 * w.point_gain                # kleiner Team-Anteil, deckt noch unbekannte Punktefelder ab
        r += 0.05 * w.near_relic(unit.pos)     # in der Nähe einer Relic -> Anreiz zum Suchen
        r += 1.0 * (w.point_tile[x, y] == 1) / w.units_at(unit.pos)   # Punktefeld; gestapelt teilen, zählt nur 1x
        return float(r)

    # --- Entscheiden ---------------------------------------------------
    def pick(self, s, mask):
        """Mit Wahrscheinlichkeit epsilon zufällig, sonst höchster Q-Wert. Nur erlaubte Aktionen."""
        if random.random() < self.epsilon:
            return int(random.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            q = self.q(torch.as_tensor(s))
        q[~torch.as_tensor(mask)] = -float("inf")
        return int(q.argmax())

    # --- Lernen --------------------------------------------------------
    def add(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))

    def train(self):
        """Ziel: Q(s, a) = r + gamma * max Q_ziel(s2). Lernt aus zufälligen alten Erfahrungen."""
        if len(self.buffer) < BATCH_SIZE:
            return
        batch = random.sample(self.buffer, BATCH_SIZE)
        s, a, r, s2 = (torch.as_tensor(np.array(col), dtype=torch.float32) for col in zip(*batch))

        q_pred = self.q(s).gather(1, a.long().unsqueeze(1)).squeeze(1)   # Q der gewählten Aktion
        with torch.no_grad():
            # ponytail: max über ALLE Aktionen, auch verbotene; Maske für s2 mitspeichern, falls Q-Werte auffällig hoch
            q_target = r + self.gamma * self.target(s2).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(q_pred, q_target)   # Huber: robuster als MSE bei Ausreißern
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.stats["loss"].append(loss.item())
        self.stats["q"].append(q_pred.mean().item())

        self.train_steps += 1
        if self.train_steps % TARGET_SYNC == 0:
            self.target.load_state_dict(self.q.state_dict())

    def pop_stats(self):
        """Ø Loss und Ø Q seit dem letzten Aufruf (für das Training-Log), danach leeren."""
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in self.stats.items()}
        self.stats = {k: [] for k in self.stats}
        return out

    # --- Modus / Speichern ---------------------------------------------
    def eval_mode(self):
        self.epsilon = 0.0

    def save(self):
        if self.path:
            torch.save(self.q.state_dict(), self.path)

    def _load(self, path):
        try:
            self.q.load_state_dict(torch.load(path))
        except RuntimeError as e:
            raise RuntimeError(f"{path} passt nicht zu QNet/N_FEATURES. Datei löschen oder umbenennen.") from e


_BRAINS = {}   # path -> Brain, lebt solange Python läuft (also über alle Spiele)


def get_brain(path):
    """Immer get_brain() benutzen, nicht Brain(): Der Agent wird jedes Spiel neu gebaut,
    das Brain (Netz + Erfahrungen) soll aber über alle Spiele weiterleben."""
    if path not in _BRAINS:
        _BRAINS[path] = Brain(path)
    return _BRAINS[path]
"""
Brain = alles, was lernt oder dem Lernen dient. Von oben nach unten lesen:
"""
import copy
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.team_eric_v3.world import (ASTEROID, MAP_H, MAP_W, MAX_STEPS_IN_MATCH, MAX_TILE_ENERGY,
                                       MAX_UNIT_ENERGY, MOVES, RELIC_CONFIG_SIZE, STAY)

# ---------------------------------------------------------------------
# [1] ActionMask - verringert Aktionsspielraum
#     Neue Regel: Methode (unit, action) -> True = verboten, in self.rules eintragen.
# ---------------------------------------------------------------------
class ActionMask:
    def __init__(self, world):
        self.w = world
        self.rules = [
            self.no_energy,
            self.blocked,
            self.stacked,
        ]
        self.checks = self.hits = 0                                  # fürs Log: Entscheidungen / davon mit Verbot
        self.counts = {rule.__name__: 0 for rule in self.rules}      # fürs Log: Verbote je Regel

    def allowed(self, unit):
        """Lässt nur erlaubte Spielzüge zurück: bool pro Aktion (Index = STAY, UP, RIGHT, DOWN, LEFT).
        Ist alles verboten, bleibt STAY erlaubt - das Netz braucht immer eine Wahl."""
        hits = [[rule.__name__ for rule in self.rules if rule(unit, action)] for action in MOVES]
        ok = np.array([not h for h in hits])
        self.checks += 1                                             # fürs Log zählen
        self.hits += any(hits)
        for name in sum(hits, []):
            self.counts[name] += 1
        if not ok.any():
            ok[STAY] = True
        return ok

    def no_energy(self, unit, action):
        """Zu wenig Energie -> Bewegung würde scheitern (STAY kostet nichts)."""
        return action != STAY and unit.energy < self.w.unit_move_cost

    def blocked(self, unit, action):
        """Nicht aus der Karte oder in einen (zuletzt gesehenen) Asteroiden laufen."""
        if action == STAY:
            return False
        x, y = unit.pos[0] + MOVES[action][0], unit.pos[1] + MOVES[action][1]
        return not (0 <= x < MAP_W and 0 <= y < MAP_H) or self.w.tile_type[x, y] == ASTEROID

    def stacked(self, unit, action):
        """Wenn mehrere Units auf einer Position sind, ist STAY verboten -
        außer für die mit der kleinsten ID: das Feld zählt nur 1x, eine darf bleiben."""
        if action != STAY:
            return False
        here = [u.id for u in self.w.units if u.pos[0] == unit.pos[0] and u.pos[1] == unit.pos[1]]
        return len(here) > 1 and unit.id != min(here)


# ---------------------------------------------------------------------
# [2] Reward - Wie gut war der letzte Schritt für eine Unit? Summe aus gewichteten Termen.
#     Neuer Term: Methode (unit, w) -> Zahl schreiben und mit Gewicht in self.terms eintragen.
# ---------------------------------------------------------------------
class Reward:
    def __init__(self):
        self.terms = {                     # Term -> Gewicht
            self.on_point_tile: 2.0,
            self.team_gain: 0.03,
            self.near_relic: 0.05,
            self.explore: 0.1,
        }
        self.reset()

    def reset(self):
        """Zähler fürs Log leeren (zu Spielbeginn)."""
        self.totals = {term.__name__: 0.0 for term in self.terms}    # fürs Log: Summe je Term
        self.count = 0                                               # fürs Log: Anzahl Rewards

    def compute(self, unit, w):
        values = {term.__name__: weight * term(unit, w) for term, weight in self.terms.items()}
        for name, value in values.items():
            self.totals[name] += value
        self.count += 1
        return float(sum(values.values()))

    def on_point_tile(self, unit, w):
        """1, wenn die Unit auf einem bekannten Punktefeld steht - geteilt durch Units am Feld (zählt nur 1x)."""
        x, y = unit.pos
        if w.point_tile[x, y] != 1:
            return 0.0
        return 1 / sum(u.pos[0] == x and u.pos[1] == y for u in w.units)

    def team_gain(self, unit, w):
        """Punktezuwachs des Teams - deckt auch noch unbekannte Punktefelder ab."""
        return w.point_gain

    def near_relic(self, unit, w):
        """1 im 5x5 um eine bekannte Relic - lockt zum Suchen dorthin."""
        return float(w.near_relic(unit.pos))

    def explore(self, unit, w):
        """0..1: viel Karte im Sichtfeld (nicht an Wand/Ecke) und wenig eigene Units darin.
        Mitte, allein = 1 · Ecke = ca. 0.4 · jede weitere eigene Unit im Sichtfeld teilt den Wert."""
        r, (x, y) = w.unit_sensor_range, unit.pos
        cols = min(x + r, MAP_W - 1) - max(x - r, 0) + 1
        rows = min(y + r, MAP_H - 1) - max(y - r, 0) + 1
        on_map = cols * rows / (2 * r + 1) ** 2                  # Anteil des Sichtfelds auf der Karte
        mates = sum(max(abs(u.pos[0] - x), abs(u.pos[1] - y)) <= r for u in w.units) - 1   # ohne sich selbst
        return on_map / (1 + mates)

    def positive_energy_tiles(self, unit, w):
        """Reward, wenn Unit auf einem Positiven Energy Tile steht."""
        pass


# ---------------------------------------------------------------------
# [3] FeatureEngineering - was das Netz sieht: World + Unit -> s_t
# ---------------------------------------------------------------------
class FeatureEngineering:
    """Jedes Feature ist eine Methode (unit, w) -> Liste von Zahlen (etwa -1..1).
    Neues Feature: Methode schreiben und mit Anzahl Zahlen in PARTS eintragen.
    PARTS ändern = andere Netzgröße -> alte .pth umbenennen."""

    PARTS = [                              # Methode, Anzahl Zahlen
        ("energy", 1),
        ("match_progress", 1),
        ("tile_energy", 1),
        ("to_relic", 2),
        ("to_point_tile", 2),
        ("on_point_tile", 1),
        ("near_relic", 1),
        ("to_unexplored", 2),
        ("stacked", 1),
    ]
    SIZE = sum(n for _, n in PARTS)        # Länge von s_t = Eingang des Netzes

    def encode(self, unit, w):
        s = [x for name, _ in self.PARTS for x in getattr(self, name)(unit, w)]
        assert len(s) == self.SIZE, "Anzahl in PARTS passt nicht zur Länge, die eine Methode liefert"
        return np.array(s, dtype=np.float32)

    # --- Features ---------------------------------------------------------
    def energy(self, unit, w):
        """Eigene Energie."""
        return [unit.energy / MAX_UNIT_ENERGY]

    def match_progress(self, unit, w):
        """Wie weit das Match ist: früh erkunden, spät punkten."""
        return [w.match_steps / MAX_STEPS_IN_MATCH]

    def tile_energy(self, unit, w):
        """Energie des eigenen Felds: aufladen oder weiterziehen?"""
        x, y = unit.pos
        return [w.tile_energy[x, y] / MAX_TILE_ENERGY]

    def to_relic(self, unit, w):
        """Richtung zur nächsten bekannten Relic."""
        return self._direction(unit.pos, w.relics)

    def to_point_tile(self, unit, w):
        """Richtung zum nächsten bekannten Punktefeld."""
        return self._direction(unit.pos, np.argwhere(w.point_tile == 1))

    def on_point_tile(self, unit, w):
        """Steht die Unit auf einem bekannten Punktefeld?"""
        x, y = unit.pos
        return [float(w.point_tile[x, y] == 1)]

    def near_relic(self, unit, w):
        """Im 5x5 um eine Relic - hier liegen Punktefelder."""
        return [float(w.near_relic(unit.pos))]

    def to_unexplored(self, unit, w):
        """Richtung zum nächsten Relic-Feld, dessen Status noch unbekannt ist (dort lohnt Ausprobieren)."""
        r = RELIC_CONFIG_SIZE // 2
        tiles = [(x + dx, y + dy) for x, y in w.relics for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                 if 0 <= x + dx < MAP_W and 0 <= y + dy < MAP_H and w.point_tile[x + dx, y + dy] == -1]
        return self._direction(unit.pos, tiles)

    def stacked(self, unit, w):
        """Andere eigene Units auf meinem Feld: 0 = allein, 1 = vier oder mehr (z.B. am Spawn)."""
        here = sum(u.pos[0] == unit.pos[0] and u.pos[1] == unit.pos[1] for u in w.units)
        return [min(here - 1, 4) / 4]

    # --- Hilfsfunktion -----------------------------------------------------
    @staticmethod
    def _direction(pos, targets):
        """[dx, dy] zum nächsten Ziel, geteilt durch Kartenbreite. Kein Ziel -> [0, 0]."""
        if len(targets) == 0:
            return [0.0, 0.0]
        offsets = np.array(list(targets)) - pos
        dx, dy = offsets[np.abs(offsets).sum(axis=1).argmin()] / MAP_W
        return [float(dx), float(dy)]


# ---------------------------------------------------------------------
# [4] Neuronales Netz
# ---------------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# [5] Brain
# ---------------------------------------------------------------------
class Brain:
    """DQN. Ablauf pro Schritt: decide() -> env -> learn_from() -> train().
    Lebt über alle Spiele (siehe get_brain), deshalb new_game() zu Spielbeginn."""

    def __init__(self, agent_path=None):
        # Hyperparameter
        self.learning_rate = 1e-3
        self.gamma = 0.95                  # wie wichtig ist die Zukunft (0.95 = ca. 20 Schritte)
        self.epsilon = 0.1                 # Anteil Zufallsaktionen (Exploration)
        self.batch_size = 64                # TODO KA
        self.target_sync = 500             # alle N Lernschritte das Target-Netz nachziehen TODO KA

        # Bausteine
        self.path = agent_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = FeatureEngineering()
        self.rewards = Reward()
        self.nn = Network(FeatureEngineering.SIZE, len(MOVES)).to(self.device)
        self.loaded = bool(agent_path and os.path.isfile(agent_path))   # fürs Log: .pth geladen oder neu?
        if self.loaded:
            self.load_agent()                              # erst laden, wenn das Netz existiert
        self.target_nn = copy.deepcopy(self.nn)            # langsame Kopie -> stabileres Lernziel # TODO KA
        self.opt = optim.Adam(self.nn.parameters(), lr=self.learning_rate)

        # Gedächtnis
        self.buffer = deque(maxlen=100_000)                # Replay Buffer: (s, a, r, s2) # TODO KA
        self.train_steps = 0
        self.new_game()

    def new_game(self):
        """Zu Spielbeginn: Gemerktes aus dem letzten Spiel verwerfen."""
        self.pending = {}                                  # (team_id, unit_id) -> (s, a) aus dem letzten Schritt
        self.picks = np.zeros(len(MOVES), dtype=int)       # fürs Log: gewählte Aktionen
        self.losses, self.qs = [], []                      # fürs Log: Lernfehler, Q-Werte
        self.rewards.reset()

    def decide(self, unit, w, mask):
        """ε-greedy unter den erlaubten Aktionen (mask). Merkt (s, a) fürs Lernen. -> [aktion, 0, 0]"""
        s = self.features.encode(unit, w)
        if random.random() < self.epsilon:
            a = int(random.choice(np.flatnonzero(mask)))                   # ausprobieren
        else:
            with torch.no_grad():
                q = self.nn(torch.as_tensor(s, device=self.device)).cpu()
            q[~torch.as_tensor(mask)] = -float("inf")                      # verboten -> nie wählen
            a = int(q.argmax())                                             # beste bekannte Aktion
        self.pending[w.team_id, unit.id] = (s, a)           # Team im Schlüssel: Selbstspiel teilt ein Brain
        self.picks[a] += 1
        return [a, 0, 0]

    def learn_from(self, w):
        """Nach world.update(): (s, a) vom letzten Schritt + neue Welt -> Erfahrung (s, a, r, s2)."""
        if w.match_steps > 0:                                               # Match-Grenze: Übergang verwerfen
            for unit in w.units:
                if (w.team_id, unit.id) in self.pending:
                    s, a = self.pending.pop((w.team_id, unit.id))
                    self.buffer.append((s, a, self.rewards.compute(unit, w), self.features.encode(unit, w)))
        self.pending = {k: v for k, v in self.pending.items() if k[0] != w.team_id}   # tote Units verwerfen

    def train(self):
        """Ein Lernschritt: Q(s, a) soll r + γ · max Q_target(s2) werden. Zufälliger Batch aus dem Buffer."""
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2 = (torch.as_tensor(np.array(col), dtype=torch.float32, device=self.device) for col in zip(*batch))
        q_pred = self.nn(s).gather(1, a.long().unsqueeze(1)).squeeze(1)    # Q der gewählten Aktion
        with torch.no_grad():
            q_target = r + self.gamma * self.target_nn(s2).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(q_pred, q_target)               # Huber: robust gegen Ausreißer
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.losses.append(loss.item())
        self.qs.append(q_pred.mean().item())
        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

    def eval_mode(self):
        """Evaluation: keine Zufallszüge."""
        self.epsilon = 0.0

    def load_agent(self):
        self.nn.load_state_dict(torch.load(self.path, map_location=self.device))

    def save(self):
        if self.path:
            torch.save(self.nn.state_dict(), self.path)


_BRAINS = {}   # agent_path -> Brain, lebt solange Python läuft (also über alle Spiele)


def get_brain(agent_path):
    """Immer get_brain() benutzen, nicht Brain(): ein Brain (Netz + Erfahrungen) über alle Spiele."""
    if agent_path not in _BRAINS:
        _BRAINS[agent_path] = Brain(agent_path)
    return _BRAINS[agent_path]
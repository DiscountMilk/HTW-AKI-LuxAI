"""
World = alles, was der Agent über das laufende Spiel weiß.
obs zeigt nur den aktuellen Schritt und nur das Sichtfeld. World merkt sich mehr.

Regel: Nur update() schreibt. agent_rules.py und agent_brain.py lesen nur (Attribute + Abfragen).

Neues Wissen hinzufügen:
  1. Attribut in __init__() anlegen       (Startwert = "weiß ich noch nicht")
  2. _update_xyz(obs) schreiben           (rechnet aus obs + bisherigem Wissen)
  3. in update() eintragen                (Reihenfolge beachten)
  4. optional: Abfrage in Abschnitt 4     (nur wenn das Lesen mehr als eine Zeile ist)
"""
from dataclasses import dataclass

import numpy as np

# Aktionen (Spalte 0 im actions-Array)
STAY, UP, RIGHT, DOWN, LEFT, SAP = range(6)
MOVES = {STAY: (0, 0), UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}

# Feldtypen aus obs["map_features"]["tile_type"]
UNKNOWN, EMPTY, NEBULA, ASTEROID = -1, 0, 1, 2

MAX_UNIT_ENERGY = 400   # fester Spielwert, steht nicht in env_cfg


@dataclass
class Unit:
    id: int            # Zeile im actions-Array
    pos: np.ndarray    # np.array([x, y])
    energy: int


class World:
    # ------------------------------------------------------------------
    # 1 __init__ - einmal pro Spiel: Agent.set_env_cfg macht self.world = World(env_cfg)
    #   ALLE Attribute stehen hier.
    # ------------------------------------------------------------------
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_units = cfg["max_units"]
        W, H = cfg["map_width"], cfg["map_height"]

        # Bleibt über alle Matches eines Spiels
        self.relics = set()                                    # {(x, y), ...}
        self.tile_type = np.full((W, H), UNKNOWN, dtype=int)   # zuletzt gesehen
        self.tile_energy = np.zeros((W, H), dtype=int)         # zuletzt gesehen
        self.point_tile = np.full((W, H), -1, dtype=int)       # 1 ja, 0 nein, -1 unbekannt

        # Jeder Schritt neu
        self.step = 0
        self.match_step = 0
        self.points = 0
        self.point_gain = 0                                    # Punkte seit letztem Schritt (= Reward)
        self.units = []                                        # eigene Einheiten (Liste von Unit)
        self.enemies = np.zeros((0, 2), dtype=int)             # sichtbare Gegner-Positionen
        self.new_known = set()                                 # Relic-Felder, deren Status gerade geklärt wurde

    # ------------------------------------------------------------------
    # 2 update - jeden Schritt, als Erstes in act()
    # ------------------------------------------------------------------
    def update(self, obs, team):
        self._update_step(obs, team)
        self._update_units(obs, team)
        self._update_map(obs)
        self._update_relics(obs)
        self._update_point_tiles()   # braucht units, relics, point_gain

    # ------------------------------------------------------------------
    # 3 Ein _update_ pro Wissensbereich (nur von update() aufgerufen)
    # ------------------------------------------------------------------
    def _update_step(self, obs, team):
        self.step = int(obs["steps"])
        self.match_step = int(obs["match_steps"])
        points = int(obs["team_points"][team])
        # Neues Match: Punkte starten wieder bei 0, Vergleich mit vorher ist sinnlos
        self.point_gain = points - self.points if self.match_step > 0 else 0
        self.points = points

    def _update_units(self, obs, team):
        mask = obs["units_mask"].astype(bool)
        pos, energy = obs["units"]["position"], obs["units"]["energy"]
        self.units = [Unit(int(i), np.array(pos[team][i]), int(energy[team][i]))
                      for i in np.flatnonzero(mask[team])]
        self.enemies = np.array(pos[1 - team])[mask[1 - team]]

    def _update_map(self, obs):
        """Nur sichtbare Felder überschreiben - Wissen über den Rest bleibt."""
        seen = obs["sensor_mask"].astype(bool)
        self.tile_type[seen] = obs["map_features"]["tile_type"][seen]
        self.tile_energy[seen] = obs["map_features"]["energy"][seen]

    def _update_relics(self, obs):
        for x, y in obs["relic_nodes"][obs["relic_nodes_mask"].astype(bool)]:
            self.relics.add((int(x), int(y)))

    def _update_point_tiles(self):
        """Punkte-Zuwachs = Anzahl besetzter Punktefelder (jedes Feld zählt nur einmal).
        Bekannte Punktefelder abziehen, über den Rest der unbekannten besetzten Felder entscheiden:
          Rest 0                -> keins der unbekannten Felder ist ein Punktefeld
          Rest == Anzahl        -> alle unbekannten Felder sind Punktefelder
        Geklärte Felder landen in new_known (für den Erforschen-Reward)."""
        self.new_known = set()
        if self.match_step == 0:
            return
        occupied = {tuple(u.pos) for u in self.units if self.near_relic(u.pos)}
        unknown = [p for p in occupied if self.point_tile[p] == -1]
        rest = self.point_gain - sum(self.point_tile[p] == 1 for p in occupied)
        if rest == 0:
            for p in unknown:
                self.point_tile[p] = 0
            self.new_known = set(unknown)
        elif rest == len(unknown):
            for p in unknown:
                self.point_tile[p] = 1
            self.new_known = set(unknown)
        # ponytail: 0 < rest < len(unknown) wird ignoriert; Upgrade: Kandidaten pro Feld über Schritte zählen

    # ------------------------------------------------------------------
    # 4 Abfragen - nur lesen, nichts ändern
    # ------------------------------------------------------------------
    def near_relic(self, pos):
        """Liegt pos im 5x5-Bereich um eine bekannte Relic?"""
        return any(max(abs(pos[0] - x), abs(pos[1] - y)) <= 2 for x, y in self.relics)

    def is_blocked(self, x, y):
        """Außerhalb der Karte oder (zuletzt gesehen) Asteroid."""
        W, H = self.tile_type.shape
        return not (0 <= x < W and 0 <= y < H) or self.tile_type[x, y] == ASTEROID

    def offset_to_nearest(self, pos, targets):
        """(dx, dy) zum nächsten Ziel, geteilt durch Kartenbreite. Kein Ziel -> (0, 0)."""
        if len(targets) == 0:
            return 0.0, 0.0
        offsets = np.array(list(targets)) - pos
        dx, dy = offsets[np.abs(offsets).sum(axis=1).argmin()] / self.cfg["map_width"]
        return float(dx), float(dy)

    def point_tiles(self):
        """Array von [x, y] aller sicher erkannten Punktefelder."""
        return np.argwhere(self.point_tile == 1)

    def candidate_tiles(self):
        """Felder im 5x5 um bekannte Relics, deren Status noch unbekannt ist (dort lohnt Ausprobieren)."""
        W, H = self.point_tile.shape
        tiles = {(x + dx, y + dy) for x, y in self.relics for dx in range(-2, 3) for dy in range(-2, 3)
                 if 0 <= x + dx < W and 0 <= y + dy < H}
        return [t for t in tiles if self.point_tile[t] == -1]

    def units_at(self, pos):
        """Wie viele eigene Einheiten stehen auf pos?"""
        return sum(u.pos[0] == pos[0] and u.pos[1] == pos[1] for u in self.units)
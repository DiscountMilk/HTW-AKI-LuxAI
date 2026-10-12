"""
GameMemory ist das Gedächtnis des Agenten für ein Spiel. `obs` zeigt nur den aktuellen Schritt und nur das Sichtfeld.
GameMemory sammelt über alle Schritte und Matches, was der Agent bisher gelernt hat.
Regeln und RL lesen aus GameMemory anstatt aus obs.

Regel: Nur update() schreibt ins Gedächtnis. Regeln/RL benutzen nur die Abfragen (Abschnitt 5).
"""
from dataclasses import dataclass

import numpy as np

# tile_type-Werte aus obs["map_features"]["tile_type"]
UNKNOWN, EMPTY, NEBULA, ASTEROID = -1, 0, 1, 2


@dataclass
class Unit:
    """Eine eigene Einheit im aktuellen Schritt. Wird von my_units() erzeugt."""
    id: int            # Index 0..max_units-1, Zeile im actions-Array
    pos: np.ndarray    # np.array([x, y])
    energy: int


class GameMemory:
    """Gedächtnis für EIN Spiel. Wird in Agent.set_env_cfg zurückgesetzt."""

    # ------------------------------------------------------------------
    # 1 Zu speichernde Attribute initialisieren
    # ------------------------------------------------------------------
    def __init__(self):
        # Ohne env_cfg kennen wir die Kartengröße noch nicht. Die Arrays legt reset() an.
        # __init__ ruft reset() NICHT auf, damit es nur EINE Liste aller Attribute gibt (in reset).
        self.env_cfg = None
        self.team_id = None

    # ------------------------------------------------------------------
    # 2 reset - einmal pro Spiel, aus Agent.set_env_cfg
    # ------------------------------------------------------------------
    def reset(self, env_cfg):
        """Alle Attribute auf "noch nichts gewusst" setzen.
        Die ersten echten Werte kommen mit dem ersten update(obs) in act()."""
        self.env_cfg = env_cfg
        W, H = env_cfg["map_width"], env_cfg["map_height"]

        # --- Wissen, das über alle Matches eines Spiels bleibt
        self.relics = []                                         # Liste np.array([x, y])
        self.tile_type = np.full((W, H), UNKNOWN, dtype=int)     # zuletzt gesehener Feldtyp
        self.energy = np.zeros((W, H), dtype=int)                # zuletzt gesehene Feld-Energie
        self.last_seen = np.full((W, H), -1, dtype=int)          # Schritt, an dem Feld zuletzt sichtbar war
        self.point_tile = np.full((W, H), -1, dtype=int)         # 1 = Punktefeld, 0 = keins, -1 = unbekannt
        self.enemy_seen_at = np.full((W, H), -100, dtype=int)    # Schritt der letzten Gegner-Sichtung

        # --- Werte des letzten Schritts (für Vorher/Nachher-Vergleiche)
        self.prev_points = 0
        self.point_gain = 0                                      # Punkte-Zuwachs in diesem Schritt

        # --- aktueller Schritt (wird jedes update komplett überschrieben)
        self.step = 0
        self.match_step = 0
        self.points = 0
        self.opp_points = 0
        self.my_mask = np.zeros(env_cfg["max_units"], dtype=bool)
        self.my_pos = np.full((env_cfg["max_units"], 2), -1, dtype=int)
        self.my_energy = np.zeros(env_cfg["max_units"], dtype=int)
        self.opp_mask = np.zeros(env_cfg["max_units"], dtype=bool)
        self.opp_pos = np.full((env_cfg["max_units"], 2), -1, dtype=int)

    # ------------------------------------------------------------------
    # 3 update - jeden Schritt, als Erstes in act()
    # ------------------------------------------------------------------
    def update(self, obs, team_id):
        """Neue Beobachtung einarbeiten. Reihenfolge ist wichtig:
        erst Rohdaten auspacken, dann alles, was darauf aufbaut, zuletzt "prev" merken."""
        self.team_id = team_id
        self._read_current(obs)
        self._update_point_gain()
        self._update_map(obs)
        self._update_relics(obs)
        self._update_enemies()
        self._update_point_tiles()    # braucht relics, my_pos und prev_points
        # ...
        self._remember_prev()          # immer zuletzt

    # ------------------------------------------------------------------
    # 4 Berechnungen aus den Rohdaten - nur von update() benutzt
    # ---------------------------------------------------w---------------
    def _read_current(self, obs):
        """obs auspacken. .copy(), weil obs-Arrays sonst später überschrieben werden könnten."""
        me, opp = self.team_id, 1 - self.team_id
        self.step = int(obs["steps"])
        self.match_step = int(obs["match_steps"])
        self.points = int(obs["team_points"][me])
        self.opp_points = int(obs["team_points"][opp])
        self.my_mask = np.array(obs["units_mask"][me], dtype=bool)
        self.my_pos = np.array(obs["units"]["position"][me], dtype=int)      # unsichtbar = [-1, -1]
        self.my_energy = np.array(obs["units"]["energy"][me], dtype=int)
        self.opp_mask = np.array(obs["units_mask"][opp], dtype=bool)
        self.opp_pos = np.array(obs["units"]["position"][opp], dtype=int)

    def _update_point_gain(self):
        """Wie viele Punkte hat das Team seit dem letzten Schritt bekommen? (Reward für RL)"""
        self.point_gain = 0 if self.match_step == 0 else self.points - self.prev_points

    def _update_map(self, obs):
        """Nur sichtbare Felder überschreiben - Wissen über unsichtbare Felder bleibt."""
        seen = np.array(obs["sensor_mask"], dtype=bool)
        self.tile_type[seen] = np.array(obs["map_features"]["tile_type"])[seen]
        self.energy[seen] = np.array(obs["map_features"]["energy"])[seen]
        self.last_seen[seen] = self.step

    def _update_relics(self, obs):
        """Neu gesehene Relic Nodes zur Liste hinzufügen (nie entfernen)."""
        mask = np.array(obs["relic_nodes_mask"], dtype=bool)
        for pos in np.array(obs["relic_nodes"], dtype=int)[mask]:
            if not any(np.array_equal(pos, r) for r in self.relics):
                self.relics.append(pos)

    def _update_enemies(self):
        for x, y in self.opp_pos[self.opp_mask]:
            self.enemy_seen_at[x, y] = self.step

    def _update_point_tiles(self):
        """Punkte-Zuwachs = Anzahl verschiedener Punktefelder, auf denen eigene Einheiten stehen.
        Zuwachs 0          -> keines der besetzten Felder nahe einer Relic ist ein Punktefeld.
        Zuwachs == Anzahl  -> alle besetzten Felder nahe einer Relic sind Punktefelder."""
        if self.match_step == 0:
            return  # neues Match: Punkte wurden auf 0 gesetzt, Vergleich mit prev ist sinnlos
        gain = self.points - self.prev_points
        occupied = {tuple(p) for p in self.my_pos[self.my_mask] if self._near_relic(p)}
        if not occupied:
            return
        if gain == 0:
            for x, y in occupied:
                self.point_tile[x, y] = 0
        elif gain == len(occupied):
            for x, y in occupied:
                self.point_tile[x, y] = 1
        # ponytail: 0 < gain < len(occupied) wird ignoriert; Upgrade: Kandidaten pro Feld zählen

    def _remember_prev(self):
        self.prev_points = self.points

    def _near_relic(self, pos):
        """Liegt pos im 5x5-Bereich um eine bekannte Relic Node?"""
        return any(np.abs(r - pos).max() <= 2 for r in self.relics)

    # ------------------------------------------------------------------
    # 5 Abfragen - DAS benutzen Regeln und RL (nur lesen, nichts ändern)
    # ------------------------------------------------------------------
    def my_units(self):
        """Alle eigenen, sichtbaren Einheiten als Liste von Unit."""
        return [Unit(int(i), self.my_pos[i], int(self.my_energy[i])) for i in np.where(self.my_mask)[0]]

    def nearest_relic(self, pos):
        """Nächste bekannte Relic Node (Manhattan-Distanz) oder None."""
        if not self.relics:
            return None
        return min(self.relics, key=lambda r: np.abs(r - pos).sum())

    def known_point_tiles(self):
        """Array von [x, y] aller sicher erkannten Punktefelder."""
        return np.argwhere(self.point_tile == 1)

    def is_blocked(self, x, y):
        """True, wenn außerhalb der Karte oder (zuletzt gesehen) Asteroid."""
        W, H = self.tile_type.shape
        return not (0 <= x < W and 0 <= y < H) or self.tile_type[x, y] == ASTEROID

    def recent_enemies(self, max_age=5):
        """Array von [x, y], wo in den letzten max_age Schritten Gegner gesehen wurden."""
        return np.argwhere(self.step - self.enemy_seen_at <= max_age)

from dataclasses import dataclass
import numpy as np


MAP_W, MAP_H = 24, 24
MAX_UNITS = 16
MAX_STEPS_IN_MATCH = 100
SPAWN_RATE = 3
INIT_UNIT_ENERGY, MIN_UNIT_ENERGY, MAX_UNIT_ENERGY = 100, 0, 400
MIN_TILE_ENERGY, MAX_TILE_ENERGY = -20, 20
MAX_RELIC_NODES, RELIC_CONFIG_SIZE, RELIC_SPAWN_MAX_STEP = 6, 5, 50
STAY, UP, RIGHT, DOWN, LEFT, SAP = range(6)
MOVES = {STAY: (0, 0), UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
UNKNOWN, EMPTY, NEBULA, ASTEROID = -1, 0, 1, 2


# ----------------------------------------------------------------------
# MIRROR - die Karte ist an der Nebendiagonale symmetrisch: (x, y) <-> (23 - y, 23 - x).
# Gilt für Feldtypen, Feld-Energie, Relics und Punktefelder (gegen den echten Spielzustand geprüft).
# Was ich auf einer Seite sehe, gilt genauso auf der anderen.
# ----------------------------------------------------------------------
def mirror(x, y):
    """Gespiegeltes Feld."""
    return MAP_W - 1 - y, MAP_H - 1 - x


def mirror_map(m):
    """Ganze (W, H)-Karte spiegeln: m'[mirror(x, y)] = m[x, y]."""
    return m[::-1, ::-1].T


@dataclass
class Unit:
    id: int            # Zeile im actions-Array
    pos: np.ndarray    # np.array([x, y])
    energy: int


class World:
    def __init__(self, env_cfg, team_id: int):

        # FIX env_cfg
        self.team_id = team_id
        self.opp_id = 1 - team_id
        self.match_count_per_episode = env_cfg["match_count_per_episode"]
        self.num_teams = env_cfg["num_teams"]
        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.unit_sap_range = env_cfg["unit_sap_range"]
        self.unit_sensor_range = env_cfg["unit_sensor_range"]

        # VARIABEL - Bleibt über alle Matches eines Spiels (Wissen über Felder, die gerade nicht sichtbar sind)
        self.tile_type = np.full((MAP_W, MAP_H), UNKNOWN)
        self.tile_energy = np.zeros((MAP_W, MAP_H))
        self.last_seen = np.full((MAP_W, MAP_H), -1)    # Schritt, an dem das Feld (oder sein Spiegelfeld) zuletzt sichtbar war
        self.point_tile = np.full((MAP_W, MAP_H), -1)   # 1 ja, 0 nein, -1 unbekannt
        self.relics = set()                             # {(x, y)}
        self.total_points = 0                           # eigene Punkte über alle Matches

        # VARIABEL - Neu für jeden Schritt
        self.steps = -1                                 # -1: noch keine Beobachtung
        self.match_steps = -1
        self.team_points = 0                            # eigene Punkte im aktuellen Match
        self.point_gain = 0                             # eigener Punktezuwachs seit letztem Schritt
        self.team_wins = 0
        self.opp_wins = 0
        self.units = []                                 # eigene Units (Liste von Unit)

    def update(self, obs):
        # 0) Schon eingearbeitet? (z.B. update() und act() bekommen dieselbe obs)
        if int(obs["steps"]) == self.steps:
            return

        # 1) Zeit, Punkte, Siege. team_points startet jedes Match bei 0 -> bei match_steps 0 kein Zuwachs
        self.steps = int(obs["steps"])
        self.match_steps = int(obs["match_steps"])
        points = int(obs["team_points"][self.team_id])
        self.point_gain = points - self.team_points if self.match_steps > 0 else 0
        self.team_points = points
        self.team_wins = int(obs["team_wins"][self.team_id])
        self.opp_wins = int(obs["team_wins"][self.opp_id])
        self.total_points += self.point_gain

        # 2) Eigene Units: nur Zeilen mit units_mask=True sind gültig, der Rest ist -1
        mask = obs["units_mask"][self.team_id].astype(bool)
        pos, energy = obs["units"]["position"][self.team_id], obs["units"]["energy"][self.team_id]
        self.units = [Unit(int(i), np.array(pos[i], dtype=int), int(energy[i])) for i in np.flatnonzero(mask)]

        # 3) Karte: sichtbare Felder übernehmen - auf dem Feld UND auf seinem Spiegelfeld
        seen = obs["sensor_mask"].astype(bool)
        tile_type, tile_energy = obs["map_features"]["tile_type"], obs["map_features"]["energy"]
        for tt, te, s in ((tile_type, tile_energy, seen),
                          (mirror_map(tile_type), mirror_map(tile_energy), mirror_map(seen))):
            self.tile_type[s] = tt[s]
            self.tile_energy[s] = te[s]
            self.last_seen[s] = self.steps
        # ponytail: Asteroiden/Nebel/Energie driften - gemerkte Werte veralten; last_seen sagt, wie alt

        # 4) Relics: neu gesehene merken (nie entfernen) - samt Spiegel-Relic.
        #    Eine neue Relic kann Felder in ihrem 5x5 zu Punktefeldern machen (Relics erscheinen erst
        #    im Lauf der ersten Matches) -> dort "kein Punktefeld" wieder auf "unbekannt" setzen.
        for x, y in obs["relic_nodes"][obs["relic_nodes_mask"].astype(bool)]:
            for rx, ry in ((int(x), int(y)), mirror(int(x), int(y))):
                if (rx, ry) not in self.relics:
                    self.relics.add((rx, ry))
                    area = self.point_tile[max(rx - 2, 0):rx + 3, max(ry - 2, 0):ry + 3]
                    area[area == 0] = -1

        # 5) Punktefelder: Zuwachs = Anzahl besetzter Punktefelder (jedes Feld zählt nur 1x).
        #    Bekannte Punktefelder abziehen; Rest 0 -> alle unbekannten besetzten Felder sind keine,
        #    Rest == Anzahl -> alle sind welche. Ergebnis gilt auch fürs Spiegelfeld.
        if self.match_steps > 0:
            occupied = {(int(u.pos[0]), int(u.pos[1])) for u in self.units if self.near_relic(u.pos)}
            unknown = [p for p in occupied if self.point_tile[p] == -1]
            rest = self.point_gain - sum(self.point_tile[p] == 1 for p in occupied)
            if rest == 0 or rest == len(unknown):
                for p in unknown:
                    self.point_tile[p] = self.point_tile[mirror(*p)] = int(rest > 0)

    # ------------------------------------------------------------------
    # Abfragen - nur lesen
    # ------------------------------------------------------------------
    def near_relic(self, pos):
        """Liegt pos im 5x5 um eine bekannte Relic?"""
        return any(max(abs(pos[0] - x), abs(pos[1] - y)) <= RELIC_CONFIG_SIZE // 2 for x, y in self.relics)

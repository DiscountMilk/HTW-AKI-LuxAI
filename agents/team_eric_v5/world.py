import numpy as np

RELIC_RADIUS = 2   # Punktfelder liegen im 5x5 um eine Relic

# TODO Impl (vorher prüfen ob sinnvoll oder schon implmentiert):
# Mirror Player?
# Globale Werte für Moves und Feldtypen (Nummer zu Value Mappen) - Auch für andere Klassen und Dokumene todo ist es sinnvoll, variablen zu vergeben oder kann system besseer lernen, wenn es nur zahlen hat oder ist es egal?
# weitere Informationen extrahieren?
#   Ist ein Feld ein Potential point_tile? - Alles im Umkreis nahe Relic?

class World:
    def __init__(self, env_cfg, team_id: int):

        # ---------- GAME values (fest für das ganze Spiel) ----------
        self.me, self.opp = team_id, 1 - team_id
        self.W = env_cfg["map_width"]
        self.H = env_cfg["map_height"]
        self.max_units = env_cfg["max_units"]                       # N, Units pro Team
        self.match_count_per_episode = env_cfg["match_count_per_episode"]  # 3 Matches
        self.max_steps_in_match = env_cfg["max_steps_in_match"]     # 100 Züge pro Match
        self.unit_move_cost = env_cfg["unit_move_cost"]             # Energie pro Schritt
        self.unit_sap_cost = env_cfg["unit_sap_cost"]               # Energie pro Schuss
        self.unit_sap_range = env_cfg["unit_sap_range"]             # Schuss-Reichweite
        self.unit_sensor_range = env_cfg["unit_sensor_range"]       # Sichtweite

        W, H, N = self.W, self.H, self.max_units

        # ---------- ZEIT ----------
        self.step = 0            # Zug im ganzen Spiel
        self.match_step = 0      # Zug im aktuellen Match
        self.match_nr = 0        # welches Match gerade läuft (0..2)

        # ---------- MAP values (Gedächtnis der Karte) ----------
        self.tile_type   = np.full((W, H), -1)    # -1 unbekannt, 0 leer, 1 Nebel, 2 Asteroid; (tile_type[5][7]   = 1    → Feld (5,7) ist Nebel)
        self.tile_energy = np.zeros((W, H))       # zuletzt gesehene Feldenergie; tile_energy[5][7] = 3    → Feld (5,7) gibt 3 Energie
        self.last_seen   = np.full((W, H), -1)    # in welchem Zug zuletzt gesehen; last_seen[5][7] = 42   → Feld (5,7) zuletzt in Zug 42 gesehen
        self.visible     = np.zeros((W, H), bool) # Sicht aller deiner Units zusammen: ein Feld ist True, wenn mindestens eine Unit es sieht.
        self.new_tiles   = 0                      # Felder, die im letzten Zug zum ersten Mal gesehen wurden

        # ---------- RELIC values ----------
        self.relic_positions = set()               # bekannte Relics als (x, y)
        self.point_tile = np.full((W, H), -1)
        self.new_relics = 0                        # Relics, die im letzten Zug neu entdeckt wurden

        # ---------- UNIT values: meine ----------
        self.my_pos    = np.full((N, 2), -1)       # Position jeder Unit
        self.my_energy = np.zeros(N)               # Energie jeder Unit; my_energy[1] = 87   → Unit 1 hat 87 Energie
        self.my_alive  = np.zeros(N, bool)
        self.my_respawned = np.zeros(N, bool)     # gestorben und im selben Zug neu gespawnt (gleiche ID, andere Unit)
        self.my_prev_pos   = np.full((N, 2), -1)   # Position / lebt? vor dem letzten update() (für Reward "näher gekommen")
        self.my_prev_alive = np.zeros(N, bool)
        self.sap_targets   = np.full((N, 2), -1)   # Zielfeld des letzten Saps je Unit, -1 = kein Sap (für Reward)

        # ---------- UNIT values: Gegner ----------
        self.opp_pos       = np.full((N, 2), -1)   # zuletzt gesehene Position
        self.opp_energy    = np.zeros(N)           # zuletzt gesehene Energie
        self.opp_visible   = np.zeros(N, bool)     # JETZT sichtbar?
        self.opp_last_seen = np.full(N, -1)

        # ---------- PUNKTE ----------
        self.my_points = 0
        self.opp_points = 0
        self.points_gained = 0   # wie viele Punkte ich im letzten Zug bekommen habe
        self.my_wins = 0
        self.opp_wins = 0
        self.match_result = 0    # im Zug nach Matchende: +1 gewonnen, -1 verloren, sonst 0


    # =====================================================================
    # Update pro Zug
    # =====================================================================
    def update(self, obs):

        # Dieselbe obs zweimal (Training: update() und danach act()) -> nichts tun
        if obs["steps"] == self.step:
            return

        # ---------- ZEIT ----------
        self.step = obs["steps"]
        self.match_step = obs["match_steps"]
        self.match_nr = self.step // (self.max_steps_in_match + 1)

        # Neues Match: Units und Punkte zurücksetzen, Karte und Relics behalten
        if self.match_step == 0:
            self._reset_match()

        # ---------- MAP values (nur sichtbare Felder überschreiben) ----------
        vis = np.array(obs["sensor_mask"], dtype=bool)
        types = np.array(obs["map_features"]["tile_type"])
        energy = np.array(obs["map_features"]["energy"])

        vis_m = self.mirror(vis) & ~vis   # nur Spiegelfelder, die ich NICHT selbst sehe
        self.new_tiles = int(((vis | vis_m) & (self.last_seen < 0)).sum())   # erstmals gesehen (inkl. Spiegel)

        self.visible = vis
        self.tile_type[vis]   = types[vis]
        self.tile_energy[vis] = energy[vis]
        self.last_seen[vis]   = self.step

        # Mirror Map Values

        self.tile_type[vis_m]   = self.mirror(types)[vis_m]
        self.tile_energy[vis_m] = self.mirror(energy)[vis_m]
        self.last_seen[vis_m]   = self.step

        # ---------- RELIC values ----------
        known = len(self.relic_positions)
        for i, seen in enumerate(obs["relic_nodes_mask"]):
            if seen:
                x, y = map(int, obs["relic_nodes"][i])
                self.relic_positions.add((x, y))
                self.relic_positions.add(self.mirror_pos(x, y))
        self.new_relics = len(self.relic_positions) - known

        # ---------- UNIT values: meine ----------
        pos  = np.array(obs["units"]["position"])                    # (T, N, 2)
        en   = np.array(obs["units"]["energy"]).reshape(2, -1)       # (T, N), egal ob (T,N) oder (T,N,1) kommt
        mask = np.array(obs["units_mask"], dtype=bool)               # (T, N), sicher True/False

        jumped = np.abs(pos[self.me] - self.my_pos).sum(1) > 1      # Units ziehen max. 1 Feld
        self.my_respawned = mask[self.me] & self.my_alive & jumped
        self.my_prev_pos, self.my_prev_alive = self.my_pos, self.my_alive
        self.my_pos    = pos[self.me]
        self.my_energy = en[self.me]
        self.my_alive  = mask[self.me]

        # ---------- UNIT values: Gegner (nur sichtbare aktualisieren, Rest bleibt gemerkt) ----------
        o = self.opp
        seen = mask[o]

        self.opp_visible = seen
        self.opp_pos[seen]       = pos[o][seen]
        self.opp_energy[seen]    = en[o][seen]
        self.opp_last_seen[seen] = self.step

        # ---------- PUNKTE ----------
        new_points = obs["team_points"][self.me]

        self.points_gained = new_points - self.my_points
        self.my_points  = new_points
        self.opp_points = int(obs["team_points"][o])
        my_wins, opp_wins = int(obs["team_wins"][self.me]), int(obs["team_wins"][o])
        self.match_result = (my_wins - self.my_wins) - (opp_wins - self.opp_wins)   # +1 gewonnen, -1 verloren
        self.my_wins, self.opp_wins = my_wins, opp_wins

        # ---------- PUNKT-FELDER ableiten ----------
        if self.match_step > 0:          # im Reset-Zug gibt es keine echten Punkte
            self._learn_point_tiles()


    # =====================================================================
    # Abfragen (für Brain / FeatureGenerator)
    # =====================================================================
    def alive_units(self):
        """IDs meiner lebenden Units, z.B. [0, 1, 4]."""
        return np.flatnonzero(self.my_alive).tolist()

    def age(self):
        """Wie alt ist das Wissen pro Feld? -1 = nie gesehen."""
        return np.where(self.last_seen >= 0, self.step - self.last_seen, -1)

    def is_walkable(self, x, y):
        """Auf der Karte und kein Asteroid (unbekannt zählt als begehbar)."""
        return 0 <= x < self.W and 0 <= y < self.H and self.tile_type[x, y] != 2


    # =====================================================================
    # Hilfsmethoden
    # =====================================================================
    def mirror(self, a):
        """Ein ganzes (W, H)-Array spiegeln: Feld (x, y) bekommt den Wert von (W-1-y, H-1-x)."""
        xs, ys = np.indices(a.shape)
        return a[self.W - 1 - ys, self.H - 1 - xs]

    def mirror_pos(self, x, y):
        """Ein einzelnes Feld spiegeln: (x, y) -> (W-1-y, H-1-x)."""
        return self.W - 1 - y, self.H - 1 - x

    def merke_saps(self, actions):
        """Zielfelder der gewählten Saps merken. actions (N, 3) mit [5, dx, dy]."""
        sap = actions[:, 0] == 5
        self.sap_targets = np.where(sap[:, None], self.my_pos + actions[:, 1:], -1)

    def point_tile_candidates(self):
        """(W, H) bool: Feld noch ungeprüft (-1) und nahe einer bekannten Relic -> kann Punkte bringen."""
        near = np.zeros((self.W, self.H), dtype=bool)
        r = RELIC_RADIUS
        for x, y in self.relic_positions:
            near[max(0, x - r): x + r + 1, max(0, y - r): y + r + 1] = True
        return (self.point_tile == -1) & near

    def set_point_tile(self, x, y, value):
        """Punkt-Feld setzen, immer mit Spiegelfeld. value: 0 = nichts, 1 = Punkte."""
        self.point_tile[x, y] = value
        mx, my = self.mirror_pos(x, y)
        self.point_tile[mx, my] = value

    def _reset_match(self):
        """Am Anfang jedes Matches: alles, was pro Match gilt, zurücksetzen."""
        N = self.max_units
        self.my_points = self.opp_points = self.points_gained = 0
        self.opp_pos       = np.full((N, 2), -1)
        self.opp_energy    = np.zeros(N)
        self.opp_visible   = np.zeros(N, bool)
        self.opp_last_seen = np.full(N, -1)
        self.sap_targets   = np.full((N, 2), -1)

    def _learn_point_tiles(self):
        """
        Regel: Jedes besetzte Punkt-Feld gibt 1 Punkt pro Zug
        (mehrere Units auf einem Feld zählen nur einmal).

        points_gained = (sichere Punkt-Felder, die ich besetze) + (unbekannte, die Punkte geben)
          -> Rest == 0                 : alle unbekannten besetzten Felder bringen NICHTS
          -> Rest == Anzahl unbekannte : alle unbekannten besetzten Felder bringen Punkte
          -> sonst                     : nicht eindeutig, nichts eintragen
        """
        tiles = {tuple(p) for p in self.my_pos[self.my_alive]}   # besetzte Felder, ohne Doppelte
        if not tiles:
            return

        known_yes = sum(1 for x, y in tiles if self.point_tile[x, y] == 1)
        unknown   = [(x, y) for x, y in tiles if self.point_tile[x, y] == -1]
        rest = self.points_gained - known_yes

        if not unknown:
            return
        if rest == 0:
            for x, y in unknown:
                self.set_point_tile(x, y, 0)
        elif rest == len(unknown):
            for x, y in unknown:
                self.set_point_tile(x, y, 1)
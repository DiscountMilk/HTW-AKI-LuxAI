"""
features.py = Übersetzer von World -> Zahlen fürs Netz.
Kein Gedächtnis, keine Entscheidungen: liest NUR aus der World.

──────────────────────────────────────────────────────────────────────
IDEE
    Das Netz bekommt die ganze Karte als "Bild" mit mehreren Ebenen (Kanälen),
    wie ein Farbbild Rot/Grün/Blau hat. Jede Ebene zeigt EINE Information.
    Dazu ein kleiner Vektor mit Werten, die für die ganze Karte gelten (Zeit, Punkte).

        encode(world) -> maps     (C, W, H)   z.B. (15, 24, 24)
                         globals  (G,)        z.B. (9,)

AUFBAU
    MAP_PARTS     Liste der Karten-Features    (Methode, Anzahl Kanäle)
    GLOBAL_PARTS  Liste der globalen Features  (Methode, Anzahl Zahlen)
    encode()      ruft alle Methoden der Reihe nach auf und klebt die Ergebnisse zusammen

NEUES FEATURE HINZUFÜGEN (2 Schritte)
    1. Methode schreiben:
         Karte : def mein_feature(self, w) -> np.array der Form (n, W, H)
         Global: def mein_feature(self, w) -> Liste mit n Zahlen
    2. Zeile in MAP_PARTS oder GLOBAL_PARTS eintragen: ("mein_feature", n)

REGELN
    - Werte grob auf 0..1 (oder -1..1) skalieren, sonst lernt das Netz schlecht.
    - Kategorien (z.B. Feldtyp 0/1/2) als eigene Kanäle (One-Hot), nicht als eine Zahl.
    - PARTS ändern = andere Netzgröße -> altes Modell (.pth) umbenennen.
──────────────────────────────────────────────────────────────────────
"""
import numpy as np

# ---------- Skalierung (grobe Maximalwerte, bei Bedarf anpassen) ----------
MAX_UNIT_ENERGY = 400.0     # max. Energie einer Unit
MAX_TILE_ENERGY = 20.0      # grober Bereich der Feldenergie (kann negativ sein)
MEMORY_STEPS    = 50.0      # nach so vielen Zügen gilt altes Wissen als "verblasst"
RELIC_RADIUS    = 2         # Punktfelder liegen im 5x5 um eine Relic


class FeatureEngineering:

    # =====================================================================
    # Feature-Listen: HIER wird festgelegt, was das Netz sieht und in welcher Reihenfolge
    # =====================================================================
    MAP_PARTS = [                  # Methode,       Anzahl Kanäle
        ("my_units",     2),       # meine Units: Anzahl, Energie
        ("opp_units",    2),       # sichtbare Gegner: Anzahl, Energie
        ("opp_memory",   1),       # wo Gegner zuletzt gesehen wurden
        ("tile_type",    4),       # unbekannt, leer, Nebel, Asteroid
        ("tile_energy",  1),       # Energie des Felds
        ("vision",       2),       # jetzt sichtbar, Frische des Wissens
        ("relics",       1),       # Relic-Positionen
        ("point_tiles",  2),       # sicher Punkte, noch unklar
    ]

    GLOBAL_PARTS = [               # Methode,       Anzahl Zahlen
        ("time",         2),       # Fortschritt im Match, im Spiel
        ("score",        3),       # Punktedifferenz, meine Siege, Gegner-Siege
        ("rules",        4),       # Spielregeln aus env_cfg
    ]

    MAP_SIZE    = sum(n for _, n in MAP_PARTS)       # C = Kanäle der Karte
    GLOBAL_SIZE = sum(n for _, n in GLOBAL_PARTS)    # G = Länge des globalen Vektors

    # =====================================================================
    # Hauptfunktion
    # =====================================================================
    def encode(self, w):
        """World -> (maps (C, W, H), globals (G,))."""
        maps = []
        for name, n in self.MAP_PARTS:
            part = np.asarray(getattr(self, name)(w), dtype=np.float32)
            assert part.shape == (n, w.W, w.H), f"{name}: Form {part.shape}, erwartet {(n, w.W, w.H)}"
            maps.append(part)

        globals_ = []
        for name, n in self.GLOBAL_PARTS:
            part = list(getattr(self, name)(w))
            assert len(part) == n, f"{name}: {len(part)} Zahlen, erwartet {n}"
            globals_ += part

        return np.concatenate(maps), np.array(globals_, dtype=np.float32)

    def channel(self, name):
        """Index des ersten Kanals eines Features, z.B. channel('relics') -> 12. Zum Debuggen."""
        i = 0
        for part, n in self.MAP_PARTS:
            if part == name:
                return i
            i += n
        raise KeyError(name)

    # =====================================================================
    # Karten-Features: jede Methode liefert (n, W, H)
    # =====================================================================
    def my_units(self, w):
        """Kanal 0: wie viele meiner Units auf dem Feld stehen. Kanal 1: ihre Energie."""
        m = self._empty(w, 2)
        for i in w.alive_units():
            x, y = w.my_pos[i]
            m[0, x, y] += 1
            m[1, x, y] += w.my_energy[i] / MAX_UNIT_ENERGY
        return m

    def opp_units(self, w):
        """Wie my_units, aber für gerade sichtbare Gegner."""
        m = self._empty(w, 2)
        for i in np.flatnonzero(w.opp_visible):
            x, y = w.opp_pos[i]
            m[0, x, y] += 1
            m[1, x, y] += w.opp_energy[i] / MAX_UNIT_ENERGY
        return m

    def opp_memory(self, w):
        """Wo Gegner zuletzt gesehen wurden: 1 = gerade eben, verblasst auf 0."""
        m = self._empty(w, 1)
        for i in np.flatnonzero(w.opp_last_seen >= 0):
            x, y = w.opp_pos[i]
            fresh = 1 - (w.step - w.opp_last_seen[i]) / MEMORY_STEPS
            m[0, x, y] = max(m[0, x, y], fresh, 0)
        return m

    def tile_type(self, w):
        """One-Hot: pro Feld ist genau einer der 4 Kanäle 1."""
        t = w.tile_type
        return [t == -1, t == 0, t == 1, t == 2]          # unbekannt, leer, Nebel, Asteroid

    def tile_energy(self, w):
        """Zuletzt gesehene Feldenergie, begrenzt auf -1..1."""
        return [np.clip(w.tile_energy / MAX_TILE_ENERGY, -1, 1)]

    def vision(self, w):
        """Kanal 0: sehe ich das Feld JETZT? Kanal 1: wie frisch ist das Wissen (1 = jetzt, 0 = alt/nie)."""
        age = w.age()                                      # -1 = nie gesehen
        fresh = np.where(age >= 0, np.clip(1 - age / MEMORY_STEPS, 0, 1), 0)
        return [w.visible, fresh]

    def relics(self, w):
        """1 an jeder bekannten Relic."""
        m = self._empty(w, 1)
        for x, y in w.relic_positions:
            m[0, x, y] = 1
        return m

    def point_tiles(self, w):
        """Kanal 0: Feld bringt sicher Punkte. Kanal 1: noch unklar und nahe einer Relic (lohnt Ausprobieren)."""
        near = np.zeros((w.W, w.H), dtype=bool)
        r = RELIC_RADIUS
        for x, y in w.relic_positions:
            near[max(0, x - r): x + r + 1, max(0, y - r): y + r + 1] = True
        return [w.point_tile == 1, (w.point_tile == -1) & near]

    # =====================================================================
    # Globale Features: jede Methode liefert eine Liste mit n Zahlen
    # =====================================================================
    def time(self, w):
        """Wie weit Match und Spiel sind: früh erkunden, spät punkten."""
        return [w.match_step / w.max_steps_in_match,
                w.match_nr / w.match_count_per_episode]

    def score(self, w):
        """Punktedifferenz (tanh staucht große Abstände auf -1..1) und Siege."""
        matches = w.match_count_per_episode
        return [np.tanh((w.my_points - w.opp_points) / 20),
                w.my_wins / matches,
                w.opp_wins / matches]

    def rules(self, w):
        """Spielregeln: ändern sich von Spiel zu Spiel, also muss das Netz sie kennen."""
        return [w.unit_move_cost / 5,
                w.unit_sap_cost / 50,
                w.unit_sap_range / 7,
                w.unit_sensor_range / 4]

    # =====================================================================
    # Hilfsfunktion
    # =====================================================================
    @staticmethod
    def _empty(w, n):
        """Leere Karte mit n Kanälen, zum Befüllen."""
        return np.zeros((n, w.W, w.H), dtype=np.float32)
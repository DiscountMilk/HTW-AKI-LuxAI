"""
reward.py = Belohnung aus der World. Liest NUR aus der World, kennt kein Netz.

──────────────────────────────────────────────────────────────────────
IDEE
    Aufruf NACH World.update(next_obs): bewertet, was der letzte Zug gebracht hat.
    Ergebnis ist ein Reward PRO UNIT, weil das Netz pro Unit einen Q-Wert lernt.

        compute_reward(world) -> np.array (max_units,)

    Team-Terme   (gleicher Wert für jede Unit): Punkte, Match-Ergebnis, Erkunden
    Unit-Terme   (eigener Wert pro Unit):       steht auf Punkt-Feld

AUFBAU
    TERMS               Liste (Methode, Gewicht)
    compute_reward()    Summe aller Terme mal Gewicht
    reward_parts()      dieselben Terme einzeln, zum Loggen / Debuggen

NEUER TERM (2 Schritte)
    1. Methode schreiben: def mein_term(w) -> np.array (max_units,)
       (Team-Wert: _team(w, zahl) benutzen)
    2. Zeile in TERMS eintragen: (mein_term, gewicht)

REGELN
    - Gewichte so wählen, dass ein Zug grob -1..1 bringt.
    - Units, die nicht leben, bekommen trotzdem einen Wert; das Modell ignoriert sie.
──────────────────────────────────────────────────────────────────────
"""
import numpy as np


# =====================================================================
# Terme: jede Funktion liefert (max_units,)
# =====================================================================
def _team(w, value):
    """Ein Team-Wert für jede Unit."""
    return np.full(w.max_units, float(value), dtype=np.float32)


def points(w):
    """Punkte, die das Team im letzten Zug bekommen hat (deckt auch noch unbekannte Punkt-Felder ab)."""
    return _team(w, w.points_gained)


def match_result(w):
    """+1 Match gewonnen, -1 verloren (nur im Zug nach Matchende, sonst 0). Das eigentliche Ziel."""
    return _team(w, w.match_result)


def new_tiles(w):
    """Wie viele Felder erstmals gesehen wurden (Erkunden, hört auf, wenn die Karte bekannt ist)."""
    return _team(w, w.new_tiles)


def new_relics(w):
    """Neu entdeckte Relics (Relics erscheinen auch später im Spiel, darum extra)."""
    return _team(w, w.new_relics)


def on_point_tile(w):
    """1, wenn die Unit auf einem sicheren Punkt-Feld steht, geteilt durch die eigenen Units dort (zählt nur 1x)."""
    r = np.zeros(w.max_units, dtype=np.float32)
    alive = w.alive_units()
    tiles = [tuple(w.my_pos[i]) for i in alive]
    for i, (x, y) in zip(alive, tiles):
        if w.point_tile[x, y] == 1:
            r[i] = 1 / tiles.count((x, y))
    return r


TERMS = [                      # Funktion,       Gewicht
    (on_point_tile,  0.5),     # pro Unit: auf Punkt-Feld stehen
    (points,         0.05),    # Team: Punkte im letzten Zug
    (match_result,   5.0),     # Team: Match gewonnen / verloren
    (new_tiles,      0.01),    # Team: Karte erkunden
    (new_relics,     0.5),     # Team: Relic gefunden
]


# =====================================================================
# Hauptfunktionen
# =====================================================================
def reward_parts(w):
    """Jeder Term einzeln, schon mit Gewicht: {name: (max_units,)}."""
    return {f.__name__: weight * f(w) for f, weight in TERMS}


def compute_reward(w):
    """Gesamt-Reward pro Unit: (max_units,)."""
    return sum(reward_parts(w).values()).astype(np.float32)

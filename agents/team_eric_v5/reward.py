"""
reward.py = Belohnung aus der World. Liest NUR aus der World, kennt kein Netz.

──────────────────────────────────────────────────────────────────────
IDEE
    Aufruf NACH World.update(next_obs): bewertet, was der letzte Zug gebracht hat.
    Ergebnis ist ein Reward PRO UNIT, weil das Netz pro Unit einen Q-Wert lernt.

        compute_reward(world) -> np.array (max_units,)

    Team-Terme   (gleicher Wert für jede Unit): Punkte, Match-Ergebnis, Erkunden
    Unit-Terme   (eigener Wert pro Unit):       steht auf Punkt-Feld, steht mit anderen auf einem Feld,
                                                steht auf Feld mit positiver Energie, kommt freiem Punkt-Feld näher

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

from agents.team_eric_v5.features import MAX_TILE_ENERGY


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


def stacked(w):
    """1, wenn die Unit ein Feld mit anderen eigenen Units teilt (Ziel: eine Unit pro Feld). Gewicht negativ."""
    # ponytail: Spawn-Ecke nicht ausgenommen; frisch gespawnte Units zählen kurz mit
    r = np.zeros(w.max_units, dtype=np.float32)
    alive = w.alive_units()
    tiles = [tuple(w.my_pos[i]) for i in alive]
    for i, tile in zip(alive, tiles):
        r[i] = tiles.count(tile) > 1
    return r


def energy_tile(w):
    """0..1 je höher die positive Feldenergie unter der Unit, geteilt durch eigene Units dort (wie on_point_tile)."""
    r = np.zeros(w.max_units, dtype=np.float32)
    alive = w.alive_units()
    tiles = [tuple(w.my_pos[i]) for i in alive]
    for i, (x, y) in zip(alive, tiles):
        r[i] = np.clip(w.tile_energy[x, y] / MAX_TILE_ENERGY, 0, 1) / tiles.count((x, y))
    return r


def toward_point_tile(w):
    """Felder, die die Unit einem FREIEN Ziel näher gekommen ist (+1 näher, -1 weiter weg).
    Ziel = sicheres Punkt-Feld ODER noch ungeprüftes Feld nahe einer Relic (zweite Relic: dort ist noch nichts sicher).
    Frei = keine andere eigene Unit steht drauf -> sind die Felder an Relic 1 besetzt, ziehen übrige Units weiter.
    Potential-Shaping: Hin und zurück ergibt 0, Hinlaufen allein lohnt sich nicht, nur Ankommen."""
    # ponytail: Manhattan-Abstand ignoriert Asteroiden; BFS-Abstand, falls Units an Hindernissen hängen bleiben
    # ponytail: ohne Gamma (Φ_neu - Φ_alt), sonst gäbe es fürs Stehen weit weg einen Mini-Bonus
    r = np.zeros(w.max_units, dtype=np.float32)
    targets = [tuple(t) for t in np.argwhere((w.point_tile == 1) | w.point_tile_candidates())]
    alive = w.alive_units()
    taken = [tuple(w.my_pos[i]) for i in alive]
    for i in alive:
        if not w.my_prev_alive[i] or w.my_respawned[i]:
            continue                                   # neue Unit: kein "vorher"
        me = tuple(w.my_pos[i])
        free = [t for t in targets if taken.count(t) - (t == me) == 0]
        if free:
            free = np.array(free)
            before = np.abs(free - w.my_prev_pos[i]).sum(1).min()
            after = np.abs(free - w.my_pos[i]).sum(1).min()
            r[i] = before - after
    return r


TERMS = [                      # Funktion,       Gewicht
    (on_point_tile,  0.5),     # pro Unit: auf Punkt-Feld stehen
    (stacked,       -0.5),     # pro Unit: Feld mit eigenen Units teilen
    (energy_tile,    0.1),     # pro Unit: auf Feld mit positiver Energie stehen (max 0.1)
    (toward_point_tile, 0.2),  # pro Unit: ein Feld näher an freies Punkt-/Kandidaten-Feld (+0.1 / -0.1)
    (points,         0.05),    # Team: Punkte im letzten Zug
    (match_result,   5.0),     # Team: Match gewonnen / verloren
    (new_tiles,      0.01),    # Team: Karte erkunden
    (new_relics,     0.5),     # Team: Relic gefunden
]

# Energie-Reward darf nie den Punkt-Feld-Reward übersteigen (beide 0..1 und durch Units pro Feld geteilt)
_WEIGHT = {f.__name__: weight for f, weight in TERMS}
assert _WEIGHT["energy_tile"] < _WEIGHT["on_point_tile"], "energy_tile-Gewicht muss kleiner als on_point_tile sein"


# =====================================================================
# Hauptfunktionen
# =====================================================================
def reward_parts(w):
    """Jeder Term einzeln, schon mit Gewicht: {name: (max_units,)}."""
    return {f.__name__: weight * f(w) for f, weight in TERMS}


def compute_reward(w):
    """Gesamt-Reward pro Unit: (max_units,)."""
    return sum(reward_parts(w).values()).astype(np.float32)

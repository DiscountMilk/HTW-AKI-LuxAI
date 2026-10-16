"""
action_mask.py = Welche Aktionen darf eine Unit gerade? Liest NUR aus der World, kennt kein Netz.

    allowed_actions(world, unit) -> np.array (5,) bool, Index = 0 stay, 1 up, 2 right, 3 down, 4 left

    Eine Aktion ist verboten, wenn mindestens eine Regel in RULES True liefert.
    Vorerst ist keine Regel aktiv: alles erlaubt.

NEUE REGEL (2 Schritte)
    1. Funktion schreiben: def meine_regel(w, unit, action) -> True = verboten
    2. In RULES eintragen
"""
import numpy as np

MOVES = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]     # (dx, dy) für stay, up, right, down, left


def no_energy(w, unit, action):
    """Zu wenig Energie für eine Bewegung (STAY kostet nichts)."""
    return action != 0 and w.my_energy[unit] < w.unit_move_cost


def blocked(w, unit, action):
    """Ziel außerhalb der Karte oder (zuletzt gesehener) Asteroid."""
    x, y = w.my_pos[unit]
    dx, dy = MOVES[action]
    return action != 0 and not w.is_walkable(x + dx, y + dy)


RULES = [                      # aktive Regeln; vorerst keine
    # no_energy,
    # blocked,
]


def allowed_actions(w, unit):
    return np.array([not any(rule(w, unit, a) for rule in RULES) for a in range(len(MOVES))])

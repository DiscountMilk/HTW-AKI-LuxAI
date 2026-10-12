"""
- Wird nur bei offensichtlichen Notfällen verwendet.
- Greift nur selten und sind klare Bedingungen
- Greift immer vor Policy!

Öffentlich ist nur check_safety_rules(). Jede Regel ist eine _funktion(unit, memory),
die [aktion, dx, dy] zurückgibt, wenn sie greift, sonst None.

Neue Regel hinzufügen:
  1. def _meine_regel(unit, memory): ... return [aktion, dx, dy] oder None
  2. In SAFETY_RULES an der richtigen Stelle eintragen (oben = höchste Priorität)
"""
import numpy as np

# Aktionen: [aktion, dx, dy]
STAY = [0, 0, 0]
SAP = 5


def check_safety_rules(unit, memory):
    """Prüft alle Safety-Regeln der Reihe nach. Die ERSTE Regel, die greift, gewinnt.
    Rückgabe: [aktion, dx, dy] oder None (keine Regel greift -> policy entscheidet)."""
    for rule in SAFETY_RULES:
        action = rule(unit, memory)
        if action is not None:
            return action
    return None


# ----------------------------------------------------------------------
# Regeln (intern). Alle haben die gleiche Signatur: (unit, memory) -> Aktion oder None
# ----------------------------------------------------------------------
def _no_energy_to_move(unit, memory):
    """Zu wenig Energie für einen Schritt -> stehen bleiben (Bewegung würde sowieso scheitern)."""
    if unit.energy < memory.env_cfg["unit_move_cost"]:
        return STAY
    return None


def _sap_enemy_in_range(unit, memory):
    """Sichtbarer Gegner in Sap-Reichweite und genug Energie -> auf den nächsten Gegner schießen.
    Reichweite ist ein Quadrat: max(|dx|, |dy|) <= unit_sap_range."""
    sap_range = memory.env_cfg["unit_sap_range"]
    # ponytail: 2x Sap-Kosten als Reserve, damit die Einheit danach nicht festsitzt; Wert tunen
    if unit.energy < 2 * memory.env_cfg["unit_sap_cost"]:
        return None
    enemies = memory.opp_pos[memory.opp_mask]               # nur sichtbare Gegner
    if len(enemies) == 0:
        return None
    offsets = enemies - unit.pos                             # (n, 2) relative Positionen
    dist = np.abs(offsets).max(axis=1)
    in_range = offsets[dist <= sap_range]
    if len(in_range) == 0:
        return None
    dx, dy = min(in_range, key=lambda o: np.abs(o).max())    # nächster Gegner
    return [SAP, int(dx), int(dy)]


# Reihenfolge = Priorität. Oben steht, was zuerst geprüft wird.
SAFETY_RULES = [
    _no_energy_to_move,
    _sap_enemy_in_range,
]

"""
Handgeschriebene Regeln. Zwei Arten, zwei Klassen:

SafetyRules : Regel ENTSCHEIDET. decide(unit) -> [aktion, dx, dy] oder None.
              Greift eine Regel, wird das Netz übersprungen.
              Nur für Fälle, in denen die Regel SICHER die beste Entscheidung ist.
ActionMask  : Regel VERBIETET. allowed(unit) -> bool[5], Index = Aktion (STAY..LEFT).
              Streicht nur Unsinn. Das Netz wählt unter dem Rest und lernt weiter.

Neue Regel: Methode in der passenden Klasse schreiben und in self.rules eintragen.
"""
import numpy as np
from agents.team_eric_v2.world import MOVES, SAP, STAY


class SafetyRules:
    def __init__(self, world):
        self.w = world
        self.rules = [            # Reihenfolge = Priorität, oben zuerst

        ]

    def decide(self, unit):
        """Aktion der ERSTEN Regel, die greift. Keine greift -> None (Netz entscheidet)."""
        for rule in self.rules:
            action = rule(unit)
            if action is not None:
                return action
        return None

    # ------------------------------------------------------------------
    # Regeln: (unit) -> [aktion, dx, dy] oder None
    # ------------------------------------------------------------------

    # TODO regeln implementieren...




class ActionMask:
    def __init__(self, world):
        self.w = world
        self.rules = [
            self.no_energy_to_move,
            self.blocked,
        ]

    def allowed(self, unit):
        """bool[5], Index = Aktion. True = erlaubt. STAY ist immer erlaubt,
        damit das Netz immer mindestens eine Wahl hat."""
        ok = np.array([not any(rule(unit, a) for rule in self.rules) for a in MOVES])
        ok[STAY] = True
        return ok

    # ------------------------------------------------------------------
    # Regeln: (unit, action) -> True = verboten
    # ------------------------------------------------------------------

    # TODO regeln implementieren...
    def no_energy_to_move(self, unit, action):
        """Zu wenig Energie -> Bewegung würde scheitern."""
        return action != STAY and unit.energy < self.w.cfg["unit_move_cost"]

    def blocked(self, unit, action):
        """Nicht aus der Karte oder in einen (zuletzt gesehenen) Asteroiden laufen."""
        dx, dy = MOVES[action]
        return self.w.is_blocked(unit.pos[0] + dx, unit.pos[1] + dy)
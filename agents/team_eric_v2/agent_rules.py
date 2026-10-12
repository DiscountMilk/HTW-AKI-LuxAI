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
from agents.team_eric_v2.agent_world import MOVES, SAP, STAY


class SafetyRules:
    def __init__(self, world):
        self.w = world
        self.rules = [            # Reihenfolge = Priorität, oben zuerst

        ]
        self.counts = {rule.__name__: 0 for rule in self.rules}   # wie oft hat welche Regel entschieden

    def decide(self, unit):
        """Aktion der ERSTEN Regel, die greift. Keine greift -> None (Netz entscheidet)."""
        for rule in self.rules:
            action = rule(unit)
            if action is not None:
                self.counts[rule.__name__] += 1 # Logging-Zähler
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
            self.multiple_player_on_one_spot,
        ]
        self.counts = {rule.__name__: 0 for rule in self.rules}   # wie oft hat welche Regel etwas verboten

    def allowed(self, unit):
        """bool[5], Index = Aktion. True = erlaubt. Ist alles verboten, wird STAY wieder erlaubt,
        damit das Netz immer mindestens eine Wahl hat."""
        ok = np.ones(len(MOVES), dtype=bool)
        for a in MOVES:
            for rule in self.rules:      # alle Regeln prüfen (kein any), damit jede einzeln gezählt wird
                if rule(unit, a):
                    self.counts[rule.__name__] += 1
                    ok[a] = False
        if not ok.any():
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
        """Nicht aus der Karte oder in einen (zuletzt gesehenen) Asteroiden laufen.
        STAY ist nie blockiert, auch nicht, wenn ein Asteroid auf die Einheit gewandert ist."""
        if action == STAY:
            return False
        dx, dy = MOVES[action]
        return self.w.is_blocked(unit.pos[0] + dx, unit.pos[1] + dy)

    def multiple_player_on_one_spot(self, unit, action):
        """Stehen mehrere eigene Einheiten auf einem Feld, darf nur die mit der kleinsten ID stehen bleiben.
        Die anderen müssen weg: ein Feld zählt nur einmal Punkte, und ein Stapel ist ein leichtes Sap-Ziel."""
        if action != STAY or self.w.units_at(unit.pos) < 2:
            return False
        first = min(u.id for u in self.w.units if u.pos[0] == unit.pos[0] and u.pos[1] == unit.pos[1])
        return unit.id != first


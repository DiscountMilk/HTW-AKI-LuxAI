"""
Schnittstelle zum Runner (lux/runner.py). Enthält KEINE Spiellogik, nur den Ablauf.

Pro Spiel :  __init__ -> [set_to_eval_mode] -> set_env_cfg -> act, (update), act, (update), ...
Pro act() :  1. world.update   2. aus letztem Schritt lernen   3. pro Einheit entscheiden
Entscheiden: SafetyRules entscheidet -> sonst wählt Brain unter den Aktionen der ActionMask
"""
import os

import numpy as np

from agents.team_eric_v2.agent_brain import get_brain
from agents.team_eric_v2.agent_rules import ActionMask, SafetyRules
from agents.team_eric_v2.training_log import ACTION_NAMES, TrainingLog
from agents.team_eric_v2.agent_world import World

# ponytail: spielen zwei team_eric_v2 gegeneinander, schreiben beide in dieselbe Datei
LOG_PATH = os.path.join(os.path.dirname(__file__), "training_log.csv")


class Agent:
    def __init__(self, player, name="Team Eric v2", agent_path=None):
        self.player = player                               # "player_0" oder "player_1"
        self.name = name                                   # erscheint im Replay-Viewer
        self.team = 0 if player == "player_0" else 1       # Index in den obs-Arrays
        # Leerer Load_Path in Paths.csv kommt als NaN an -> None
        self.brain = get_brain(agent_path if isinstance(agent_path, str) else None)  # lebt über alle Spiele
        self.training = True
        self.last = {}                                     # unit_id -> (features, aktion) vom letzten Schritt
        # Brauchen die Kartengröße -> erst in set_env_cfg
        self.world = self.safety = self.mask = None
        # Zähler fürs Training-Log. Der Agent wird pro Spiel neu gebaut -> zählen pro Spiel
        self.points = 0                                    # Teampunkte über alle Matches
        self.rewards = []                                  # Reward jeder gelernten Netz-Entscheidung
        self.rule_decisions = 0                            # wie oft SafetyRules entschieden hat
        self.picks = np.zeros(len(ACTION_NAMES), dtype=int)   # wie oft das Netz welche Aktion gewählt hat

    # ------------------------------------------------------------------
    # 1 Runner-Schnittstelle
    # ------------------------------------------------------------------
    def set_to_eval_mode(self):
        """Vom Runner bei train=False, VOR set_env_cfg."""
        self.training = False
        self.brain.eval_mode()

    def set_env_cfg(self, env_cfg):
        """Einmal pro Spiel, vor dem ersten act()."""
        self.world = World(env_cfg)
        self.safety = SafetyRules(self.world)
        self.mask = ActionMask(self.world)

    def act(self, step, obs):
        """Jeden Schritt. Rückgabe: int-Array (max_units, 3), pro Einheit [aktion, dx, dy]."""
        self.world.update(obs, self.team)
        self.points += self.world.point_gain
        if self.training:
            self._learn_from_last_step()
        actions = np.zeros((self.world.max_units, 3), dtype=int)
        for unit in self.world.units:
            actions[unit.id] = self._decide(unit)
        return actions

    def update(self, terminated=False, next_obs=None, **_):
        """Vom Runner nach jedem env.step, nur bei train=True."""
        self.brain.train()
        if terminated:          # Runner speichert nur VOR Spielen, nie am Ende -> selbst speichern
            self.brain.save()
            self._log_game(next_obs)

    def save_agent(self):
        self.brain.save()

    # ------------------------------------------------------------------
    # 2 Intern
    # ------------------------------------------------------------------
    def _decide(self, unit):
        action = self.safety.decide(unit)                  # Regel entscheidet?
        if action is not None:
            self.rule_decisions += 1
            return action
        s = self.brain.features(unit, self.world)
        i = self.brain.pick(s, self.mask.allowed(unit))    # Netz wählt unter Erlaubtem
        self.last[unit.id] = (s, i)                        # nur Netz-Entscheidungen werden gelernt
        self.picks[i] += 1
        return [i, 0, 0]                                   # Index = Lux-Aktion (STAY..LEFT)

    def _learn_from_last_step(self):
        """Was ist aus den Netz-Entscheidungen des letzten Schritts geworden? -> Erfahrung speichern."""
        # ponytail: Übergang über Match-Grenze wird verworfen (1 Transition pro Match fehlt)
        if self.world.match_step > 0:
            for unit in self.world.units:
                if unit.id in self.last:
                    s, a = self.last[unit.id]
                    r = self.brain.reward(unit, self.world)
                    s2 = self.brain.features(unit, self.world)
                    self.brain.add(s, a, r, s2)
                    self.rewards.append(r)
        self.last = {}

    def _log_game(self, final_obs):
        """Kennzahlen dieses Spiels ins Training-Log. Neue Kennzahl: hier eine Zeile ergänzen."""
        wins, decisions = final_obs["team_wins"], max(self.picks.sum(), 1)
        TrainingLog(LOG_PATH).add({
            "siege": int(wins[self.team]),
            "niederlagen": int(wins[1 - self.team]),
            "gewonnen": int(wins[self.team] > wins[1 - self.team]),
            "punkte": self.points,
            "reward": float(np.mean(self.rewards)) if self.rewards else 0.0,   # Ø pro Netz-Entscheidung TODO wasbedeutet das?
            "relics": len(self.world.relics),
            "punktefelder": len(self.world.point_tiles()), # TODO was wird hier gezählt?
            "regel": self.rule_decisions / max(self.rule_decisions + self.picks.sum(), 1),
            **self.brain.pop_stats(),                                          # loss, q
            **{f"safety_{k}": v / max(self.rule_decisions + self.picks.sum(), 1)   # Anteil aller Entscheidungen
               for k, v in self.safety.counts.items()},
            **{f"mask_{k}": v / decisions for k, v in self.mask.counts.items()},  # Ø verbotene Aktionen pro Netz-Entscheidung
            "epsilon": self.brain.epsilon,
            **{name: self.picks[a] / decisions for a, name in enumerate(ACTION_NAMES)},
        })
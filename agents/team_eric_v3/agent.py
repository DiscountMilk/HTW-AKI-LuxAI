"""
Schnittstelle zum Runner (lux/runner.py). Enthält KEINE Spiellogik, nur den Ablauf.

Pro Spiel :  __init__ -> [set_to_eval_mode] -> set_env_cfg -> act, update, act, update, ...
act()     :  World aktualisieren -> pro Unit entscheiden (Brain wählt unter der ActionMask)
update()  :  nur bei train=True: World aktualisieren -> aus letztem Schritt lernen -> Lernschritt
"""
import os

import numpy as np

from agents.team_eric_v3.brain import ActionMask, get_brain
from agents.team_eric_v3.logger import Logger
from agents.team_eric_v3.world import MAX_UNITS, World


class Agent:
    def __init__(self, player: str, name="Team Eric V3", agent_path=None):
        self.player = player                               # liest der Runner ("player_0" / "player_1")
        self.name = name                                   # Name im Replay-Viewer
        self.team_id = 0 if player == "player_0" else 1
        # Leerer Load_Path in Paths.csv kommt als NaN an -> None
        self.brain = get_brain(agent_path if isinstance(agent_path, str) else None)   # lebt über alle Spiele
        self.logger = Logger()

    # ---------------------------------
    # Runner-Schnittstelle
    # ---------------------------------
    def set_to_eval_mode(self):
        """Vom Runner bei train=False, vor set_env_cfg: keine Zufallszüge."""
        self.brain.eval_mode()

    def set_env_cfg(self, env_cfg):
        """Einmal pro Spiel, vor dem ersten act()."""
        self.world = World(env_cfg, self.team_id)
        self.action_mask = ActionMask(self.world)          # hält nur einen Verweis auf world -> immer aktuell
        self.brain.new_game()                              # (s, a) vom letzten Spiel nicht mit diesem verknüpfen

    def act(self, step, obs):
        """Jeden Schritt: Wissen aktualisieren, pro Unit entscheiden. -> int[MAX_UNITS, 3]"""
        self.world.update(obs)                             # im Training schon in update() passiert -> No-op
        actions = np.zeros((MAX_UNITS, 3), dtype=int)
        for unit in self.world.units:
            actions[unit.id] = self.brain.decide(unit, self.world, self.action_mask.allowed(unit))
        return actions

    def update(self, next_obs, terminated=False, **_):
        """Nur bei train=True, nach jedem env.step: aus dem letzten Schritt lernen."""
        self.world.update(next_obs)
        self.brain.learn_from(self.world)
        self.brain.train()
        if terminated:                                     # Runner speichert nur VOR Spielen, nie am Ende
            self.brain.save()
            self.logger.log_game(self)

    def save_agent(self):
        self.brain.save()

    def load_agent(self):
        if self.brain.path and os.path.isfile(self.brain.path):
            self.brain.load_agent()

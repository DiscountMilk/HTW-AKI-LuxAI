"""
Schnittstelle zum Runner. Ruft nur die anderen auf.

Ablauf pro Spiel (siehe lux/runner.py):
  __init__  -> set_env_cfg  -> act, act, act, ...  (jeden Schritt)
Der Agent wird pro Spiel NEU erzeugt. Alles in self ist danach weg.
"""
import numpy as np

from agents.team_eric_test.memory import GameMemory
from agents.team_eric_test.safety_rules import check_safety_rules


class Agent:
    def __init__(self, player: str, name: str = "Team Eric Test", agent_path=None):
        self.player = player                              # "player_0" oder "player_1"
        self.name = name                                  # erscheint im Replay-Viewer
        self.team_id = 0 if player == "player_0" else 1   # Index in den obs-Arrays
        self.agent_path = agent_path                      # Pfad zu RL-Gewichten (später)
        self.env_cfg = None

        # Gedächtnis nur ERZEUGEN. Die Kartengröße kennen wir erst in set_env_cfg.
        self.memory = GameMemory()

        # Policy braucht env_cfg -> wird erst in set_env_cfg erzeugt.
        self.policy = None
        # Der Runner ruft set_to_eval_mode() VOR set_env_cfg() auf. Darum hier merken
        # und erst beim Erzeugen der Policy anwenden.
        self.eval = False

    def set_env_cfg(self, env_cfg):
        """Einmal pro Spiel, direkt nach env.reset(), vor dem ersten act().
        Speichert die festen Spielregeln und legt das Gedächtnis in Kartengröße an."""
        self.env_cfg = env_cfg
        self.memory.reset(env_cfg)

    def act(self, step, obs):
        """Jeden Schritt. Rückgabe: int-Array (max_units, 3).
        Die 3 Spalten pro Einheit sind [aktion, dx, dy]:
          aktion 0 = stehen, 1 = hoch, 2 = rechts, 3 = runter, 4 = links, 5 = sap
          dx, dy = Zielfeld relativ zur Einheit, nur für sap, sonst 0.
        Es werden also NICHT alle möglichen Aktionen ausgelesen, sondern
        hier wird pro Einheit EINE Aktion eingetragen. Zeilen ohne Einheit bleiben 0."""
        self.memory.update(obs, self.team_id)            # 1. Wissen aktualisieren - IMMER zuerst

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit in self.memory.my_units():
            a = check_safety_rules(unit, self.memory)     # 2. Pflicht-Regeln (feste Regeln, die immer gelten)
            if a is None:
                a = self.policy.act(unit, self.memory)   # 3. eigentliche Entscheidung
            actions[unit.id] = a
        return actions
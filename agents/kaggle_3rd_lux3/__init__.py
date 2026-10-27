# 3. Platz Lux AI S3 (andreyd41/lux3-bot, Release 0.4.8_fp, Apache-2.0).
# Imitation Learning (2 UNets, trainiert auf "Frog Parade"-Replays).
# Hier nur ein Adapter auf die Kurs-Runner-Schnittstelle.
from pathlib import Path

from .agent import Agent as _Lux3Agent
from .agent.base import Global

WEIGHTS_DIR = str(Path(__file__).parent / "weights")


class Agent:
    def __init__(self, player: str, name: str = "Kaggle_3rd_lux3") -> None:
        self.player = player
        self.name = name
        # ponytail: Bot speichert Spielwissen in Klassen-Globals -> nicht gegen sich selbst spielen
        Global.clear()

    def set_env_cfg(self, env_cfg):
        self.bot = _Lux3Agent(self.player, env_cfg, WEIGHTS_DIR)

    def act(self, step: int, obs):
        return self.bot.act(step, obs)

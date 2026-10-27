# 8. Platz Lux AI S3 (gregorlied/lux-s3, Release v73a, Apache-2.0).
# Imitation Learning (UNet-Policy + SAP-Netz). Adapter auf die Kurs-Runner-Schnittstelle.
from .bot import Agent as _GregorAgent
from .game import Global

# Werte aus agent_fn() in bot.py (Submission-IDs der imitierten Top-Teams)
AGENT_SUB_ID = "43276830"
AGENT_SUB_IDS = [["43330490", "43330358", "43320130", "43317109"], ["43294433", "43293811"], ["43276830"], ["43212846", "43212163"], ["43277936"]]


class Agent:
    def __init__(self, player: str, name: str = "Kaggle_8th_gregor") -> None:
        self.player = player
        self.name = name
        # ponytail: Bot speichert Spielwissen in Klassen-Globals -> nicht gegen sich selbst spielen
        Global.reset()

    def set_env_cfg(self, env_cfg):
        self.bot = _GregorAgent(self.player, env_cfg, AGENT_SUB_ID, AGENT_SUB_IDS)

    def act(self, step: int, obs):
        self.bot.process(step, obs, 60)
        return self.bot.act(step, obs, 60)

# Sherlock (elhassanmahmoud/sherlock-luxAI-team15), nur als Sparringspartner.
# Eingefroren: lernt nie, speichert nie (sonst überschreibt train=True seine Gewichte).
from .main import Agent_x


class Agent(Agent_x):
    def __init__(self, player: str, agent_path=None) -> None:
        super().__init__(player, agent_path=agent_path)
        self.set_to_eval_mode()

    def update(self, *_, **__):
        pass

    def save_agent(self, *_, **__):
        pass

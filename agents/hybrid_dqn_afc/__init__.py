# "Two Angels for Charlie" Agent 7 (BlackMagicCoder/luxai-s3-hybrid-agent).
# Regelbasiert + DQN (epsilon_rl = Anteil Einheiten, die das DQN steuert). Checkpoint afc_dqn_v3_1.
import os

from .agent_afc_dqn import AfcAgent

WEIGHTS = os.path.join(os.path.dirname(__file__), "afc_dqn_v3_1.pth")


class Agent(AfcAgent):
    def __init__(self, player: str, name: str = "Hybrid_DQN_AFC") -> None:
        # 80 % DQN / 20 % Regeln war laut results.md die beste Mischung
        super().__init__(player, name=name, epsilon_rl=0.8, load_path=WEIGHTS, save_path=os.devnull)
        self.set_to_eval_mode()  # eingefroren: nur Sparringspartner

    def update(self, *_, **__):
        pass

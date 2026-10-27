"""Training von team_eric_v5 gegen wechselnde Gegner in einem Durchlauf.

Pro Block: Gegner ziehen (gewichtet), v5 immer player_0, BLOCK Spiele mit train=True.
Das Netz von v5 bleibt im Prozess erhalten (get_model-Cache) und wird nach jedem Spiel gespeichert.
Nur Gegner ohne Lernen in den Pool (3, 4-8 haben update() und würden ihre .pth überschreiben).
"""
import random

from lux.runner import run_agents

ME = 13                 # team_eric_v5 (Paths.csv)
POOL = {                # Moodle-Gruppe: Ziehgewicht
    1: 1,               # DefaultAgent
    2: 3,               # Solution_92_Kaggle
    11: 2,              # Hybrid_DQN_AFC (eingefroren)
    12: 3,              # Sherlock (eingefroren)
    9: 2,               # Kaggle 3. Platz
    10: 2,              # Kaggle 8. Platz
}
BLOCKS = 200            # Anzahl Gegnerwechsel
BLOCK = 5              # Spiele pro Block

if __name__ == "__main__":
    for block in range(BLOCKS):
        opp = random.choices(list(POOL), weights=list(POOL.values()))[0]
        print(f"=== Block {block + 1}/{BLOCKS}: {ME} vs {opp}")
        run_agents(ME, opp, games_to_play=BLOCK, train=True, save_frequency=BLOCK, matches_per_game=3)   # v5 immer player_0

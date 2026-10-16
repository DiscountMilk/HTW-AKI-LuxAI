import numpy as np

from agents.team_eric_v4.world import World
from agents.team_eric_v4.features import FeatureEngineering
from agents.team_eric_v4.action_mask import allowed_actions
from agents.team_eric_v4.model import get_model, N_ACTIONS
from agents.team_eric_v4.reward import compute_reward
from agents.team_eric_v4.logger import Logger


class Agent:

    def __init__(self, player: str, name="Team Eric V4", agent_path=None):
        self.player = player
        self.name = name
        self.team_id = 0 if player == "player_0" else 1
        self.path = agent_path if isinstance(agent_path, str) else None   # NaN aus CSV -> None
        self.model = get_model(self.path)                     # bleibt über alle Spiele, beide Spieler teilen es
        self.features = FeatureEngineering() # TODO check hier richtig
        self.training = True
        self.last = None


    def set_to_eval_mode(self):
        self.training = False


    def set_env_cfg(self, env_cfg):
        self.world = World(env_cfg, self.team_id)             # neues Gedächtnis pro Spiel
        self.logger = Logger(self.model)                      # neue Zähler pro Spiel
        self.last = None


    def act(self, step, obs):
        self.world.update(obs)
        x = self.features.encode(self.world)# TODO check ob encode richtig ist
        w = self.world
        mask = self._masks()
        actions = self.model.choose(x, w.my_pos, w.my_alive, mask, explore=self.training)
        self.logger.on_act(actions, w.my_alive, mask)
        if not self.training:                                 # Runner ruft dann kein update() auf
            self.logger.eval_step(self.team_id, w, self.last)

        # Zustand merken, für update(): wo stand welche Unit, wer lebte, was wurde gewählt
        self.last = dict(x=x, pos=w.my_pos.copy(), alive=w.my_alive.copy(), actions=actions.copy())
        return actions


    def update(self, next_obs, terminated=False, **_):
        self.world.update(next_obs)
        if self.last is None:
            return
        self.logger.on_update(self.world, self.last["alive"])
        self.model.train_step(
            **self.last,                                        # x, pos, alive, actions  (vor dem Zug)
            reward=compute_reward(self.world),                  # (max_units,)
            next_x=self.features.encode(self.world),            # nach dem Zug
            next_pos=self.world.my_pos.copy(),
            next_alive=self.world.my_alive & ~self.world.my_respawned,   # dieselbe Unit lebt noch
            next_mask=self._masks(),                            # erlaubte Aktionen im neuen Zustand
            done=terminated,
        )
        if terminated:
            self.save_agent()
            self.logger.log_game(self.team_id, self.world, self.model)


    def _masks(self):
        """Erlaubte Aktionen aller Units (N, A). Tote Units: alles erlaubt (wird nicht benutzt)."""
        m = np.ones((self.world.max_units, N_ACTIONS), dtype=bool)
        for unit in self.world.alive_units():
            m[unit] = allowed_actions(self.world, unit)
        return m


    def save_agent(self):
        if self.path:
            self.model.save(self.path)


    def load_agent(self):
        self.model.load(self.path)
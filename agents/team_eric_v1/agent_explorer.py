import torch
import torch.nn as nn
# ----------------------------
# Q-Network
# ----------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Agent
# ----------------------------
from typing import Optional
import os
import torch.optim as optim
import numpy as np
import agents.team_eric_v1.feat_generator as fg

class AgentExplorer:

    def __init__(
            self,
            player: str,
            name: str='Agent Explorer',
            learning_rate: float = 1e-3, # TODO
            gamma: float = 0.99, # TODO ?
            epsilon: float = 0.1, # TODO Prozentuale Zufallsaktion
            agent_path: Optional[str] = None,
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch config
        self.agent_path = agent_path # Speicherort der Gewichte für save_agent() merken
        self.env_cfg = None # Spiele config e.a. map_width, units, ... (Werte erst nach Spielinitialisierung)

        # Spieler und Team Info
        self.name = name
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

        # Network Config
        self.state_dim = 17 # TODO Anzahl Input Features
        self.n_actions = 5 # TODO Anzahl Output Features

        # Network Init
        self.gamma = gamma
        self.epsilon = epsilon

        self.q = QNet(self.state_dim, self.n_actions).to(self.device)
        if agent_path is not None:
            self.load_agent(agent_path)
        self.opt = optim.Adam(self.q.parameters(), lr=learning_rate)


    def save_agent(self):
        """
        Speichert Agenten Status
        """
        torch.save(self.q.state_dict(), self.agent_path)


    def load_agent(self, load_path):
        """
        Lädt Agenten, falls vorhanden
        """
        if os.path.isfile(load_path):
            model = torch.load(load_path)
            self.q.load_state_dict(model)
        else:
            print(f"Agent load path {load_path} does not exist. Starting with untrained model.")


    def set_to_eval_mode(self):
        """
        - witch to eval mode (no exploration, no dropout, etc)
        - LuxAI Methode
        """
        self.epsilon = 0.0
        self.q.eval()


    def set_env_cfg(self, env_cfg):
        """
        - Gibt Game env config an den Agenten befor das erste Spiel startet (enthält: map size, unit parameter, match length)
        - LuxAI Methode
        """
        self.env_cfg = env_cfg


    def _epsilon_greedy(self, state_vec: np.ndarray) -> int:
        """
        Wählt eine Zufallsaktion, aus dem gegebenen Aktionsraum
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q(s)
            return int(torch.argmax(q_vals, dim=1).item())


    def _online_td_update(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        """
        Lernfunktion des Networks
        """
        if len(batch_states) == 0: # Wenn es nichts gibt, dann wird auch nichts gelernt
            return

        # TODO hier noch anpassungen vornehmen????
        states = torch.as_tensor(np.stack(batch_states, axis=0), dtype=torch.float32, device=self.device)   # (B, S) Lage jeder Einheit vor dem Zug
        next_states = torch.as_tensor(np.stack(batch_next_states, axis=0), dtype=torch.float32, device=self.device)  # (B, S) Lage jeder Einheit nach dem Zug
        actions = torch.as_tensor(np.array(batch_actions), dtype=torch.long, device=self.device)             # (B,) welcher Zug gemacht wurde (als Nummer)
        rewards = torch.as_tensor(np.array(batch_rewards), dtype=torch.float32, device=self.device).unsqueeze(1)     # (B,1) Belohnung
        dones = torch.as_tensor(np.array(batch_dones, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1) 1=Einheit tot / Spiel Vorbei; else 0

        q_pred_all = self.q(states)  # (B, A)

        with torch.no_grad():
            next_q_all = self.q(next_states)
            max_next_q = next_q_all.max(dim=1, keepdim=True).values  # (B,1)
            target_vals = rewards + (1.0 - dones) * self.gamma * max_next_q  # (B,1)

            q_target_all = q_pred_all.detach().clone()
            q_target_all[torch.arange(q_target_all.size(0), device=self.device), actions] = target_vals.squeeze(1)

        loss = nn.functional.mse_loss(q_pred_all, q_target_all)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.opt.step()

    def act(self, step: int, obs):
        """
        LuxAI API
        Produce actions for all available units.
        Returns: np.array shape (max_units, 3) with per-unit [direction, 0, 0].
        """

        # find available units
        ally_mask, _, _ = fg.get_ally_arrays(self.team_id, obs)
        available_unit_ids = np.where(ally_mask)[0]

        # initialize actions array
        actions = np.zeros((int(self.env_cfg["max_units"]), 3), dtype=int)

        # per-unit action selection
        for uid in available_unit_ids:

            # build state vector for this unit
            state_vec = fg.state_for_unit(self, uid, step, obs)

            # select action with epsilon-greedy
            # if exploration, then choose random action
            # otherwise calculate Q-values with QNet and choose best action
            action_dir = self._epsilon_greedy(state_vec)

            actions[uid, 0] = action_dir  # movement dir 0..4
            actions[uid, 1] = 0 # no Sap action
            actions[uid, 2] = 0 # no Sap action

        return actions
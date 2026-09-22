import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import agents.DQN.feat_generator as fg

# ----------------------------
# Q-Network (shared across all units)
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
# Lux DQN Agent (online TD, tailored features)
# ----------------------------
class Agent:
    """
    - Movement-only policy: outputs direction in {0..4}; other action slots set to 0.
    - Online TD(0) update each step from (obs, action, reward, next_obs, terminated).
    - Feature vector is built ONLY from fields present in your obs/env_cfg.
    """
    def __init__(
        self,
        player: str,
        name: str='DQN Agent',
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        agent_path: Optional[str] = None,
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_path = agent_path
        self.env_cfg = None

        # --- identity / teams
        self.player = player
        self.name = name
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        # --- per-unit policy setup
        self.state_dim = 17
        self.n_actions = 5  # movement only

        # --- initialize network, optimizer, etc.
        self.q = QNet(self.state_dim, self.n_actions).to(self.device)
        if agent_path is not None:
            self.load_agent(agent_path)
        self.opt = optim.Adam(self.q.parameters(), lr=learning_rate)

        # --- set exploration + discount
        self.gamma = gamma
        self.epsilon = epsilon

    # store agent state
    def save_agent(self):
        torch.save(self.q.state_dict(), self.agent_path)

    # load agent state
    def load_agent(self, load_path):
        # if file exists, load model state dict into Q-network
        if os.path.isfile(load_path):
            model = torch.load(load_path)
            self.q.load_state_dict(model)
        else:
            print(f"Agent load path {load_path} does not exist. Starting with untrained model.")

    # switch to eval mode (no exploration, no dropout, etc)
    def set_to_eval_mode(self):
        self.epsilon = 0.0
        self.q.eval()

    # --------- env config (map size, max units, etc)
    def set_env_cfg(self, env_cfg):
        self.env_cfg = env_cfg

    # --------- action selection
    def _epsilon_greedy(self, state_vec: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q(s)
            return int(torch.argmax(q_vals, dim=1).item())

    # --------- learning
    def _online_td_update(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        if len(batch_states) == 0:
            return

        states = torch.as_tensor(np.stack(batch_states, axis=0), dtype=torch.float32, device=self.device)   # (B, S)
        next_states = torch.as_tensor(np.stack(batch_next_states, axis=0), dtype=torch.float32, device=self.device)  # (B, S)
        actions = torch.as_tensor(np.array(batch_actions), dtype=torch.long, device=self.device)             # (B,)
        rewards = torch.as_tensor(np.array(batch_rewards), dtype=torch.float32, device=self.device).unsqueeze(1)     # (B,1)
        dones = torch.as_tensor(np.array(batch_dones, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)

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

    # --------- PUBLIC API expected by the Lux environment
    def act(self, step: int, obs):
        """
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

    def update(self, step: int, obs, reward: float, action, terminated: bool, next_obs):
        """
        Called AFTER the env applies the previous action.
        Learns from (obs, action, reward, next_obs, terminated).
        """
        action_arr = np.array(action)  # expected shape (max_units, 3)
        ally_mask_obs, _, _ = fg.get_ally_arrays(self.team_id, obs)
        ally_mask_next, _, _ = fg.get_ally_arrays(self.team_id, next_obs)

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        for uid in np.where(ally_mask_obs)[0]:
            s = fg.state_for_unit(self, uid, step, obs)

            # First action slot = direction 0..4
            try:
                a_dir = int(action_arr[uid][0])
            except Exception:
                continue
            if a_dir < 0 or a_dir >= self.n_actions:
                continue

            ns = fg.state_for_unit(self, uid, step + 1, next_obs)
            unit_done = bool(terminated or (uid >= len(ally_mask_next) or not ally_mask_next[uid]))

            batch_states.append(s)
            batch_actions.append(a_dir)
            batch_rewards.append(float(reward))
            batch_next_states.append(ns)
            batch_dones.append(1.0 if unit_done else 0.0)

        self._online_td_update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)

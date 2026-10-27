"""Replay Buffer (vorallokierter Ringpuffer).

Transitions werden PRO EINHEIT gespeichert: ein Env-Step liefert bis zu 16
Transitions. Der Puffer bricht die zeitliche Korrelation der Samples auf
(Voraussetzung fuer stabiles DQN-Training).
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.s = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.ns = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.mask = np.ones((self.capacity, 1), dtype=bool)  # wird in push ersetzt
        self._mask_dim = 1
        self.pos = 0
        self.size = 0

    def _ensure_mask(self, n_actions: int) -> None:
        if self._mask_dim != n_actions:
            self.mask = np.ones((self.capacity, n_actions), dtype=bool)
            self._mask_dim = n_actions

    def push(self, s, a, r, ns, done, next_mask) -> None:
        self._ensure_mask(len(next_mask))
        i = self.pos
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.ns[i] = ns
        self.done[i] = float(done)
        self.mask[i] = next_mask
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.s[idx], self.a[idx], self.r[idx], self.ns[idx],
                self.done[idx], self.mask[idx])

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------- persistenz
    def save(self, path: str) -> None:
        """Gefuellten Teil speichern (aelteste Transition zuerst).

        Ohne gespeicherten Buffer startet Weitertraining mit leerem Puffer und
        lernt nur auf wenigen frischen Transitions -> das Netz verlernt
        (beobachtet in der Verbesserungsrunde vom 10.10.)."""
        if self.size < self.capacity:
            idx = np.arange(self.size)
        else:
            idx = (np.arange(self.size) + self.pos) % self.capacity
        np.savez(path, s=self.s[idx], ns=self.ns[idx], a=self.a[idx], r=self.r[idx],
                 done=self.done[idx], mask=self.mask[idx])

    def load(self, path: str) -> int:
        """Transitions aus `path` anhaengen. Gibt die Anzahl geladener zurueck."""
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}  # npz liest sonst bei jedem Zugriff neu
        if d["s"].shape[1] != self.obs_dim:
            raise ValueError(f"Buffer {path}: obs_dim {d['s'].shape[1]} != {self.obs_dim}")
        n = len(d["a"])
        for i in range(n):
            self.push(d["s"][i], d["a"][i], d["r"][i], d["ns"][i], d["done"][i], d["mask"][i])
        return n

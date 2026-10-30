"""Selbsttest Sap:  .venv/bin/python -m agents.team_eric_v5.test_sap"""
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from agents.team_eric_v5.action_mask import can_sap
from agents.team_eric_v5.features import FeatureEngineering as F
from agents.team_eric_v5.model import Model, N_ACTIONS, SAP, _sap_area
from agents.team_eric_v5.reward import sap_result

W = H = 24
N = 16


def fake_x(rng):
    return rng.random((F.MAP_SIZE, W, H), dtype=np.float32), rng.random(F.GLOBAL_SIZE, dtype=np.float32)


def test_area():
    a = _sap_area(np.array([[0, 0], [10, 10]]), 3, (W, H))
    assert a[0].sum() == 16 and a[1].sum() == 49            # Ecke abgeschnitten, Mitte 7 x 7
    assert a[1, 7, 13] and not a[1, 6, 10]
    assert _sap_area(np.array([[10, 10]]), np.array([1]), (W, H)).sum() == 9


def test_old_weights():
    """Alte .pth mit 5 Ausgängen: Bewegungs-Q-Werte bleiben exakt gleich."""
    rng = np.random.default_rng(0)
    m = Model()
    x = fake_x(rng)
    state = {k: v.clone() for k, v in m.net.state_dict().items()}
    last = [k for k in state if state[k].shape[:1] == (N_ACTIONS + 1,)]
    assert len(last) == 2                                   # Gewicht + Bias der letzten Schicht
    for k in last:
        state[k] = state[k][:N_ACTIONS]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "alt.pth")
        torch.save({"net": state, "steps": 123}, path)
        before = m.net(*m._tensors([x[0]], [x[1]]))[:, :N_ACTIONS]
        neu = Model()
        neu.load(path)
    after = neu.net(*neu._tensors([x[0]], [x[1]]))[:, :N_ACTIONS]
    assert torch.equal(before, after) and neu.steps == 123
    assert torch.equal(neu.net(*neu._tensors([x[0]], [x[1]]))[0, SAP], neu.target(*neu._tensors([x[0]], [x[1]]))[0, SAP])


def choose_with(sap_value, sap_ok, explore=False, peak=50):
    """Modell, dessen Sap-Karte überall sap_value ist, außer +peak auf Feld (12, 9). Bewegungen: Q = 0."""
    m = Model()
    pos = np.full((N, 2), -1)
    pos[:2] = [[10, 10], [20, 20]]
    alive = np.zeros(N, bool)
    alive[:2] = True
    out = np.zeros((N_ACTIONS + 1, W, H), dtype=np.float32)
    out[SAP] = sap_value
    out[SAP, 12, 9] = sap_value + peak
    m.net = lambda *_: torch.as_tensor(out)[None]
    x = fake_x(np.random.default_rng(1))
    return m.choose(x, pos, alive, np.ones((N, N_ACTIONS), bool), np.array([sap_ok] * N), 3, explore=explore)


def test_choose():
    a = choose_with(5.0, True)                              # Sap schlägt Bewegungen (Q = 0)
    assert a[0].tolist() == [SAP, 2, -1]                    # bestes Feld in Reichweite
    assert a[1, 0] == SAP and abs(a[1, 1:]).max() <= 3      # (12, 9) nicht in Reichweite: anderes Feld
    assert choose_with(5.0, False)[0, 0] != SAP             # zu wenig Energie
    assert choose_with(-5.0, True, peak=0)[0, 0] != SAP     # Bewegung besser

    np.random.seed(0)
    picks = np.concatenate([choose_with(-5.0, True, explore=True, peak=0)[:2, 0] for _ in range(300)])   # ε = 1
    assert 0.08 < (picks == SAP).mean() < 0.28              # Sap zählt wie eine von 6 Aktionen


def test_learn():
    """Lernschritt läuft, Q für Sap wird am Zielfeld gelesen."""
    rng = np.random.default_rng(2)
    m = Model()
    pos = np.tile([10, 10], (N, 1))
    actions = np.zeros((N, 3), int)
    actions[:] = (SAP, 2, -1)
    alive, nope = np.ones(N, bool), np.zeros(N, bool)
    for _ in range(40):
        m.train_step(fake_x(rng), pos, alive, actions, np.ones(N, np.float32), fake_x(rng), pos, alive,
                     np.ones((N, N_ACTIONS), bool), alive, 3, False)
        m.train_step(fake_x(rng), pos, alive, actions, np.ones(N, np.float32), fake_x(rng), pos, nope,
                     np.ones((N, N_ACTIONS), bool), nope, 3, True)
    assert m.updates > 0 and np.isfinite(m.loss_sum)
    assert (m.buffer.data["sap_target"][0, 0] == [12, 9]).all()


def test_reward():
    w = SimpleNamespace(max_units=3, sap_targets=np.array([[5, 5], [9, 9], [-1, -1]]),
                        opp_pos=np.array([[5, 5], [6, 5], [0, 0]]), opp_visible=np.array([True, True, False]))
    assert sap_result(w).tolist() == [0.5, -1, 0]           # 1 + 0,5 - 1; Fehlschuss; kein Sap
    w.opp_visible = np.zeros(3, bool)
    assert sap_result(w).tolist() == [-1, -1, 0]            # keine Gegner sichtbar


def test_can_sap():
    w = SimpleNamespace(my_alive=np.array([True, True, False]), my_energy=np.array([30, 29, 99]), unit_sap_cost=30)
    assert can_sap(w).tolist() == [True, False, False]


if __name__ == "__main__":
    for test in (test_area, test_old_weights, test_choose, test_learn, test_reward, test_can_sap):
        test()
    print("test_sap.py ok")

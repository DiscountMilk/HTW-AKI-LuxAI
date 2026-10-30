"""
mirror.py = Spieler 1 sieht die Welt wie Spieler 0. Nur Eingang (obs) und Ausgang (Aktionen).

    obs_als_spieler0(obs)      Observation von Spieler 1 -> Sicht von Spieler 0 (Kopie)
    aktionen_zurueck(actions)  im Spiegelraum gewählte Aktionen -> echte Aktionen von Spieler 1

Spiegelung (Karte ist symmetrisch): (x, y) -> (W-1-y, H-1-x). Startecke (W-1, H-1) wird (0, 0).
Teams werden getauscht: Index 1 (ich) wird Index 0.
Richtungen: up <-> right, down <-> left. Sap-Ziel (dx, dy) -> (-dy, -dx).
Alles dahinter (World, Features, Reward, Maske, Modell) bleibt unverändert.
"""
import numpy as np

FLIP_ACTION = np.array([0, 2, 1, 4, 3, 5])     # stay, up->right, right->up, down->left, left->down, sap


def mirror_map(a):
    """(W, H)-Array: Feld (x, y) bekommt den Wert von (W-1-y, H-1-x). Wie World.mirror."""
    a = np.asarray(a)
    W, H = a.shape
    xs, ys = np.indices(a.shape)
    return a[W - 1 - ys, H - 1 - xs]


def mirror_pos(p, W, H):
    """(..., 2)-Positionen spiegeln; -1 (unsichtbar / tot) bleibt -1."""
    p = np.asarray(p)
    out = np.stack([W - 1 - p[..., 1], H - 1 - p[..., 0]], axis=-1)
    return np.where((p < 0).any(axis=-1, keepdims=True), -1, out)


def swap(a):
    """Teams tauschen: Index 0 <-> 1."""
    return np.asarray(a)[::-1].copy()


def obs_als_spieler0(obs):
    """Kopie der Observation von Spieler 1, so umgebaut, dass er sich als Spieler 0 sieht."""
    W, H = np.asarray(obs["sensor_mask"]).shape
    return {
        **obs,
        "units": {
            "position": mirror_pos(swap(obs["units"]["position"]), W, H),
            "energy": swap(obs["units"]["energy"]),
        },
        "units_mask": swap(obs["units_mask"]),
        "team_points": swap(obs["team_points"]),
        "team_wins": swap(obs["team_wins"]),
        "sensor_mask": mirror_map(obs["sensor_mask"]),
        "map_features": {k: mirror_map(v) for k, v in obs["map_features"].items()},
        "relic_nodes": mirror_pos(obs["relic_nodes"], W, H),
        # unverändert: steps, match_steps, relic_nodes_mask
    }


def aktionen_zurueck(actions):
    """(N, 3) [typ, dx, dy] aus dem Spiegelraum -> echte Aktionen."""
    a = np.array(actions, copy=True)
    a[:, 0] = FLIP_ACTION[a[:, 0]]
    a[:, 1], a[:, 2] = -actions[:, 2], -actions[:, 1]
    return a


if __name__ == "__main__":
    # Selbsttest:  .venv/bin/python -m agents.team_eric_v5.mirror
    W = H = 24
    MOVES = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]   # wie action_mask.MOVES
    rng = np.random.default_rng(0)

    # 1. Zweimal spiegeln = Original
    pos = rng.integers(0, 24, (2, 16, 2))
    pos[rng.random((2, 16)) < 0.3] = -1                  # unsichtbare / tote Units: (-1, -1)
    obs = dict(steps=7, match_steps=7,
               units=dict(position=pos, energy=rng.integers(0, 400, (2, 16))),
               units_mask=rng.random((2, 16)) < 0.5, team_points=np.array([3, 9]), team_wins=np.array([1, 0]),
               sensor_mask=rng.random((W, H)) < 0.5,
               map_features=dict(energy=rng.integers(-20, 20, (W, H)), tile_type=rng.integers(0, 3, (W, H))),
               relic_nodes=np.array([[3, 5], [-1, -1]]), relic_nodes_mask=np.array([True, False]))
    back = obs_als_spieler0(obs_als_spieler0(obs))
    for k in ("units_mask", "team_points", "team_wins", "sensor_mask", "relic_nodes"):
        assert np.array_equal(back[k], obs[k]), k
    for k in ("position", "energy"):
        assert np.array_equal(back["units"][k], obs["units"][k]), k
    for k in obs["map_features"]:
        assert np.array_equal(back["map_features"][k], obs["map_features"][k]), k

    # 2. Startecke von Spieler 1 wird (0, 0), -1 bleibt -1, Teams getauscht
    assert mirror_pos(np.array([W - 1, H - 1]), W, H).tolist() == [0, 0]
    assert mirror_pos(np.array([-1, -1]), W, H).tolist() == [-1, -1]
    m = obs_als_spieler0(obs)
    assert m["team_points"].tolist() == [9, 3] and m["steps"] == 7

    # 3. Zug im Spiegelraum, zurückgespiegelt = echter Zug mit zurückgedrehter Aktion
    for x in range(1, W - 1):
        for y in range(1, H - 1):
            mx, my = mirror_pos(np.array([x, y]), W, H)
            for a, (dx, dy) in enumerate(MOVES):
                moved = mirror_pos(np.array([mx + dx, my + dy]), W, H)     # Spiegelung ist ihre eigene Umkehrung
                rdx, rdy = MOVES[FLIP_ACTION[a]]
                assert moved.tolist() == [x + rdx, y + rdy], (x, y, a)

    # 4. Sap-Ziel: Spiegelraum (dx, dy) -> echt (-dy, -dx)
    assert aktionen_zurueck(np.array([[5, 2, -1]])).tolist() == [[5, 1, -2]]
    # 5. Sap im Spiegelraum trifft dasselbe echte Feld
    for x, y, tx, ty in [(5, 7, 8, 9), (20, 3, 17, 6), (12, 12, 12, 12)]:
        mx, my = mirror_pos(np.array([x, y]), W, H)
        mtx, mty = mirror_pos(np.array([tx, ty]), W, H)
        real = aktionen_zurueck(np.array([[5, mtx - mx, mty - my]]))[0]
        assert [x + real[1], y + real[2]] == [tx, ty], (x, y, tx, ty)
    print("mirror.py ok")

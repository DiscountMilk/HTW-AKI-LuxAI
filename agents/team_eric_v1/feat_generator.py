import numpy as np
from typing import Tuple, Optional

def get_ally_arrays(team_id, obs):
    unit_mask = np.array(obs["units_mask"][team_id])  # (max_units,)
    pos = np.array(obs["units"]["position"][team_id])  # (max_units, 2)
    ene = np.array(obs["units"]["energy"][team_id])  # (max_units, 1) or (max_units,)
    if ene.ndim == 2 and ene.shape[1] == 1:
        ene = ene[:, 0]
    return unit_mask, pos, ene


def get_enemy_arrays(opp_team_id, obs, max_units):
    try:
        unit_mask = np.array(obs["units_mask"][opp_team_id])  # (max_units,)
        pos = np.array(obs["units"]["position"][opp_team_id])  # (max_units, 2)
        ene = np.array(obs["units"]["energy"][opp_team_id])
        if ene.ndim == 2 and ene.shape[1] == 1:
            ene = ene[:, 0]
        return unit_mask, pos, ene
    except Exception:
        # Some env wrappers might hide opponent info — degrade gracefully
        N = max_units
        return np.zeros(N, dtype=bool), np.zeros((N, 2), dtype=int), np.zeros(N, dtype=float)


def _nearest_point(W, H, from_xy: Tuple[int, int], points: np.ndarray, mask: Optional[np.ndarray] = None):
    """
    Returns (dx, dy, dist) to the nearest point among 'points' (shape: (N,2)).
    If 'mask' is provided, only consider where mask==True.
    If nothing valid, returns (0, 0, (W+H)) as a large distance placeholder.
    """
    if points is None or len(points) == 0:
        return 0.0, 0.0, float(W + H)
    if mask is not None:
        ids = np.where(mask)[0]
        if len(ids) == 0:
            return 0.0, 0.0, float(W + H)
        pts = points[ids]
    else:
        pts = points

    fx, fy = from_xy
    dists = np.abs(pts[:, 0] - fx) + np.abs(pts[:, 1] - fy)
    k = int(np.argmin(dists))
    tx, ty = pts[k]
    dx, dy = float(tx - fx), float(ty - fy)
    return dx, dy, float(dists[k])


def state_for_unit(agent, unit_id: int, step: int, obs) -> np.ndarray:
    '''
    Per-unit state vector (length 17). Let W,H be map width/height, maxE the max unit energy.

    1) x/W — Unit x-position, normalized [0,1].
    2) y/H — Unit y-position, normalized [0,1].
    3) energy/maxE — Unit energy normalized [0,1].
    4) step_frac — Episode progress (step / episode_length) ∈ [0,1).

    Relics (visible this step):
    5) has_visible_relic — 1 if any relic_nodes_mask is True, else 0.
    6) nearest_relic_dx/W — Δx to nearest visible relic, normalized (0 if none).
    7) nearest_relic_dy/H — Δy to nearest visible relic, normalized (0 if none).
    8) nearest_relic/(W+H) — Distance to nearest visible relic, normalized (1 if none).
    9) visible_relic_frac — Fraction of relic slots currently visible ∈ [0,1].

    Map geometry (edges):
    10) dist_left/W — Distance to left edge, normalized.
    11) dist_right/W — Distance to right edge, normalized.
    12) dist_top/H — Distance to top edge, normalized.
    13) dist_bottom/H — Distance to bottom edge, normalized.

    Nearby units:
    14) nearest_ally/(W+H) — Distance to nearest ally (excluding self), normalized (1 if none).
    15) nearest_enemy/(W+H) — Distance to nearest visible enemy, normalized (1 if none).
    16) nearest_enemy_dx/W — Δx to nearest visible enemy, normalized (0 if none).
    17) nearest_enemy_dy/H — Δy to nearest visible enemy, normalized (0 if none).
    '''
    ally_mask, ally_pos, ally_ene = get_ally_arrays(agent.team_id, obs)
    if unit_id >= len(ally_mask) or not ally_mask[unit_id]:
        return np.zeros(agent.state_dim, dtype=np.float32)

    max_units = int(agent.env_cfg["max_units"])
    H = int(agent.env_cfg["map_height"])
    W = int(agent.env_cfg["map_width"])
    max_energy = float(agent.env_cfg.get("max_unit_energy", 100.0))

    enemy_mask, enemy_pos, _ = get_enemy_arrays(agent.opp_team_id, obs, max_units)

    # core self info
    x, y = map(int, ally_pos[unit_id])
    e = float(ally_ene[unit_id])

    # relic info (current visibility)
    relic_mask = np.array(obs["relic_nodes_mask"])
    relic_pos = np.array(obs["relic_nodes"])
    has_visible_relic = 1.0 if relic_mask.any() else 0.0
    total_relic_slots = max(1, len(relic_mask))
    visible_relic_frac = float(np.sum(relic_mask)) / float(total_relic_slots)

    # nearest visible relic node
    r_dx, r_dy, r_dist = _nearest_point(W, H,(x,y), relic_pos, mask=relic_mask)

    # borders (distance to each edge)
    dist_left = float(x)
    dist_right = float(W - 1 - x)
    dist_top = float(y)
    dist_bottom = float(H - 1 - y)

    # nearest *ally* (excluding self)
    if np.sum(ally_mask) > 1:
        other_ids = np.where(ally_mask)[0]
        others = ally_pos[other_ids]
        # remove self
        others = others[other_ids != unit_id] if unit_id in other_ids else others
        a_dx, a_dy, a_dist = _nearest_point(W, H,(x, y), others)
    else:
        a_dx, a_dy, a_dist = 0.0, 0.0, float(W + H)

    # nearest *enemy* (if visible)
    e_dx, e_dy, e_dist = _nearest_point(W, H, (x, y), enemy_pos, mask=enemy_mask)

    # normalization helpers
    W = max(1.0, float(W))
    H = max(1.0, float(H))
    WH = max(1.0, float(W + H))
    maxE = max(1.0, float(max_energy))

    nx, ny = x / W, y / H
    ne = min(e / maxE, 1.0)

    nr_dx, nr_dy = r_dx / W, r_dy / H
    nr_d = r_dist / WH

    nleft, nright = dist_left / W, dist_right / W
    ntop, nbottom = dist_top / H, dist_bottom / H

    na_d = a_dist / WH
    ne_d = e_dist / WH
    ne_dx, ne_dy = e_dx / W, e_dy / H

    step_total = (
            agent.env_cfg.get("max_steps_in_match")
            or agent.env_cfg.get("episode_length")
            or (agent.env_cfg.get("max_steps") or 1)
    )
    step_frac = float((step % max(1, step_total))) / float(max(1, step_total))

    feat = np.array([
        nx, ny, ne, step_frac,
        has_visible_relic,
        nr_dx, nr_dy, nr_d,
        visible_relic_frac,
        nleft, nright, ntop, nbottom,
        na_d,
        ne_d, ne_dx, ne_dy
    ], dtype=np.float32)

    return feat
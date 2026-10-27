"""Feature-Engineering: ein Vektor pro Einheit, alles im kanonischen Raum.

`make_context()` berechnet einmal pro Step alle gemeinsamen Groessen
(Karten, Zielzuweisung, Gegnerinfos). `build_features()` liest daraus pro Unit
den Vektor. Das haelt die Kosten bei 16 Units pro Step klein.

FEATURE_NAMES dokumentiert das Layout exakt -> fuer die Praesentation einfach
`python -m scripts.show_features` (bzw. den Namen ausgeben) nutzen.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from .belief import TILE_ASTEROID

MAX_UNIT_ENERGY = 400.0   # feste Engine-Konstante (nicht randomisiert)
MAX_TILE_ENERGY = 20.0
N_MATCHES = 3

_DIR_NAMES = ["c", "u", "r", "d", "l"]


def _names() -> list[str]:
    n = [
        # --- eigene Lage + bekannte Spielparameter (10)
        "self_x", "self_y", "self_energy", "self_low_energy",
        "match_step_frac", "match_idx_frac",
        "cfg_move_cost", "cfg_sap_cost", "cfg_sap_range", "cfg_sensor_range",
        # --- Punktestand (4)
        "own_points", "point_diff", "own_wins", "opp_wins",
    ]
    # --- naechstes Ziel je Klasse (5 x 4 = 20)
    for cls in ["point", "cand", "relic", "unexp", "assigned"]:
        n += [f"{cls}_dx", f"{cls}_dy", f"{cls}_dist", f"{cls}_exists"]
    # --- aktuelles Tile (5)
    n += ["on_point", "on_cand", "on_nebula", "allies_here", "redundant_here"]
    # --- lokale Umgebung: 5 Richtungen x 6 Kanaele (30)
    for d in _DIR_NAMES:
        n += [f"{d}_passable", f"{d}_energy", f"{d}_point", f"{d}_cand",
              f"{d}_unexp", f"{d}_allies"]
    # --- Gegner (6)
    n += ["enemy_dx", "enemy_dy", "enemy_dist", "enemy_energy_rel",
          "enemies_in_sap", "enemies_visible"]
    # --- globale Zaehler (4)
    n += ["own_units", "known_points", "explored_frac", "cand_frac"]
    return n


FEATURE_NAMES = _names()
FEATURE_DIM = len(FEATURE_NAMES)


def _nearest(coords: np.ndarray, x: int, y: int, W: int, H: int):
    """(dx/W, dy/H, dist/(W+H), exists) zum naechsten Punkt in `coords`."""
    if coords is None or len(coords) == 0:
        return 0.0, 0.0, 1.0, 0.0
    d = np.abs(coords[:, 0] - x) + np.abs(coords[:, 1] - y)
    k = int(np.argmin(d))
    return (
        float(coords[k, 0] - x) / W,
        float(coords[k, 1] - y) / H,
        float(d[k]) / (W + H),
        1.0,
    )


def grid_targets(coords: np.ndarray, block: int) -> np.ndarray:
    """Pro block x block-Feld das unerkundete Tile, das der Blockmitte am naechsten liegt."""
    if len(coords) == 0:
        return coords
    keys = (coords[:, 0] // block) * 1000 + coords[:, 1] // block
    centre = (coords // block) * block + (block - 1) / 2.0
    d = np.abs(coords - centre).sum(axis=1)
    order = np.lexsort((d, keys))
    first = np.ones(len(order), dtype=bool)
    first[1:] = keys[order][1:] != keys[order][:-1]
    return coords[order[first]]


def assign_targets(belief, unit_ids, unit_pos, W, H,
                   enemy_half_penalty: float = 0.0,
                   seen_before_penalty: float = 0.0,
                   grid: int = 0, stale: int = 0) -> dict[int, tuple[int, int]]:
    """Greedy-Zuweisung: jede Unit bekommt ein EIGENES Ziel.

    Prioritaet: bewiesene Punkte-Tiles < Kandidaten-Tiles < unerkundete Tiles.
    Eindeutige Ziele verhindern, dass alle Units auf dasselbe Tile laufen
    (ein Tile bringt maximal einen Punkt, Slide 38) und liefern gleichzeitig
    das Potential fuer das Reward-Shaping (siehe reward.py).

    enemy_half_penalty > 0: Ziele in der gegnerischen Kartenhaelfte werden
    teurer (cfg.home_first). Mit Symmetrie kennt der Belief auch die Punkte-
    Tiles beim Gegner; ohne Malus laufen Units dorthin und sterben (Analyse
    11.10.: 7,2 statt 5,8 Tote pro Match).

    seen_before_penalty > 0 (cfg.explore_memory): unerkundete Tiles, die schon in
    einem frueheren Match gesehen wurden, sind als Erkundungsziel teurer. Sonst
    laeuft der Agent jedes Match wieder in dieselbe (falsche) Ecke, weil
    `explored` pro Match zurueckgesetzt wird (Analyse 11.10.: auf manchen Karten
    in allen 3 Matches kein Relic gefunden).

    grid > 0 (cfg.explore_grid): Erkundungsziele nur je ein Tile pro grid x grid-
    Block, alle Bloecke statt nur die naechsten am Schwerpunkt -> die Units
    verteilen sich ueber die Karte statt als Klumpen zu wandern.
    stale > 0 (cfg.explore_stale): siehe Belief.explore_coords.
    """
    if len(unit_ids) == 0:
        return {}
    pools = [belief.point_coords(), belief.candidate_coords()]
    targets = []
    prios = []
    for prio, pool in enumerate(pools):
        if len(pool):
            targets.append(pool)
            prios.append(np.full(len(pool), prio, dtype=np.int32))
    n_hard = int(sum(len(t) for t in targets))
    if n_hard < len(unit_ids):
        unexp = belief.explore_coords(stale) if stale else belief.unexplored_coords()
        if grid:
            unexp = grid_targets(unexp, grid)
        if len(unexp) == 0:
            unexp = belief.relic_coords()
        if len(unexp):
            # nur die naechsten Kandidaten betrachten, damit die Matrix klein bleibt
            centre = unit_pos.mean(axis=0)
            d = np.abs(unexp[:, 0] - centre[0]) + np.abs(unexp[:, 1] - centre[1])
            seen = np.zeros(len(unexp))
            if seen_before_penalty:
                seen = belief.explored_ever[unexp[:, 0], unexp[:, 1]] * float(seen_before_penalty)
            order = np.argsort(d + seen, kind="stable") if seen_before_penalty else np.argsort(d)
            keep = order if grid else order[: max(len(unit_ids) * 4, 16)]
            targets.append(unexp[keep])
            prios.append(np.full(len(keep), 2, dtype=np.int32))
            extra_seen = seen[keep]
    if not targets:
        return {}
    tgt = np.concatenate(targets, axis=0)
    prio = np.concatenate(prios, axis=0)
    extra = np.zeros(len(tgt))
    if seen_before_penalty and prio[-1] == 2:
        extra[len(tgt) - len(extra_seen):] = extra_seen

    # Kostenmatrix (Units x Ziele); Prioritaet dominiert die Distanz
    cost = (
        np.abs(unit_pos[:, None, 0] - tgt[None, :, 0])
        + np.abs(unit_pos[:, None, 1] - tgt[None, :, 1])
        + prio[None, :] * 1000
        + extra[None, :]
    ).astype(np.float64)
    if enemy_half_penalty:
        # eigener Spawn: kanonisch immer (0, 0); ohne Spiegelung fuer Team 1 (W-1, H-1)
        s = tgt[:, 0] + tgt[:, 1]
        own_origin = belief.canonical or belief.team_id == 0
        enemy = s > W - 1 if own_origin else s < W - 1
        cost += enemy[None, :] * float(enemy_half_penalty)

    out: dict[int, tuple[int, int]] = {}
    n_assign = min(len(unit_ids), len(tgt))
    for _ in range(n_assign):
        flat = int(np.argmin(cost))
        ui, ti = divmod(flat, cost.shape[1])
        if not np.isfinite(cost[ui, ti]):
            break
        out[int(unit_ids[ui])] = (int(tgt[ti, 0]), int(tgt[ti, 1]))
        cost[ui, :] = np.inf
        cost[:, ti] = np.inf
    return out


def make_context(agent, obs, belief=None) -> dict:
    """Alle pro Step einmalig benoetigten Groessen (kanonischer Raum).

    `belief` ueberschreibt agent.belief (Ablation no_belief: Features aus einer
    gedaechtnislosen Sicht, Reward weiter aus dem vollen Belief).
    """
    belief = belief if belief is not None else agent.belief
    W, H = belief.W, belief.H
    cfg = agent.env_cfg or {}
    team, opp = belief.team_id, belief.opp_team_id

    ally_mask = np.asarray(obs["units_mask"], dtype=bool)[team]
    enemy_mask = np.asarray(obs["units_mask"], dtype=bool)[opp]
    pos_all = np.asarray(obs["units"]["position"], dtype=np.int32)
    ene_all = np.asarray(obs["units"]["energy"], dtype=np.float32)
    if ene_all.ndim == 3 and ene_all.shape[-1] == 1:
        ene_all = ene_all[..., 0]
    ally_pos, enemy_pos = pos_all[team], pos_all[opp]
    ally_ene, enemy_ene = ene_all[team], ene_all[opp]
    if belief.mirrored:
        ally_pos = geo.mirror_positions(ally_pos, W, H)
        enemy_pos = geo.mirror_positions(enemy_pos, W, H)

    ally_ids = np.where(ally_mask)[0]
    enemy_ids = np.where(enemy_mask)[0]

    # Belegungskarte eigener Units (fuer Stacking-Features)
    ally_count = np.zeros((W, H), dtype=np.float32)
    first_on_tile: dict[tuple[int, int], int] = {}
    for uid in ally_ids:
        x, y = int(ally_pos[uid][0]), int(ally_pos[uid][1])
        if 0 <= x < W and 0 <= y < H:
            ally_count[x, y] += 1.0
            first_on_tile.setdefault((x, y), int(uid))

    enemy_coords = np.array(
        [[int(enemy_pos[i][0]), int(enemy_pos[i][1])] for i in enemy_ids
         if enemy_pos[i][0] >= 0], dtype=np.int32
    ).reshape(-1, 2)
    enemy_energies = np.array(
        [float(enemy_ene[i]) for i in enemy_ids if enemy_pos[i][0] >= 0],
        dtype=np.float32,
    )

    points = np.asarray(obs["team_points"], dtype=np.float32)
    wins = np.asarray(obs["team_wins"], dtype=np.float32)
    match_step = int(np.asarray(obs["match_steps"]))
    max_match_steps = float(cfg.get("max_steps_in_match", 100))

    assigned = assign_targets(
        belief, ally_ids,
        ally_pos[ally_ids] if len(ally_ids) else np.zeros((0, 2), np.int32),
        W, H, enemy_half_penalty=600.0 if agent.cfg.home_first else 0.0,
        seen_before_penalty=300.0 if agent.cfg.explore_memory else 0.0,
        grid=4 if agent.cfg.explore_grid else 0, stale=int(agent.cfg.explore_stale),
    )

    return dict(
        belief=belief, W=W, H=H,
        ally_ids=ally_ids, ally_pos=ally_pos, ally_ene=ally_ene,
        enemy_coords=enemy_coords, enemy_energies=enemy_energies,
        ally_count=ally_count, first_on_tile=first_on_tile,
        point_coords=belief.point_coords(),
        cand_coords=belief.candidate_coords(),
        relic_coords=belief.relic_coords(),
        unexp_coords=belief.unexplored_coords(),
        assigned=assigned,
        passable=belief.passable, nebula=belief.nebula,
        energy_map=belief.energy_map,
        point_tiles=belief.point_tiles, candidate=belief.candidate,
        explored=belief.explored,
        own_points=float(points[team]), opp_points=float(points[opp]),
        own_wins=float(wins[team]), opp_wins=float(wins[opp]),
        match_step=match_step, max_match_steps=max_match_steps,
        match_idx=belief.match_idx,
        move_cost=float(cfg.get("unit_move_cost", 3)),
        sap_cost=float(cfg.get("unit_sap_cost", 40)),
        sap_range=int(cfg.get("unit_sap_range", 5)),
        sensor_range=float(cfg.get("unit_sensor_range", 2)),
        max_units=int(cfg.get("max_units", 16)),
    )


def build_features(ctx: dict, uid: int) -> np.ndarray:
    """Feature-Vektor fuer eine Unit (Nullvektor, falls Unit nicht existiert)."""
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    W, H = ctx["W"], ctx["H"]
    if uid not in set(ctx["ally_ids"].tolist()):
        return f
    x = int(ctx["ally_pos"][uid][0])
    y = int(ctx["ally_pos"][uid][1])
    if not (0 <= x < W and 0 <= y < H):
        return f
    e = float(ctx["ally_ene"][uid])
    WH = float(W + H)
    i = 0

    # --- eigene Lage + Parameter
    f[i] = x / W; i += 1
    f[i] = y / H; i += 1
    f[i] = min(max(e, 0.0) / MAX_UNIT_ENERGY, 1.0); i += 1
    f[i] = 1.0 if e < ctx["move_cost"] else 0.0; i += 1
    f[i] = ctx["match_step"] / max(1.0, ctx["max_match_steps"]); i += 1
    f[i] = ctx["match_idx"] / N_MATCHES; i += 1
    f[i] = ctx["move_cost"] / 5.0; i += 1
    f[i] = ctx["sap_cost"] / 50.0; i += 1
    f[i] = ctx["sap_range"] / 7.0; i += 1
    f[i] = ctx["sensor_range"] / 4.0; i += 1

    # --- Punktestand
    f[i] = min(ctx["own_points"] / 200.0, 1.0); i += 1
    f[i] = float(np.tanh((ctx["own_points"] - ctx["opp_points"]) / 20.0)); i += 1
    f[i] = ctx["own_wins"] / N_MATCHES; i += 1
    f[i] = ctx["opp_wins"] / N_MATCHES; i += 1

    # --- naechste Ziele je Klasse
    for coords in (ctx["point_coords"], ctx["cand_coords"],
                   ctx["relic_coords"], ctx["unexp_coords"]):
        dx, dy, dist, ex = _nearest(coords, x, y, W, H)
        f[i] = dx; f[i + 1] = dy; f[i + 2] = dist; f[i + 3] = ex
        i += 4
    tgt = ctx["assigned"].get(int(uid))
    if tgt is None:
        f[i] = 0.0; f[i + 1] = 0.0; f[i + 2] = 1.0; f[i + 3] = 0.0
    else:
        f[i] = (tgt[0] - x) / W
        f[i + 1] = (tgt[1] - y) / H
        f[i + 2] = (abs(tgt[0] - x) + abs(tgt[1] - y)) / WH
        f[i + 3] = 1.0
    i += 4

    # --- aktuelles Tile
    f[i] = float(ctx["point_tiles"][x, y]); i += 1
    f[i] = float(ctx["candidate"][x, y]); i += 1
    f[i] = float(ctx["nebula"][x, y]); i += 1
    f[i] = min(ctx["ally_count"][x, y] / 4.0, 1.0); i += 1
    owner = ctx["first_on_tile"].get((x, y), int(uid))
    f[i] = 1.0 if owner != int(uid) else 0.0; i += 1

    # --- lokale Umgebung in 5 Richtungen
    for d in range(5):
        tx, ty = x + int(geo.DIR_DX[d]), y + int(geo.DIR_DY[d])
        if 0 <= tx < W and 0 <= ty < H:
            f[i] = float(ctx["passable"][tx, ty])
            f[i + 1] = float(np.clip(ctx["energy_map"][tx, ty] / MAX_TILE_ENERGY, -1, 1))
            f[i + 2] = float(ctx["point_tiles"][tx, ty])
            f[i + 3] = float(ctx["candidate"][tx, ty])
            f[i + 4] = float(not ctx["explored"][tx, ty])
            f[i + 5] = min(ctx["ally_count"][tx, ty] / 4.0, 1.0)
        # ausserhalb der Karte -> alles 0 (inkl. passable=0)
        i += 6

    # --- Gegner
    ec = ctx["enemy_coords"]
    if len(ec):
        d = np.abs(ec[:, 0] - x) + np.abs(ec[:, 1] - y)
        k = int(np.argmin(d))
        f[i] = (ec[k, 0] - x) / W
        f[i + 1] = (ec[k, 1] - y) / H
        f[i + 2] = float(d[k]) / WH
        f[i + 3] = float(np.tanh((e - ctx["enemy_energies"][k]) / 100.0))
        R = ctx["sap_range"]
        in_range = (np.abs(ec[:, 0] - x) <= R) & (np.abs(ec[:, 1] - y) <= R)
        f[i + 4] = min(float(in_range.sum()) / 4.0, 1.0)
        f[i + 5] = min(len(ec) / float(ctx["max_units"]), 1.0)
    else:
        f[i + 2] = 1.0
    i += 6

    # --- globale Zaehler
    f[i] = len(ctx["ally_ids"]) / float(ctx["max_units"]); i += 1
    f[i] = min(len(ctx["point_coords"]) / 20.0, 1.0); i += 1
    f[i] = float(ctx["explored"].mean()); i += 1
    f[i] = min(len(ctx["cand_coords"]) / 50.0, 1.0); i += 1

    assert i == FEATURE_DIM, f"Feature-Layout inkonsistent: {i} != {FEATURE_DIM}"
    return f

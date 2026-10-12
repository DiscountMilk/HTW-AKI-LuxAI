"""
Was das RL SIEHT (Input), was es TUN kann (Output) und was es NICHT darf (Verbote).

INPUT : encode(unit, memory)      -> float32[OBS_DIM]   eine Zahl pro Funktion in FEATURES
OUTPUT: ACTIONS[i]                -> [aktion, dx, dy]    i kommt vom Netz (learner.pick)
REGELN: action_mask(unit, memory) -> bool[N_ACTIONS]     False = Aktion verboten, Netz kann sie nicht wählen

Erweitern - gleiches Muster wie safety_rules.py: Funktion schreiben, in die Liste eintragen.
  Neues Feature: def _f_name(unit, memory) -> float (etwa -1..1)               -> FEATURES
  Neue Aktion  : [aktion, dx, dy]                                             -> ACTIONS
  Neues Verbot : def _v_name(unit, memory, action) -> bool (True = verboten)  -> FORBIDDEN

ACHTUNG: FEATURES oder ACTIONS ändern = andere Netzgröße. Die alte .pth passt dann nicht mehr -> löschen.
"""
import numpy as np

MAX_UNIT_ENERGY = 400   # fester Spielwert, steht nicht in env_cfg
MAX_TILE_ENERGY = 20


# ----------------------------------------------------------------------
# 1 INPUT - jede Funktion liefert GENAU EINE Zahl. Reihenfolge = Position im Vektor.
# ----------------------------------------------------------------------
def _nearest_offset(unit, memory, targets):
    """[dx, dy] zum nächsten Ziel, geteilt durch Kartenbreite. Kein Ziel -> [0, 0]."""
    if len(targets) == 0:
        return np.zeros(2)
    offsets = np.array(targets) - unit.pos
    return offsets[np.abs(offsets).sum(axis=1).argmin()] / memory.env_cfg["map_width"]


def _f_energy(unit, memory):
    """Eigene Energie, 0..1."""
    return unit.energy / MAX_UNIT_ENERGY


def _f_match_progress(unit, memory):
    """Wie weit das Match ist, 0..1."""
    return memory.match_step / memory.env_cfg["max_steps_in_match"]


def _f_tile_energy(unit, memory):
    """Energie des eigenen Felds, -1..1."""
    x, y = unit.pos
    return memory.energy[x, y] / MAX_TILE_ENERGY


def _f_relic_known(unit, memory):
    """1, wenn mindestens eine Relic bekannt ist."""
    return float(len(memory.relics) > 0)


def _f_relic_dx(unit, memory):
    return _nearest_offset(unit, memory, memory.relics)[0]


def _f_relic_dy(unit, memory):
    return _nearest_offset(unit, memory, memory.relics)[1]


def _f_on_point_tile(unit, memory):
    """1, wenn die Einheit auf einem sicheren Punktefeld steht."""
    x, y = unit.pos
    return float(memory.point_tile[x, y] == 1)


def _f_point_tile_known(unit, memory):
    return float(len(memory.known_point_tiles()) > 0)


def _f_point_tile_dx(unit, memory):
    return _nearest_offset(unit, memory, memory.known_point_tiles())[0]


def _f_point_tile_dy(unit, memory):
    return _nearest_offset(unit, memory, memory.known_point_tiles())[1]


def _f_enemy_near(unit, memory):
    """1, wenn kürzlich ein Gegner in Sap-Reichweite gesehen wurde."""
    enemies = memory.recent_enemies()
    return float(len(enemies) > 0 and np.abs(enemies - unit.pos).max(axis=1).min() <= memory.env_cfg["unit_sap_range"])


FEATURES = [
    _f_energy,
    _f_match_progress,
    _f_tile_energy,
    _f_relic_known,
    _f_relic_dx,
    _f_relic_dy,
    _f_on_point_tile,
    _f_point_tile_known,
    _f_point_tile_dx,
    _f_point_tile_dy,
    _f_enemy_near,
]
OBS_DIM = len(FEATURES)


def encode(unit, memory):
    return np.array([f(unit, memory) for f in FEATURES], dtype=np.float32)


# ----------------------------------------------------------------------
# 2 OUTPUT - das Netz gibt einen Index i aus, der Agent führt ACTIONS[i] aus.
# ----------------------------------------------------------------------
ACTIONS = [
    [0, 0, 0],   # 0 stehen
    [1, 0, 0],   # 1 hoch
    [2, 0, 0],   # 2 rechts
    [3, 0, 0],   # 3 runter
    [4, 0, 0],   # 4 links
]                # Sap macht vorerst safety_rules.py
N_ACTIONS = len(ACTIONS)

MOVE = {0: (0, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}   # aktion -> Feldänderung (dx, dy)


# ----------------------------------------------------------------------
# 3 VERBOTE - Regeln, die das RL einschränken. Anders als safety_rules entscheiden sie
#   nicht selbst, sondern streichen nur Aktionen. Das RL wählt unter dem Rest und lernt weiter.
# ----------------------------------------------------------------------
def _v_blocked(unit, memory, action):
    """Nicht aus der Karte oder in einen Asteroiden laufen."""
    dx, dy = MOVE.get(action[0], (0, 0))
    return memory.is_blocked(unit.pos[0] + dx, unit.pos[1] + dy)


FORBIDDEN = [
    _v_blocked,
]


def action_mask(unit, memory):
    mask = np.array([not any(rule(unit, memory, a) for rule in FORBIDDEN) for a in ACTIONS])
    mask[0] = True   # Stehen ist immer erlaubt, damit es immer mindestens eine Aktion gibt
    return mask


if __name__ == "__main__":
    # .venv/bin/python -m agents.team_eric_test.rl.spaces
    from agents.team_eric_test.memory import ASTEROID, GameMemory, Unit

    m = GameMemory()
    m.reset({"max_units": 16, "map_width": 24, "map_height": 24, "max_steps_in_match": 100, "unit_sap_range": 6})
    m.relics = [np.array([5, 5])]
    m.tile_type[1, 0] = ASTEROID
    u = Unit(id=0, pos=np.array([0, 0]), energy=100)

    obs = encode(u, m)
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    assert obs[FEATURES.index(_f_relic_dx)] == 5 / 24
    # Ecke oben links, Asteroid rechts: nur stehen und runter erlaubt
    assert action_mask(u, m).tolist() == [True, False, False, True, False]
    print("spaces.py OK")

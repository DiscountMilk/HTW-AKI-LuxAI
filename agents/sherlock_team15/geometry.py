"""Seitenunabhaengige Koordinaten ("canonical space").

Die Lux-S3-Karte ist spiegelsymmetrisch zur Antidiagonale:

    M(x, y) = (W - 1 - y, H - 1 - x)

Nachgewiesen fuer tile_type und map_features.energy (siehe docs/presentation/notes.md).
player_0 spawnt auf (0, 0), player_1 auf (W-1, H-1) = M(0, 0).

Deshalb rechnen wir alle Features von player_1 in den kanonischen Raum um, in dem
der eigene Spawn immer auf (0, 0) liegt. Dadurch braucht nur EIN Netz trainiert zu
werden (statt getrennter Gewichte je Seite) und jedes Spiel liefert Erfahrung fuer
beide Seiten.

Aktionsrichtungen: 0 stay, 1 up (y-1), 2 right (x+1), 3 down (y+1), 4 left (x-1).
Unter M wird aus "up" ein "right" usw. -> MIRROR_DIR ist eine Involution.
"""
from __future__ import annotations

import numpy as np

STAY, UP, RIGHT, DOWN, LEFT, SAP = 0, 1, 2, 3, 4, 5

# dx, dy je Bewegungsrichtung (Index = Aktions-ID 0..4)
DIR_DX = np.array([0, 0, 1, 0, -1], dtype=np.int32)
DIR_DY = np.array([0, -1, 0, 1, 0], dtype=np.int32)

# Richtungsabbildung unter der Antidiagonal-Spiegelung (0<->0, 1<->2, 3<->4)
MIRROR_DIR = np.array([0, 2, 1, 4, 3], dtype=np.int32)


def mirror_xy(x, y, W: int = 24, H: int = 24):
    """Einzelne Koordinate spiegeln (funktioniert auch mit numpy-Arrays)."""
    return (W - 1 - y, H - 1 - x)


def mirror_positions(pos: np.ndarray, W: int = 24, H: int = 24) -> np.ndarray:
    """Positionsarray (..., 2) spiegeln. Unsichtbare Eintraege (-1) bleiben -1."""
    out = np.asarray(pos).copy()
    valid = (out[..., 0] >= 0) & (out[..., 1] >= 0)
    x = out[..., 0]
    y = out[..., 1]
    nx = np.where(valid, W - 1 - y, -1)
    ny = np.where(valid, H - 1 - x, -1)
    out[..., 0] = nx
    out[..., 1] = ny
    return out


def mirror_map(arr: np.ndarray) -> np.ndarray:
    """Kartenarray [x][y] an der Antidiagonale spiegeln: out[x][y] = arr[W-1-y][H-1-x]."""
    return np.ascontiguousarray(np.asarray(arr)[::-1, ::-1].T)


def mirror_dir(direction: int) -> int:
    """Bewegungsrichtung zwischen echtem und kanonischem Raum uebersetzen."""
    return int(MIRROR_DIR[int(direction)])


def mirror_delta(dx: int, dy: int):
    """Sap-Delta (dx, dy) spiegeln. Aus M folgt (dx, dy) -> (-dy, -dx)."""
    return (-int(dy), -int(dx))

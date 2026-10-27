"""Gedaechtnis ueber das ganze Spiel (3 Matches): Relics, Punkte-Tiles, Karte.

Alles wird im KANONISCHEN Raum gespeichert (siehe geometry.py), d. h. aus Sicht
eines Spielers, der auf (0, 0) spawnt. Damit sind Belief und Features
seitenunabhaengig und ein Gewichtssatz gilt fuer beide Seiten.

Kernidee "Punkte-Tiles finden":
Die punktebringenden Tiles um einen Relic-Node sind unsichtbar (5x5-Maske,
Slide 38) und muessen durch Probieren erschlossen werden. Pro Step gilt
  gain      = team_points[t] - team_points[t-1]
  besetzt   = eindeutige Tiles mit mindestens einer eigenen Unit mit energy >= 0
              (die Engine zaehlt Units mit negativer Energie nicht mit,
               env.py: exclude_negative_energy_units=True)
  bekannt   = Anzahl besetzter, bereits bewiesener Punkte-Tiles
  U         = besetzt - bewiesene Punkte-Tiles - bewiesene Null-Tiles
  rest      = gain - bekannt   = Anzahl punktender Tiles in U
Daraus:
  rest == 0         -> alle Tiles in U bringen KEINE Punkte
  rest == |U|       -> alle Tiles in U SIND Punkte-Tiles
  sonst             -> unentschieden: als Constraint merken und spaeter erneut
                       aufloesen (Constraint-Propagation).
Die Constraints laufen bewusst ueber ALLE besetzten Tiles, nicht nur ueber die
Kandidaten in Relic-Naehe. Dadurch ist der Schluss auch dann korrekt, wenn ein
Relic noch nicht entdeckt wurde (sonst waere `rest` zu gross und es entstuenden
falsche Punkte-Tiles).

WICHTIG - wann darf ueberhaupt gefolgert werden:
Die Punkte-Schwelle der Engine haengt davon ab, wie viele Relic-Paare schon
gespawnt sind, und ein Paar spawnt in Match j bei match_step 0..49. In dieser
Phase kann dasselbe Tile erst punktlos und dann punktend sein - jeder Schluss
daraus ist potenziell falsch (im Experiment: 27 falsche Punkte-Tiles von 108).
Deshalb werden Constraints erst ab match_step >= 50 gesammelt.

WICHTIG - "dead" ist nicht dauerhaft gueltig:
Die Engine vergibt Punkte nur fuer Tiles, deren Relic schon gespawnt ist
(luxai_s3/env.py):

    scores = (unit_counts > 0) & (relic_nodes_map_weights <= relic_nodes_mask.sum()//2)
             & (relic_nodes_map_weights > 0)

Relic-Paare spawnen erst im Laufe der ersten k Matches. Ein Tile kann also in
Match 0 punktlos sein und ab Match 1 punkten (z. B. wenn sich die 5x5-Masken
zweier Relics ueberlappen und das Tile zum spaeteren Relic gehoert). Deshalb:
  * bewiesene Punkte-Tiles sind FUER IMMER gueltig (die Schwelle waechst nur),
  * "dead"-Tiles und die Constraint-Historie werden bei jedem Matchwechsel und
    bei jedem Widerspruch verworfen (_reopen_dead).
Ohne diese Korrektur entstehen falsche Punkte-Tiles (im Experiment 3 von 44).

SYMMETRIE (11.10.2026, Schalter `symmetry`): Relics spawnen paarweise
spiegelsymmetrisch zur Antidiagonale M(x, y) = (W-1-y, H-1-x), und die
punktebringenden Tiles sind zu JEDEM Zeitpunkt symmetrisch (geprueft ueber
1818 Steps auf 6 Seeds, 0 Verstoesse). Deshalb gilt jeder Schluss auch fuer das
Spiegeltile: gesehenes Relic -> Spiegel-Relic bekannt, bewiesenes Punkte-/
Null-Tile -> Spiegeltile ebenso, erkundetes Gebiet -> Spiegelgebiet erkundet.
Die Analyse vom 11.10. (scripts/analyze_losses.py) zeigte: wir kennen am
Matchende nur ~5 Punkte-Tiles, Solution_92 besetzt ~6.

Implementierungsdetail: die Propagation laeuft auf flachen int-Indizes und
Python-Sets, nicht auf numpy-Skalarindizierung - das ist ca. 50x schneller und
war im Profiling der Flaschenhals.
"""
from __future__ import annotations

import numpy as np

# tile_type-Werte der Engine
TILE_UNKNOWN = -1
TILE_EMPTY = 0
TILE_NEBULA = 1
TILE_ASTEROID = 2

RELIC_REWARD_RANGE = 2  # 5x5-Maske = Chebyshev-Radius 2 um den Relic-Node
MAX_RECORDS = 400       # Obergrenze fuer die Constraint-Historie
MAX_MATCH_STEPS = 100

# Relic-Paare spawnen laut luxai_s3/state.py:348
#   schedule_j = randint(0, max_steps_in_match // 2) + j * (max_steps_in_match + 1)
# also in Match j bei match_step in [0, 49] - und beide Kartenhaelften gleichzeitig.
# Ab diesem match_step aendert sich die Punkte-Schwelle innerhalb des Matches
# nicht mehr, nur dann sind Schluesse ueber Punkte-Tiles verlaesslich.
STABLE_FROM_MATCH_STEP = 50
NEVER_SEEN = -1000


class Belief:
    """Persistentes Weltmodell eines Agenten fuer ein komplettes Spiel."""

    def __init__(self, team_id: int, W: int = 24, H: int = 24, canonical: bool = True,
                 symmetry: bool = False, early_inference: bool = False):
        self.W, self.H = W, H
        self.canonical = bool(canonical)
        self.symmetry = bool(symmetry)
        self.early_inference = bool(early_inference)
        self.set_team(team_id)
        self.reset_game()

    # ------------------------------------------------------------------ setup
    def set_team(self, team_id: int) -> None:
        self.team_id = int(team_id)
        self.opp_team_id = 1 - self.team_id
        # player_1 sieht die Welt gespiegelt -> in kanonischen Raum drehen
        # (Ablation no_canonical: nie spiegeln, dafuer getrennte Netze je Seite)
        self.mirrored = self.canonical and self.team_id == 1

    def reset_game(self) -> None:
        W, H = self.W, self.H
        self.relic_positions: set[tuple[int, int]] = set()
        self.relic_zone = np.zeros((W, H), dtype=bool)   # 5x5 um bekannte Relics
        self.candidate = np.zeros((W, H), dtype=bool)    # relic_zone, noch unbekannt
        self.point_tiles = np.zeros((W, H), dtype=bool)  # bewiesen: bringt Punkte
        self.dead_tiles = np.zeros((W, H), dtype=bool)   # bewiesen: bringt nichts
        self.explored = np.zeros((W, H), dtype=bool)
        # wie explored, aber NICHT pro Match zurueckgesetzt (Zielwahl explore_memory)
        self.explored_ever = np.zeros((W, H), dtype=bool)
        # match_step, an dem ein Tile (oder sein Spiegeltile) zuletzt gesehen wurde
        self.last_seen = np.full((W, H), NEVER_SEEN, dtype=np.int16)
        self.match_step = 0
        self.tile_type = np.full((W, H), TILE_UNKNOWN, dtype=np.int8)
        self.energy_map = np.zeros((W, H), dtype=np.float32)

        # schnelle Spiegelbilder der Statusmengen (flache Indizes x*H + y)
        self._point: set[int] = set()
        self._dead: set[int] = set()
        self._cand_dirty = True

        self.records: list[tuple[tuple[int, ...], int]] = []
        self.last_points = 0
        self.last_opp_points = 0
        self.match_idx = 0
        self.newly_proven = 0        # in diesem Step neu bewiesene Punkte-Tiles
        self.newly_explored = 0      # in diesem Step neu gesehene Tiles
        self.contradictions = 0      # erkannte Widersprueche (Diagnose)
        self._last_step = -1
        self._coords_dirty = True
        self._cache: dict[str, np.ndarray] = {}

    def reset_match(self) -> None:
        """Matchwechsel: Punktestand faellt auf 0, Relics koennen neu erscheinen."""
        self.last_points = 0
        self.last_opp_points = 0
        # Relics erscheinen in den ersten k Matches noch nachtraeglich (Slide 38)
        # -> Erkundungswissen pro Match zuruecksetzen, Relic-/Punktewissen behalten.
        self.explored[:] = False
        self.last_seen[:] = NEVER_SEEN
        self._coords_dirty = True
        # Neues Match => moeglicherweise neues Relic-Paar => Punkte-Regel aendert sich
        self._reopen_dead()

    # ------------------------------------------------------------ observation
    def observe(self, obs) -> None:
        """Beobachtung einarbeiten. Idempotent pro Step - act() und update()
        duerfen beide aufrufen."""
        step = int(np.asarray(obs["steps"]))
        if step == self._last_step:
            return
        self._last_step = step
        self.newly_proven = 0
        self.newly_explored = 0
        self.match_idx = step // (MAX_MATCH_STEPS + 1)

        match_step = int(np.asarray(obs["match_steps"]))
        self.match_step = match_step
        if match_step == 0:
            self.reset_match()
            return  # zu Matchbeginn existieren noch keine Units

        from . import geometry as geo  # lokal: vermeidet Zyklus

        sensor = np.asarray(obs["sensor_mask"], dtype=bool)
        tile_type = np.asarray(obs["map_features"]["tile_type"], dtype=np.int32)
        energy = np.asarray(obs["map_features"]["energy"], dtype=np.float32)
        relic_pos = np.asarray(obs["relic_nodes"], dtype=np.int32)
        relic_mask = np.asarray(obs["relic_nodes_mask"], dtype=bool)
        ally_pos = np.asarray(obs["units"]["position"], dtype=np.int32)[self.team_id]
        ally_mask = np.asarray(obs["units_mask"], dtype=bool)[self.team_id]
        ally_energy = np.asarray(obs["units"]["energy"], dtype=np.int32)[self.team_id]
        if ally_energy.ndim == 2 and ally_energy.shape[-1] == 1:
            ally_energy = ally_energy[:, 0]

        if self.mirrored:
            sensor = geo.mirror_map(sensor)
            tile_type = geo.mirror_map(tile_type)
            energy = geo.mirror_map(energy)
            relic_pos = geo.mirror_positions(relic_pos, self.W, self.H)
            ally_pos = geo.mirror_positions(ally_pos, self.W, self.H)

        self._update_map(sensor, tile_type, energy)
        self._update_relics(relic_pos, relic_mask)

        points = int(np.asarray(obs["team_points"])[self.team_id])
        gain = points - self.last_points
        self.last_points = points
        self.last_opp_points = int(np.asarray(obs["team_points"])[self.opp_team_id])
        self._infer_point_tiles(ally_pos, ally_energy, ally_mask, gain, match_step)
        self._refresh_candidates()

    def _update_map(self, sensor, tile_type, energy) -> None:
        explored_now = sensor
        if self.symmetry:
            from . import geometry as geo
            explored_now = sensor | geo.mirror_map(sensor)  # nur fuer "erkundet"
        new = int(np.count_nonzero(explored_now & ~self.explored))
        self.newly_explored = new
        if new:
            self.explored |= explored_now
            self._coords_dirty = True
        self.explored_ever |= explored_now
        self.last_seen[explored_now] = self.match_step
        # Asteroiden/Nebel driften (Slides 30/31) -> sichtbare Tiles ueberschreiben
        np.copyto(self.tile_type, tile_type.astype(np.int8), where=sensor)
        np.copyto(self.energy_map, energy, where=sensor)

    def _update_relics(self, relic_pos, relic_mask) -> None:
        for i in np.nonzero(relic_mask)[0]:
            x, y = int(relic_pos[i][0]), int(relic_pos[i][1])
            if x < 0 or y < 0:
                continue
            self._add_relic(x, y)
            if self.symmetry:
                self._add_relic(self.W - 1 - y, self.H - 1 - x)

    def _add_relic(self, x: int, y: int) -> None:
        if (x, y) in self.relic_positions:
            return
        self.relic_positions.add((x, y))
        r = RELIC_REWARD_RANGE
        self.relic_zone[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1] = True
        self._cand_dirty = True
        self._coords_dirty = True

    def _refresh_candidates(self) -> None:
        """candidate ist abgeleitet: Relic-Umfeld, das noch nicht entschieden ist."""
        if not self._cand_dirty:
            return
        np.logical_and(self.relic_zone, ~(self.point_tiles | self.dead_tiles),
                       out=self.candidate)
        self._cand_dirty = False

    # ----------------------------------------------------- punkte-tile-logik
    def known_pairs(self) -> int:
        """Anzahl bekannter Relic-Paare (Relic und Spiegelrelic spawnen gemeinsam)."""
        return len({min((x, y), (self.W - 1 - y, self.H - 1 - x))
                    for x, y in self.relic_positions})

    def pair_spawned(self) -> bool:
        """Ist das Relic-Paar des aktuellen Matches sicher schon gespawnt?

        Paar j spawnt in Match j (falls j < k) bei match_step 0..49, hoechstens
        eines pro Match. Kennen wir j+1 Paare, sind 0..j alle da. In Match 0 gibt
        es vor dem ersten Paar gar keine Relics: jeder Punkt (auch des Gegners)
        beweist den Spawn.
        """
        if self.known_pairs() >= self.match_idx + 1:
            return True
        return self.match_idx == 0 and (self.last_points > 0 or self.last_opp_points > 0)

    def _stable(self, match_step: int) -> bool:
        """Darf aus dem Punktgewinn geschlossen werden? Nach dem (einzigen) Spawn
        des Matches aendert sich die Punkte-Schwelle bis Matchende nicht mehr."""
        if match_step >= STABLE_FROM_MATCH_STEP:
            return True
        return self.early_inference and self.pair_spawned()

    def _infer_point_tiles(self, ally_pos, ally_energy, ally_mask, gain: int,
                           match_step: int) -> None:
        if not self._stable(match_step):
            return  # Relic-Spawn-Fenster: Punkte-Schwelle noch instabil
        H = self.H
        occupied = set()
        for i in np.nonzero(ally_mask)[0]:
            x, y = int(ally_pos[i][0]), int(ally_pos[i][1])
            # Units mit negativer Energie zaehlen fuer die Punkte nicht mit
            if x >= 0 and y >= 0 and ally_energy[i] >= 0:
                occupied.add(x * H + y)
        if not occupied:
            return

        known = len(occupied & self._point)
        unknown = occupied - self._point - self._dead
        rest = gain - known
        if rest < 0 or rest > len(unknown):
            # Widerspruch: der Punktgewinn ist mit dem aktuellen Wissen nicht
            # erklaerbar -> ein weiteres Relic-Paar ist aktiv geworden.
            # Null-Wissen und Constraints verwerfen, Punkte-Tiles behalten.
            self.contradictions += 1
            self._reopen_dead()
            known = len(occupied & self._point)
            unknown = occupied - self._point - self._dead
            rest = gain - known
            if rest < 0 or rest > len(unknown):
                return
        if not unknown:
            return  # keine neue Information
        self.records.append((tuple(sorted(unknown)), rest))
        if len(self.records) > MAX_RECORDS:
            del self.records[: len(self.records) - MAX_RECORDS]
        self._propagate()

    def _reopen_dead(self) -> None:
        """Alle "bringt keine Punkte"-Schluesse verwerfen (siehe Modulkopf).

        Bewiesene Punkte-Tiles bleiben erhalten: die Engine-Schwelle
        relic_nodes_mask.sum()//2 waechst monoton, ein Tile das einmal punktet
        punktet immer weiter.
        """
        if not self._dead and not self.records:
            return
        self._dead.clear()
        self.records.clear()
        self.dead_tiles[:] = False
        self._cand_dirty = True
        self._coords_dirty = True

    def _propagate(self) -> None:
        """Constraint-Propagation ueber alle gesammelten Beobachtungen."""
        point, dead = self._point, self._dead
        changed = True
        while changed:
            changed = False
            still_open: list[tuple[tuple[int, ...], int]] = []
            for tiles, rest in self.records:
                open_tiles = [t for t in tiles if t not in point and t not in dead]
                if not open_tiles:
                    continue  # vollstaendig aufgeloest
                rest_now = rest - sum(1 for t in tiles if t in point)
                if rest_now <= 0:
                    for t in open_tiles:
                        self._mark_dead(t)
                    changed = True
                elif rest_now >= len(open_tiles):
                    for t in open_tiles:
                        self._mark_point(t)
                    changed = True
                else:
                    still_open.append((tuple(open_tiles), rest_now))
            self.records = still_open

    def _mirror_idx(self, f: int) -> int:
        x, y = divmod(f, self.H)
        return (self.W - 1 - y) * self.H + (self.H - 1 - x)

    def _mark_point(self, f: int) -> None:
        self._dead.discard(f)
        if f in self._point:
            return
        if self.symmetry:
            m = self._mirror_idx(f)
            if m != f:
                self._point.add(f)          # Rekursionsschutz
                self._mark_point(m)
                self._point.discard(f)
        self._point.add(f)
        x, y = divmod(f, self.H)
        self.point_tiles[x, y] = True
        self.dead_tiles[x, y] = False
        self.newly_proven += 1
        self._cand_dirty = True
        self._coords_dirty = True

    def _mark_dead(self, f: int) -> None:
        if f in self._dead or f in self._point:
            return
        self._dead.add(f)
        if self.symmetry:
            self._mark_dead(self._mirror_idx(f))
        x, y = divmod(f, self.H)
        self.dead_tiles[x, y] = True
        self._cand_dirty = True
        self._coords_dirty = True

    # ----------------------------------------------------------- abfragen
    @property
    def passable(self) -> np.ndarray:
        """True, wo (nach letztem Wissen) kein Asteroid liegt."""
        return self.tile_type != TILE_ASTEROID

    @property
    def nebula(self) -> np.ndarray:
        return self.tile_type == TILE_NEBULA

    def _coords(self, key: str, mask: np.ndarray) -> np.ndarray:
        if self._coords_dirty:
            self._cache.clear()
            self._coords_dirty = False
        hit = self._cache.get(key)
        if hit is None:
            hit = np.argwhere(mask).astype(np.int32)
            self._cache[key] = hit
        return hit

    def point_coords(self) -> np.ndarray:
        return self._coords("point", self.point_tiles)

    def candidate_coords(self) -> np.ndarray:
        self._refresh_candidates()
        return self._coords("cand", self.candidate)

    def unexplored_coords(self) -> np.ndarray:
        return self._coords("unexp", ~self.explored)

    def explore_coords(self, stale: int = 0) -> np.ndarray:
        """Erkundungsziele. stale > 0: in Match 0, solange kein Relic bekannt ist,
        zaehlen nur Tiles als erkundet, die in den letzten `stale` Steps oder ab
        match_step 49 gesehen wurden (vorher gesehene koennen ein spaeter
        gespawntes Relic verbergen)."""
        if stale <= 0 or self.match_idx != 0 or self.known_pairs() > 0:
            return self.unexplored_coords()
        ls = self.last_seen
        fresh = (ls >= self.match_step - stale) | (ls >= STABLE_FROM_MATCH_STEP - 1)
        return np.argwhere(~fresh).astype(np.int32)

    def relic_coords(self) -> np.ndarray:
        if not self.relic_positions:
            return np.zeros((0, 2), dtype=np.int32)
        key = "relic"
        if self._coords_dirty:
            self._cache.clear()
            self._coords_dirty = False
        hit = self._cache.get(key)
        if hit is None or len(hit) != len(self.relic_positions):
            hit = np.array(sorted(self.relic_positions), dtype=np.int32)
            self._cache[key] = hit
        return hit


class MemorylessBelief(Belief):
    """Ablation `no_belief`: dieselbe Schnittstelle, aber ohne Gedaechtnis.

    Vor jeder Beobachtung wird alles verworfen, d. h. Relics, Karte und
    "erkundet" spiegeln nur das aktuell Sichtbare wider. Punkte-Tiles koennen
    ohne Gedaechtnis (Punktverlauf ueber Steps) nicht abgeleitet werden und
    bleiben leer.
    """

    def observe(self, obs) -> None:
        step = int(np.asarray(obs["steps"]))
        if step == self._last_step:
            return
        self.reset_game()
        super().observe(obs)

    def _infer_point_tiles(self, *args, **kwargs) -> None:
        return

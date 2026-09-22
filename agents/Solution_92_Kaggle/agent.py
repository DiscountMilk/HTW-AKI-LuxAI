import copy
import numpy as np
from sys import stderr
from scipy.signal import convolve2d

from agents.Solution_92_Kaggle.base import (
    Global,
    NodeType,
    ActionType,
    SPACE_SIZE,
    get_match_step,
    warp_point,
    get_opposite,
    is_team_sector,
    get_match_number,
)
# from debug import show_map, show_energy_field, show_exploration_map
from agents.Solution_92_Kaggle.pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = True

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: None | bool):
        if self._explored_for_relic and self._relic and not status:
            raise ValueError(
                f"Can't change the relic status {self._relic}->{status} for {self}"
                ", the tile has already been explored"
            )

        if status is None:
            self._explored_for_relic = False
            return

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: None | bool):
        if self._explored_for_reward and self._reward and not status:
            raise ValueError(
                f"Can't change the reward status {self._reward}->{status} for {self}"
                ", the tile has already been explored"
            )

        if status is None:
            self._explored_for_reward = False
            return

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(step, obs, team_id, team_reward)

    def _update_relic_map(self, step, obs, team_id, team_reward):
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                # We have found a new relic.
                self._update_relic_status(*xy, status=True)
                for x, y in nearby_positions(*xy, Global.RELIC_REWARD_RANGE):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found

        match = get_match_number(step)
        match_step = get_match_step(step)
        num_relics_th = 2 * min(match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1

        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) >= num_relics_th:
                # all relics found, mark all nodes as explored for relics
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)

        if not Global.ALL_REWARDS_FOUND:
            if (
                    match_step > Global.LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR
                    or len(self._relic_nodes) >= num_relics_th
            ):
                self._update_reward_status_from_relics_distribution()
                self._update_reward_results(obs, team_id, team_reward)
                self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        """Infer reward-tile truth values from aggregated point deltas.

        Background / idea
        -----------------
        The environment only tells us the *total* team reward gained in the last step
        (here passed in as `team_reward` and stored in `Global.REWARD_RESULTS`).

        However, we do not directly observe which exact tiles yielded those points.
        What we *can* track is:
          - the set of tiles currently occupied by our ships at the moment points were gained
          - any tiles we already proved to be reward tiles (`node.reward == True`)
          - any tiles we already proved to be non-reward tiles (`explored_for_reward and not reward`)

        Each entry in `Global.REWARD_RESULTS` is of the form:
            {"nodes": set(Node), "reward": int}
        where:
          - `nodes` are the ship positions at that step (only ships with energy >= 0)
          - `reward` is the observed points gained by the team in that step

        We treat each result as a constraint satisfaction problem:
          - Some of the nodes might already be known reward tiles (contribute +1 each)
          - Some nodes might already be known empty (contribute +0)
          - The remaining nodes are unknown and must explain the residual reward

        Deterministic implications we can safely apply:
          - If residual reward == 0 => all unknown nodes are empty
          - If residual reward == len(unknown_nodes) => all unknown nodes are reward tiles

        We intentionally do NOT fully solve the general case (e.g. residual reward is in between),
        because that would require combinatorial reasoning / more assumptions.
        """
        # We will use Global.REWARD_RESULTS to identify which nodes yield points
        for result in Global.REWARD_RESULTS:

            # `unknown_nodes` = ship-occupied nodes whose reward status we haven't proven yet.
            unknown_nodes = set()
            known_reward = 0

            # `known_reward` counts how many of the occupied nodes are already confirmed reward tiles.
            # Those tiles must contribute +1 each to the team reward.
            for n in result["nodes"]:
                # If we've already explored this tile and confirmed it is NOT a reward,
                # then this node cannot contribute and we can ignore it.
                if n.explored_for_reward and not n.reward:
                    continue

                # If we already know it's a reward tile, account for it immediately.
                if n.reward:
                    known_reward += 1
                    continue

                # Otherwise reward status is still unknown.
                unknown_nodes.add(n)

            if not unknown_nodes:
                # All ship positions were already fully explained by known tiles.
                # Nothing new can be inferred from this result.
                continue

            # Observed team reward minus the part we can already explain by known reward tiles.
            # This is what the unknown nodes must account for.
            reward = result["reward"] - known_reward  # reward from unknown_nodes

            if reward == 0:
                # None of the unknown nodes yielded points => they are all empty.
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)

            elif reward == len(unknown_nodes):
                # Every unknown node must have yielded +1 => they are all reward tiles.
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)

            elif reward > len(unknown_nodes):
                # This should be impossible if the modeling assumptions match the environment.
                # In practice, it can happen due to:
                #   - bookkeeping mismatches (e.g. ship set not matching env's scoring set)
                #   - delayed/aggregated reward attribution across steps
                #   - or a bug/edge-case in our inference
                # The original solution chose to ignore it.
                pass

    def _update_reward_results(self, obs, team_id, team_reward):
        """Record one step of evidence for later reward inference.

        What this stores
        ----------------
        We build a set of *currently occupied* nodes by our ships and pair that with the
        observed points gained (`team_reward`). This is appended to `Global.REWARD_RESULTS`.

        Why this helps
        --------------
        Points are awarded based on ship positions, but the environment doesn't tell us
        which exact tiles gave points. Over many steps, the pair (occupied tiles, points gained)
        yields constraints that `_update_reward_status_from_reward_results()` can use to
        mark some tiles as certainly reward / certainly empty.

        Important detail: energy >= 0
        ----------------------------
        The code only adds ships with non-negative energy.
        Rationale (as used by the original solution): those are the only ships considered
        valid for scoring ("Only units with non-negative energy can give points").
        """
        ship_nodes = set()

        # Iterate over our team ships (mask, energy, position are aligned by unit_id).
        for active, energy, position in zip(
                obs["units_mask"][team_id],
                obs["units"]["energy"][team_id],
                obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                # (this mirrors the assumptions in the original Kaggle solution)
                ship_nodes.add(self.get_node(*position))

        # Append as one data point for later reasoning.
        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        """Prune impossible reward tiles based on the "rewards occur near relics" rule.

        High-level rule
        --------------
        Rewards can only appear in a local neighborhood around relics.
        Therefore, if a tile is *guaranteed* to be outside the possible relic influence,
        it cannot be a reward tile and can be marked as empty.

        Why we do this
        --------------
        The reward-inference from point deltas alone may be underdetermined. This function
        reduces the search space by marking tiles that are geometrically impossible.

        Implementation sketch
        ---------------------
        1. Build a binary `relic_map` of where relics could be:
           - 1 for tiles that are confirmed relics (node.relic)
           - 1 also for tiles that are NOT fully explored for relics yet
             (meaning: a relic could still be there)
           - 0 for tiles confirmed to have no relic

        2. Convolve `relic_map` with a square kernel of size (2*R+1)x(2*R+1),
           where R = Global.RELIC_REWARD_RANGE.
           The resulting `reward_map[y][x]` equals how many potential/known relic tiles
           fall within Chebyshev distance R (square neighborhood) of (x,y).

        3. If reward_map[y][x] == 0, then there are no (possible) relics within range,
           so (x,y) cannot be a reward tile => mark as empty.

        Notes / edge-cases
        ------------------
        - This pruning is conservative: it only marks tiles as empty when they are
          impossible under the relic-neighborhood rule.
        - It does NOT mark tiles as reward tiles; that requires additional evidence.
        """
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        # `relic_map[y][x]` = 1 means "a relic could be at (x,y)".
        # We treat unknown/not-yet-explored-for-relic tiles as possible relic locations.
        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1

        # Neighborhood side length for the convolution kernel.
        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1

        # Convolution counts possible relic tiles in the neighborhood of each cell.
        # Using `mode="same"` keeps the output the same size (SPACE_SIZE x SPACE_SIZE).
        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # No (possible) relics in range RELIC_REWARD_RANGE => reward impossible here.
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]

            if (
                    is_visible
                    and not node.is_unknown
                    and node.type.value != obs_tile_type[x, y]
            ):
                obstacles_shifted = True

            if (
                    is_visible
                    and node.energy is not None
                    and node.energy != obs_energy[x, y]
            ):
                energy_nodes_shifted = True

        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)

        def clear_map_info():
            for n in self:
                n.type = NodeType.unknown

        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction

                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                clear_map_info()

        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(
                Global.OBSTACLES_MOVEMENT_STATUS
            )
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period

            if obstacles_shifted:
                clear_map_info()

        if (
                obstacles_shifted
                and Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
                and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
        ):
            # maybe something is wrong
            clear_map_info()

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_node(*get_opposite(x, y)).energy = node.energy

            elif energy_nodes_shifted:
                # The energy field has changed
                # I cannot predict what the new energy field will be like.
                node.energy = None

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
        if len(obstacles_movement_status) < 81:
            return

        num_movements = sum(obstacles_movement_status)

        if num_movements <= 2:
            return 40
        elif num_movements <= 4:
            return 20
        elif num_movements <= 8:
            return 10
        else:
            return 20 / 3

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                        sensor_mask[x, y]
                        and not node.is_unknown
                        and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        if (
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
                and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
                and Global.OBSTACLE_MOVEMENT_PERIOD > 0
        ):
            speed = 1 / Global.OBSTACLE_MOVEMENT_PERIOD
            if (step - 2) * speed % 1 > (step - 1) * speed % 1:
                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self

    def clear_exploration_info(self):
        Global.REWARD_RESULTS = []
        Global.ALL_RELICS_FOUND = False
        Global.ALL_REWARDS_FOUND = False
        for node in self:
            if not node.relic:
                self._update_relic_status(node.x, node.y, status=None)


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])

        for ship, active, position, energy in zip(
                self.ships,
                obs["units_mask"][self.team_id],
                obs["units"]["position"][self.team_id],
                obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()


class Agent:

    def __init__(self, player: str) -> None:
        self.name = 'Solution_92_Kaggle'
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

    def set_env_cfg(self, env_cfg):
        self.env_cfg = env_cfg
        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)
        match_number = get_match_number(step)

        if match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            if match_number <= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info()
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        # self.show_explored_map()
        self.harvest()
        self.find_rewards()  # changed the call order of reward and relic
        self.find_relics()
        self.fallback_explore()

        return self.create_actions_array()

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)
        for i, ship in enumerate(ships):
            if ship.action is not None:
                if ship.task == "sap" and ship.target is not None:
                    # For sap, the action must be [5, target.x, target.y]
                    actions[i] = [ActionType.sap, ship.target.x, ship.target.y]
                else:
                    actions[i] = [ship.action, 0, 0]
        return actions

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        """
        1. If all rewards have already been found, clear the task and exit.
        2. Get unexplored relics (unexplored_relics).
        3. Check whether units already in find_rewards have a valid target; reset if invalid.
        4. Create 'high_priority_targets' and prioritize exploring the 5x5 area around relics/rewards found at step i.
        5. For all units, assign tasks to move to either the highest-priority targets or normal targets.
        """
        # 1. If all rewards are already discovered, clear tasks
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        # 2. Get unexplored relics
        unexplored_relics = self.get_unexplored_relics()

        # 3. Check whether units already in "find_rewards" have a valid target
        for ship in self.fleet:
            if ship.task == "find_rewards":
                # If there is no target, clear the task
                if ship.target is None:
                    ship.task = None
                    continue

                # Keep it if the target matches an unexplored relic and there is enough energy
                if (ship.target in unexplored_relics) and (ship.energy > Global.UNIT_MOVE_COST * 2):
                    # Continue exploring
                    pass
                else:
                    # Reset if conditions are not met
                    ship.task = None
                    ship.target = None

        # 4. Build a target set to preferentially explore areas around relics/rewards found at step i
        #    - 5x5 around relics (areas where rewards have not been found yet)
        #    - 5x5 around rewards (areas where relics have not been found yet)
        high_priority_targets = set()

        # 4.1 5x5 around relics
        for relic_node in unexplored_relics:
            for x, y in nearby_positions(relic_node.x, relic_node.y, Global.RELIC_REWARD_RANGE):
                node = self.space.get_node(x, y)
                if node.is_walkable and not node.explored_for_reward:
                    high_priority_targets.add((x, y))

        # 4.2 5x5 around rewards
        for reward_node in self.space.reward_nodes:
            for x, y in nearby_positions(reward_node.x, reward_node.y, Global.RELIC_REWARD_RANGE):
                node = self.space.get_node(x, y)
                # Areas around rewards where relic exploration is not finished yet
                if node.is_walkable and not node.explored_for_relic:
                    high_priority_targets.add((x, y))

        # Avoid assigning duplicate targets that other units are already heading to
        assigned_targets = set()

        def set_task(ship):
            """
            Assign a "search for rewards" task to the unit.
            First prioritize high_priority_targets (5x5 around newly found areas);
            if none, explore around normal unexplored_relics.
            """
            # --------------------------------------------
            # A) Explore top-priority targets (high_priority_targets)
            # --------------------------------------------
            best_target, _ = find_closest_target(ship.coordinates, list(high_priority_targets))
            if best_target and best_target not in assigned_targets:
                # Check that the move cost fits within ship.energy
                path = astar(create_weights(self.space), ship.coordinates, best_target)
                if path:
                    cost_est = estimate_energy_cost(self.space, path)
                    if cost_est <= ship.energy:
                        actions = path_to_actions(path)
                        if actions:
                            # Set task, target, and first action
                            ship.task = "find_rewards"
                            ship.target = self.space.get_node(*best_target)
                            ship.action = actions[0]
                            assigned_targets.add(best_target)
                            # [Added] If the target location is a reward, switch to harvest
                            if self.space.get_node(*best_target) in self.space.reward_nodes:
                                ship.task = "harvest"
                            #     ship.action = ActionType.center  # stay in place
                            return  # End here => the unit goes to the high-priority target

            # --------------------------------------------
            # B) Normal targets (around unexplored_relics)
            # --------------------------------------------
            normal_targets = []
            for relic_node in unexplored_relics:
                for x, y in nearby_positions(relic_node.x, relic_node.y, Global.RELIC_REWARD_RANGE):
                    node = self.space.get_node(x, y)
                    if node.is_walkable and not node.explored_for_reward:
                        normal_targets.append((x, y))

            # Find the nearest target
            best_target, _ = find_closest_target(ship.coordinates, normal_targets)
            if not best_target:
                return  # If there is no target, do nothing (ideally move via another fallback)

            if best_target in assigned_targets:
                return  # Skip if another unit is already heading to the same target

            # Pathfinding / route calculation
            path = astar(create_weights(self.space), ship.coordinates, best_target)
            if not path:
                return

            cost_est = estimate_energy_cost(self.space, path)
            if cost_est <= ship.energy:
                actions = path_to_actions(path)
                if actions:
                    ship.task = "find_rewards"
                    ship.target = self.space.get_node(*best_target)
                    ship.action = actions[0]
                    assigned_targets.add(best_target)
                    # if self.space.get_node(*best_target) in self.space.reward_nodes:
                    #     ship.task = "harvest"
                    #     ship.action = ActionType.center

        # 5. Assign tasks for all units
        for ship in self.fleet:
            # Skip if there is already another task
            if ship.task and ship.task != "find_rewards":
                continue

            # If energy is insufficient, it cannot move => skip or fall back
            if ship.energy < Global.UNIT_MOVE_COST * 2:
                continue

            # Task assignment
            set_task(ship)

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                    *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes

    def fallback_explore(self):
        """
        Assign exploration tasks to ships that have nowhere to go (no task / action == None).
        Example: use A* to move toward unknown tiles in our team's sector.
        """
        # Collect unknown tiles (type == unknown) within our team's sector
        unknown_positions = []
        for node in self.space:
            if node.is_unknown:
                # Assumes the map is split by sectors
                if is_team_sector(self.team_id, node.x, node.y):
                    unknown_positions.append(node.coordinates)

        if not unknown_positions:
            # If there are no unknown tiles in the team sector, don't force movement / could random-walk
            return

        for ship in self.fleet:
            if ship.action is not None:
                continue  # Skip if an action is already set by another task

            # Give the closest unknown tile as a target
            target_xy, dist = find_closest_target(ship.coordinates, unknown_positions)
            if target_xy:
                path = astar(create_weights(self.space), ship.coordinates, target_xy)
                if path:
                    cost_est = estimate_energy_cost(self.space, path)
                    if cost_est <= ship.energy:
                        actions = path_to_actions(path)
                        if actions:
                            # Set the first step
                            ship.action = actions[0]
                            # ship.task = "fallback_explore"  # optional
                            # Reserve the target so other ships don't pick the same tile
                            unknown_positions.remove(target_xy)
                    else:
                        # Not enough energy
                        pass

    def harvest(self):
        def set_task(ship, target_node):
            if ship.node == target_node:
                # Look for enemy units
                sap_flag = False
                for enemy in self.opp_fleet.ships:
                    if enemy.node is None:
                        continue
                    # Evaluate whether sap is possible via Manhattan distance (within Global.UNIT_SAP_RANGE)
                    d = abs(ship.coordinates[0] - enemy.node.x) + abs(ship.coordinates[1] - enemy.node.y)
                    if d <= Global.UNIT_SAP_RANGE and ship.energy >= Global.UNIT_SAP_COST:
                        # Enemy is visible and we can sap => switch task to sap
                        ship.task = "sap"
                        relative_x = enemy.node.x - ship.coordinates[0]
                        relative_y = enemy.node.y - ship.coordinates[1]
                        relative_node = Node(relative_x, relative_y)
                        ship.target = relative_node
                        ship.action = ActionType.sap
                        sap_flag = True

                if sap_flag is False:
                    ship.task = "harvest"
                    ship.target = target_node
                    ship.action = ActionType.center

                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "sap":
                ship.task = "harvest"
                ship.target = Node(ship.coordinates[0], ship.coordinates[1])
                booked_nodes.add(ship.target)
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    if ship.task == "sap":
                        booked_nodes.add(Node(ship.coordinates[0], ship.coordinates[1]))
                    else:
                        booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None
'''
    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
'''

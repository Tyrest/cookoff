import heapq
import itertools
import random

import copy
from collections import deque, Counter

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from game_constants import FoodType, ShopCosts, Team, TileType
from item import Food, Pan, Plate
from robot_controller import RobotController
from time import time

def required_counter(order) -> Counter:
    # order["required"] is a list and can contain duplicates like ["EGG","EGG"]
    return Counter(order["required"])

def plate_counter_from_tile(tile, controller: RobotController) -> Counter:
    """
    Return counts of food_name on a plate on this tile.
    Uses item_to_public_dict so we don't rely on internal state.
    """
    if not tile or not tile.item:
        return Counter()

    pub = controller.item_to_public_dict(tile.item)  # allowed API :contentReference[oaicite:1]{index=1}
    if not isinstance(pub, dict) or pub.get("type") != "Plate":
        return Counter()

    foods = pub.get("food", [])  # list of food dicts :contentReference[oaicite:2]{index=2}
    return Counter(f.get("food_name") for f in foods if isinstance(f, dict) and f.get("food_name"))

def remaining_needed(order, plate_counts: Counter) -> Counter:
    need = required_counter(order)
    # subtract what we already plated
    need.subtract(plate_counts)
    # keep only positive needs
    return Counter({k: v for k, v in need.items() if v > 0})
DEBUG = False


class States(Enum):
    INIT = auto()
    BUY_PAN = auto()
    BUY_MEAT = auto()
    PUT_MEAT_ON_COUNTER = auto()
    CHOP_MEAT = auto()
    PICKUP_MEAT = auto()
    PUT_MEAT_ON_COOKER = auto()
    BUY_EGG = auto()
    PUT_EGG_ON_COOKER = auto()
    BUY_ONIONS = auto()
    PUT_ONIONS_ON_COUNTER = auto()
    CHOP_ONIONS = auto()
    PICKUP_CHOPPED_ONIONS = auto()
    STORE_CHOPPED_ONIONS = auto()
    BUY_PLATE = auto()
    PUT_PLATE_ON_COUNTER = auto()
    BUY_NOODLES = auto()
    ADD_NOODLES_TO_PLATE = auto()
    BUY_SAUCE = auto()
    ADD_SAUCE_TO_PLATE = auto()
    WAIT_FOR_MEAT = auto()
    ADD_MEAT_TO_PLATE = auto()
    WAIT_FOR_EGG = auto()
    ADD_EGG_TO_PLATE = auto()
    PICKUP_PLATE = auto()
    SUBMIT_ORDER = auto()
    TRASH_ITEM = auto()
    TRASH_PLATE = auto()  # Trash items from plate when order expires
    DISCARD_PLATE = auto()  # Put empty plate on counter after trashing items
    RETRIEVE_BOX_ITEM = auto()
    PLATE_BOX_ITEM = auto()
    
    
class SabotageState(Enum):
    MOVE_TO_PAN = auto()
    MOVE_TO_TRASH = auto()
    PLACE_PAN_ON_COUNTER = auto()
    MOVE_TO_PLATE = auto()


class BotWorkerState:
    """Per-bot state for independent order processing."""

    def __init__(self, bot_id: int):
        self.bot_id = bot_id
        self.state = States.INIT
        self.last_state: Optional[States] = None
        self.current_order: Optional[dict] = None
        self.counter_loc: Optional[Tuple[int, int]] = None
        self.cooker_loc: Optional[Tuple[int, int]] = None
        self.box_loc: Optional[Tuple[int, int]] = None
        # Stuck detection
        self.last_pos: Optional[Tuple[int, int]] = None
        self.stuck_turns: int = 0
        self.has_plate: bool = False


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.fulfilled_orders: set[int] = set()
        self.claimed_orders: set[int] = set()  # Orders claimed by any bot
        self.worker_states: dict[int, BotWorkerState] = {}  # Per-bot state
        
        # Sabotage state
        self.sabotage = False
        self.snapshot = None
        self.sabotage_state = None
        # Capability flags (set after first turn analysis)
        self._capabilities_analyzed = False
        self._has_cooker = True
        self._has_storage = True  # Have 2+ counters or 1 counter + box
        self._can_do_onions = True  # Need storage for onions

        # Set random seed for reproducibility
        random.seed(42)

    def _flood_fill(self, start_x: int, start_y: int) -> Set[Tuple[int, int]]:
        """Flood fill to find all reachable tiles from a starting position."""
        reachable = set()
        queue = deque([(start_x, start_y)])
        w, h = self.map.width, self.map.height

        while queue:
            x, y = queue.popleft()
            if (x, y) in reachable:
                continue
            if not (0 <= x < w and 0 <= y < h):
                continue
            # Include the tile itself (even if not walkable) to check adjacent tiles
            reachable.add((x, y))
            # Only expand through walkable tiles
            if not self.map.is_tile_walkable(x, y):
                continue
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    queue.append((x + dx, y + dy))
        return reachable

    def _analyze_capabilities(self, controller: RobotController) -> None:
        """Analyze what tiles are reachable and set capability flags."""
        if self._capabilities_analyzed:
            return
        self._capabilities_analyzed = True

        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return

        # Get reachable tiles from first bot
        first_state = controller.get_bot_state(my_bots[0])
        if not first_state:
            return
        reachable = self._flood_fill(first_state["x"], first_state["y"])

        # Count accessible tile types
        counters = 0
        cookers = 0
        boxes = 0
        has_shop = False
        has_submit = False

        for x, y in reachable:
            if 0 <= x < self.map.width and 0 <= y < self.map.height:
                tile = self.map.tiles[x][y]
                if tile.tile_name == "COUNTER":
                    counters += 1
                elif tile.tile_name == "COOKER":
                    cookers += 1
                elif tile.tile_name == "BOX":
                    boxes += 1
                elif tile.tile_name == "SHOP":
                    has_shop = True
                elif tile.tile_name == "SUBMIT":
                    has_submit = True

        if DEBUG:
            print(
                f"Accessible tiles: {counters} counters, {cookers} cookers, {boxes} boxes, shop={has_shop}, submit={has_submit}"
            )

        # Set capability flags
        self._has_cooker = cookers > 0
        # Need 2 counters OR 1 counter + 1 box for storage (one for plate, one for onions)
        self._has_storage = (counters >= 2) or (counters >= 1 and boxes >= 1)
        # Can do onions if we have storage capability
        self._can_do_onions = self._has_storage

        if DEBUG:
            if not self._has_cooker:
                print("No cooker accessible - will skip orders requiring meat/egg")
            if not self._has_storage:
                print("Limited storage - will skip orders requiring onions")

    def get_proximity_cost(
        self, pos: Tuple[int, int], other_bots: Set[Tuple[int, int]]
    ) -> float:
        """Calculate a soft penalty for being near other bots.
        Returns 0 if no bots nearby, increasing cost for closer proximity.
        """
        if not other_bots:
            return 0
        x, y = pos
        cost = 0.0
        for bx, by in other_bots:
            dist = max(abs(x - bx), abs(y - by))  # Chebyshev distance
            if dist == 0:
                cost += 5.0  # High cost for exact position
            elif dist == 1:
                cost += 2.0  # Medium cost for adjacent
            elif dist == 2:
                cost += 0.5  # Small cost for 2 tiles away
        return cost

    def get_weighted_path(
        self,
        controller: RobotController,
        start: Tuple[int, int],
        target_predicate,
        other_bot_positions: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Find path using Dijkstra's algorithm with proximity costs.
        Bots prefer to avoid each other but can pass through if needed.
        """
        other_bot_positions = other_bot_positions or set()
        # Priority queue: (cost, x, y, path)
        pq = [(0.0, start[0], start[1], [])]
        visited: Dict[Tuple[int, int], float] = {}
        w, h = self.map.width, self.map.height

        while pq:
            cost, curr_x, curr_y, path = heapq.heappop(pq)

            if (curr_x, curr_y) in visited:
                continue
            visited[(curr_x, curr_y)] = cost

            tile = controller.get_tile(controller.get_team(), curr_x, curr_y)
            if target_predicate(curr_x, curr_y, tile):
                if not path:
                    return (0, 0)
                return path[0]

            for dx in [0, -1, 1]:
                for dy in [0, -1, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if self.map.is_tile_walkable(nx, ny):
                            # Base move cost is 1, plus proximity penalty
                            move_cost = 1.0 + self.get_proximity_cost(
                                (nx, ny), other_bot_positions
                            )
                            new_cost = cost + move_cost
                            heapq.heappush(pq, (new_cost, nx, ny, path + [(dx, dy)]))
        return None

    def get_other_bot_positions(
        self, controller: RobotController, exclude_bot_id: int
    ) -> Set[Tuple[int, int]]:
        """Get positions of all other bots for proximity-aware pathfinding."""
        positions = set()
        for bid in controller.get_team_bot_ids(controller.get_team()):
            if bid != exclude_bot_id:
                state = controller.get_bot_state(bid)
                if state:
                    positions.add((state["x"], state["y"]))
        return positions

    def move_randomly(
        self, controller: RobotController, bot_id: int, avoid: Set[Tuple[int, int]]
    ) -> bool:
        """Move in a random valid direction to get unstuck."""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        bx, by = bot_state["x"], bot_state["y"]

        possible = []
        for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
            if dx == 0 and dy == 0:
                continue
            nx, ny = bx + dx, by + dy
            if self.map.is_tile_walkable(nx, ny) and (nx, ny) not in avoid:
                possible.append((dx, dy))

        if possible:
            dx, dy = random.choice(possible)
            controller.move(bot_id, dx, dy)
            return True
        return False

    def is_blocking_position(self, x: int, y: int) -> bool:
        """Check if a position is in a narrow area that could block other bots.
        Returns True if the position has 3 or fewer walkable neighbors (corridor-like).
        """
        walkable_neighbors = 0
        for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map.width and 0 <= ny < self.map.height:
                if self.map.is_tile_walkable(nx, ny):
                    walkable_neighbors += 1
        # If 3 or fewer walkable neighbors, it's a narrow/blocking spot
        return walkable_neighbors <= 3

    def count_stuck_workers(self) -> int:
        """Count how many workers are currently stuck."""
        return sum(1 for w in self.worker_states.values() if w.stuck_turns >= 5)

    def move_towards(
        self,
        controller: RobotController,
        bot_id: int,
        target_x: int,
        target_y: int,
        other_bot_positions: Optional[Set[Tuple[int, int]]] = None,
    ) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        bx, by = bot_state["x"], bot_state["y"]
        other_bot_positions = other_bot_positions or set()

        def is_adjacent_to_target(x, y, tile):
            return max(abs(x - target_x), abs(y - target_y)) <= 1

        if is_adjacent_to_target(bx, by, None):
            return True

        # Use weighted pathfinding that prefers avoiding other bots but doesn't hard-block
        step = self.get_weighted_path(
            controller, (bx, by), is_adjacent_to_target, other_bot_positions
        )
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
            return False
        return False

    def find_nearest_tile(
        self,
        controller: RobotController,
        bot_x: int,
        bot_y: int,
        tile_name: str,
        avoid_positions: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest tile of given type using BFS (actual walking distance)."""
        avoid_positions = avoid_positions or set()
        queue = deque([(bot_x, bot_y, 0)])
        visited = {(bot_x, bot_y)}
        w, h = self.map.width, self.map.height

        while queue:
            cx, cy, dist = queue.popleft()
            # Check adjacent tiles for the target type
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        tile = self.map.tiles[nx][ny]
                        if tile.tile_name == tile_name:
                            return (nx, ny)
            # Expand to walkable neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if (
                            self.map.is_tile_walkable(nx, ny)
                            and (nx, ny) not in avoid_positions
                        ):
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))
        return None

    def _bfs_distance(
        self, start_x: int, start_y: int, target_x: int, target_y: int
    ) -> Optional[int]:
        """Calculate BFS distance from start to a position adjacent to target."""
        if max(abs(start_x - target_x), abs(start_y - target_y)) <= 1:
            return 0
        queue = deque([(start_x, start_y, 0)])
        visited = {(start_x, start_y)}
        w, h = self.map.width, self.map.height

        while queue:
            cx, cy, dist = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if max(abs(nx - target_x), abs(ny - target_y)) <= 1:
                            return dist + 1
                        if self.map.is_tile_walkable(nx, ny):
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))
        return None

    def find_nearest_tile_of_types(
        self,
        controller: RobotController,
        bot_x: int,
        bot_y: int,
        tile_names: List[str],
        exclude: Optional[List[Tuple[int, int]]] = None,
        avoid_positions: Optional[Set[Tuple[int, int]]] = None,
        max_distance_from: Optional[Tuple[Tuple[int, int], int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest tile matching any of the given types using BFS, excluding specified positions.

        Args:
            max_distance_from: Optional tuple of ((x, y), max_dist) - if provided, only return
                               tiles that are within max_dist BFS moves from the given position.

        If multiple tiles are found at the same distance, prefers tiles with more walkable
        neighbors (more accessible locations). Tiles with only 1 walkable neighbor get +2 distance penalty.
        """
        exclude = exclude or []
        avoid_positions = avoid_positions or set()
        queue = deque([(bot_x, bot_y, 0)])
        visited = {(bot_x, bot_y)}
        w, h = self.map.width, self.map.height

        def count_walkable_neighbors(pos: Tuple[int, int]) -> int:
            px, py = pos
            count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.map.is_tile_walkable(nx, ny):
                            count += 1
            return count

        # Collect all candidates with their effective distances
        candidates = []

        while queue:
            cx, cy, dist = queue.popleft()

            # Check adjacent tiles for the target types
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in exclude:
                        tile = self.map.tiles[nx][ny]
                        if tile.tile_name in tile_names:
                            # Check max_distance_from constraint if provided
                            if max_distance_from:
                                ref_pos, max_dist = max_distance_from
                                dist_from_ref = self._bfs_distance(
                                    ref_pos[0], ref_pos[1], nx, ny
                                )
                                if dist_from_ref is None or dist_from_ref > max_dist:
                                    continue  # Skip this tile, too far from reference
                            # Calculate effective distance with penalty for tight spaces
                            effective_dist = dist
                            if count_walkable_neighbors((nx, ny)) <= 2:
                                effective_dist += (
                                    3  # Penalty for tiles with only 1 floor around
                                )
                            if (nx, ny) not in [c[0] for c in candidates]:
                                candidates.append(((nx, ny), effective_dist))

            # Expand to walkable neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if (
                            self.map.is_tile_walkable(nx, ny)
                            and (nx, ny) not in avoid_positions
                        ):
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))

        if not candidates:
            return None

        # Sort by effective distance (ascending), then by walkable neighbors (descending) for ties
        candidates.sort(key=lambda c: (c[1], -count_walkable_neighbors(c[0])))
        return candidates[0][0]

    def calculate_ingredient_cost(self, ingredient_list: List[str]) -> int:
        total_cost = 0
        total_cost += ShopCosts.PLATE.buy_cost
        for ingredient in ingredient_list:
            if ingredient == FoodType.MEAT.food_name:
                total_cost += FoodType.MEAT.buy_cost
            elif ingredient == FoodType.EGG.food_name:
                total_cost += FoodType.EGG.buy_cost
            elif ingredient == FoodType.ONIONS.food_name:
                total_cost += FoodType.ONIONS.buy_cost
            elif ingredient == FoodType.NOODLES.food_name:
                total_cost += FoodType.NOODLES.buy_cost
            elif ingredient == FoodType.SAUCE.food_name:
                total_cost += FoodType.SAUCE.buy_cost
        return total_cost

    def estimate_time(self, ingredient_list: List[str]) -> int:
        total_cost = 0
        for ingredient in ingredient_list:
            if ingredient == FoodType.MEAT.food_name:
                total_cost += 50
            elif ingredient == FoodType.EGG.food_name:
                total_cost += 40
            elif ingredient == FoodType.ONIONS.food_name:
                total_cost += 25
            elif ingredient == FoodType.NOODLES.food_name:
                total_cost += 20
            elif ingredient == FoodType.SAUCE.food_name:
                total_cost += 20
        return total_cost

    def select_next_order(self, controller: RobotController, worker: BotWorkerState):
        """Select next order for a specific bot worker."""
        orders = controller.get_orders(controller.get_team())
        best_order = None
        best_value = -float("inf")
        team_money = controller.get_team_money(controller.get_team())

        # Reserve money for already claimed orders
        reserved_money = 0
        for ws in self.worker_states.values():
            if ws.current_order and ws.bot_id != worker.bot_id:
                reserved_money += self.calculate_ingredient_cost(
                    ws.current_order["required"]
                )
        available_money = team_money - reserved_money * 0.5

        for order in orders:
            meat_count, ontions_count = 0, 0
            for ingredient in order["required"]:
                if ingredient == FoodType.MEAT.food_name:
                    meat_count += 1
                if ingredient == FoodType.ONIONS.food_name:
                    ontions_count += 1
            if meat_count > 1 or ontions_count > 1:
                continue

            order_id = order["order_id"]
            # Skip if already fulfilled or claimed by another bot
            if order_id in self.fulfilled_orders or order_id in self.claimed_orders:
                continue
            # Skip orders with duplicate ingredients
            required = order["required"]
            if len(required) != len(set(required)):
                continue
            # Skip orders we can't fulfill due to missing tiles
            if not self._has_cooker:
                if (
                    FoodType.MEAT.food_name in required
                    or FoodType.EGG.food_name in required
                ):
                    continue
            if not self._can_do_onions:
                if FoodType.ONIONS.food_name in required:
                    continue
            time_remaining = order["expires_turn"] - controller.get_turn()
            if time_remaining <= 0:
                continue
            if time_remaining < len(order["required"]) * 50:
                continue
            ingredient_cost = self.calculate_ingredient_cost(order["required"])
            value = order["reward"] - ingredient_cost
            if value <= 0:
                continue
            value = value / time_remaining
            # Slightly penalize orders with meat or eggs (harder due to cooking)
            required = order["required"]
            if FoodType.MEAT.food_name in required:
                value *= 0.80  # 20% penalty for cooking-required orders
            if FoodType.EGG.food_name in required:
                value *= 0.80  # 20% penalty for egg orders
            if value > best_value and available_money >= ingredient_cost:
                best_value = value
                best_order = order

        if best_order:
            worker.current_order = best_order
            self.claimed_orders.add(best_order["order_id"])
            if DEBUG:
                print(f"Bot {worker.bot_id} selected order: {best_order}")

    def play_turn(self, controller: RobotController):
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return

        # Analyze capabilities on first turn
        self._analyze_capabilities(controller)

        # Initialize worker states for each bot
        for bot_id in my_bots:
            if bot_id not in self.worker_states:
                self.worker_states[bot_id] = BotWorkerState(bot_id)

        # Get all other bots' locations to use as exclusions
        all_counters = []
        all_cookers = []
        all_boxes = []
        for worker in self.worker_states.values():
            if worker.counter_loc:
                all_counters.append(worker.counter_loc)
            if worker.cooker_loc:
                all_cookers.append(worker.cooker_loc)
            if worker.box_loc:
                all_boxes.append(worker.box_loc)
                
        # If sabotage state is active, enter sabotage state
        if self.sabotage:
            switch_info = controller.get_switch_info()
            
            if switch_info["window_active"]:
                self.run_sabotage(controller)
                return 
            # Recover from sabotage state
            else:
                self.sabotage = False
                self.sabotage_state = None
                print(f"Recovered from sabotage state at turn {controller.get_turn()}")
            
        # Check condition for sabotage
        if controller.get_turn() in range(250, 325) and controller.can_switch_maps():
            # Check the states of the workers
            # 1. If no worker is holding anything
            # 2. If no cooker is cooking anything
            flag = True
            for worker in self.worker_states.values():
                if controller.get_bot_state(worker.bot_id).get("holding"):
                    flag = False
                    break
                if worker.state in [States.WAIT_FOR_EGG, States.WAIT_FOR_MEAT]:
                    flag = False
                    break
                # Check if pan is cooking
                for cooker in all_cookers:
                    pan = controller.get_tile(controller.get_team(), cooker[0], cooker[1]).item
                    if isinstance(pan, Pan) and pan.food is not None and pan.food.cooked_stage == 0:
                        flag = False
                        break
                
            if flag:
                if controller.switch_maps():
                    self.sabotage = True
                    self.sabotage_state = SabotageState.MOVE_TO_PAN
                    print(f"Sabotage state activated at turn {controller.get_turn()}")
                    print(f"Worker states: {self.worker_states[my_bots[0]].state}, {self.worker_states[my_bots[1]].state}")
                    
                    # Save snapshot of the worker states
                    self.snapshot = copy.deepcopy(self.worker_states)
                    self.run_sabotage(controller)
                    return
            
        # TODO: Implement recovering from sabotage state
        # ...

        # Run each bot independently
        for i, bot_id in enumerate(my_bots):
            worker = self.worker_states[bot_id]

            # Select order if none
            if worker.current_order is None:
                self.select_next_order(controller, worker)
                
                if worker.current_order is None:
                    continue

            # Get exclusions (tiles claimed by other bots)
            other_counters = [
                ws.counter_loc
                for ws in self.worker_states.values()
                if ws.bot_id != bot_id and ws.counter_loc
            ]
            other_cookers = [
                ws.cooker_loc
                for ws in self.worker_states.values()
                if ws.bot_id != bot_id and ws.cooker_loc
            ]
            other_boxes = [
                ws.box_loc
                for ws in self.worker_states.values()
                if ws.bot_id != bot_id and ws.box_loc
            ] + other_counters

            # Get other bot positions for proximity-aware pathfinding
            other_bot_positions = self.get_other_bot_positions(controller, bot_id)

            self.run_worker(
                controller,
                worker,
                other_counters,
                other_cookers,
                other_boxes,
                other_bot_positions,
            )

    def run_worker(
        self,
        controller: RobotController,
        worker: BotWorkerState,
        exclude_counters: List[Tuple[int, int]],
        exclude_cookers: List[Tuple[int, int]],
        exclude_boxes: List[Tuple[int, int]],
        other_bot_positions: Set[Tuple[int, int]],
    ) -> None:
        """Run the state machine for a single bot worker."""
        if worker.last_state != worker.state:
            if DEBUG:
                print(
                    f"t={controller.get_turn()} Bot {worker.bot_id} state: {worker.state}"
                )
            worker.last_state = worker.state

        bot_id = worker.bot_id
        bot_info = controller.get_bot_state(bot_id)
        if bot_info is None:
            return
        bx, by = bot_info["x"], bot_info["y"]

        # Stuck detection: if bot hasn't moved in 5 turns, consider moving randomly
        current_pos = (bx, by)
        if worker.last_pos == current_pos:
            worker.stuck_turns += 1
        else:
            worker.stuck_turns = 0
            worker.last_pos = current_pos

        if worker.stuck_turns >= 5:
            # Only move randomly if:
            # 1. We're in a blocking position (narrow corridor), OR
            # 2. Multiple bots are stuck (deadlock)
            is_blocking = self.is_blocking_position(bx, by)
            num_stuck = self.count_stuck_workers()

            if is_blocking or num_stuck >= 2:
                if DEBUG:
                    print(
                        f"Bot {bot_id} is stuck (blocking={is_blocking}, num_stuck={num_stuck}), moving randomly"
                    )
                if self.move_randomly(controller, bot_id, other_bot_positions):
                    worker.stuck_turns = 0
                    return
            # Otherwise, just wait in place - we're not blocking anyone

        # Find workstations for this bot, excluding ones claimed by others
        # First find shop to use as reference for cooker distance constraint
        shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")

        # Determine what this order needs
        if worker.current_order is None:
            return
        needs_cooker = (
            FoodType.MEAT.food_name in worker.current_order["required"]
            or FoodType.EGG.food_name in worker.current_order["required"]
        )
        needs_storage = FoodType.ONIONS.food_name in worker.current_order["required"]

        if worker.counter_loc is None:
            # Only consider counters within 16 moves of the shop
            worker.counter_loc = self.find_nearest_tile_of_types(
                controller,
                bx,
                by,
                ["COUNTER"],
                exclude=exclude_counters,
                avoid_positions=other_bot_positions,
                max_distance_from=(shop_pos, 16) if shop_pos else None,
            )
        if worker.counter_loc is None:
            return

        if needs_cooker and worker.cooker_loc is None:
            # Only consider cookers within 19 moves of the shop
            worker.cooker_loc = self.find_nearest_tile_of_types(
                controller,
                bx,
                by,
                ["COOKER"],
                exclude=exclude_cookers,
                avoid_positions=other_bot_positions,
                max_distance_from=(shop_pos, 19) if shop_pos else None,
            )
            if worker.cooker_loc is None:
                return

        if needs_storage and worker.box_loc is None:
            all_exclude = exclude_boxes + [worker.counter_loc]
            if worker.cooker_loc:
                all_exclude.append(worker.cooker_loc)
            worker.box_loc = self.find_nearest_tile_of_types(
                controller,
                bx,
                by,
                ["BOX", "COUNTER"],
                exclude=all_exclude,
                avoid_positions=other_bot_positions,
            )
            if worker.box_loc is None:
                return

        cx, cy = worker.counter_loc
        kx, ky = worker.cooker_loc if worker.cooker_loc else (0, 0)
        box_x, box_y = worker.box_loc if worker.box_loc else (0, 0)

        if not worker.current_order:
            return

        if worker.state in [
            States.BUY_MEAT,
            States.BUY_PLATE,
            States.BUY_NOODLES,
        ] and bot_info.get("holding"):
            worker.state = States.TRASH_ITEM

        if worker.state is States.INIT:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if (
                FoodType.MEAT.food_name not in worker.current_order["required"]
                and FoodType.EGG.food_name not in worker.current_order["required"]
            ) or (tile and isinstance(tile.item, Pan)):
                if FoodType.ONIONS.food_name in worker.current_order["required"]:
                    worker.state = States.BUY_ONIONS
                elif FoodType.MEAT.food_name in worker.current_order["required"]:
                    worker.state = States.BUY_MEAT
                elif FoodType.EGG.food_name in worker.current_order["required"]:
                    worker.state = States.BUY_EGG
                else:
                    worker.state = States.BUY_PLATE
            else:
                worker.state = States.BUY_PAN

        elif worker.state == States.BUY_PAN:
            holding = bot_info.get("holding")
            if holding:  # assume it's the pan
                if self.move_towards(controller, bot_id, kx, ky, other_bot_positions):
                    if controller.place(bot_id, kx, ky):
                        worker.state = States.BUY_MEAT
            else:
                shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
                if not shop_pos:
                    return
                sx, sy = shop_pos
                if self.move_towards(controller, bot_id, sx, sy, other_bot_positions):
                    if (
                        controller.get_team_money(controller.get_team())
                        >= ShopCosts.PAN.buy_cost
                    ):
                        controller.buy(bot_id, ShopCosts.PAN, sx, sy)

        elif worker.state == States.BUY_MEAT:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if shop_pos is None:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy, other_bot_positions):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.MEAT.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.MEAT, sx, sy):
                        worker.state = States.PUT_MEAT_ON_COUNTER

        elif worker.state == States.PUT_MEAT_ON_COUNTER:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.place(bot_id, cx, cy):
                    worker.state = States.CHOP_MEAT

        elif worker.state == States.CHOP_MEAT:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.chop(bot_id, cx, cy):
                    worker.state = States.PICKUP_MEAT

        elif worker.state == States.PICKUP_MEAT:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.pickup(bot_id, cx, cy):
                    worker.state = States.PUT_MEAT_ON_COOKER

        elif worker.state == States.PUT_MEAT_ON_COOKER:
            if self.move_towards(controller, bot_id, kx, ky, other_bot_positions):
                # Using the NEW logic where place() starts cooking automatically
                if controller.place(bot_id, kx, ky):
                    worker.state = States.BUY_PLATE

        elif worker.state == States.BUY_EGG:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy, other_bot_positions):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.EGG.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.EGG, sx, sy):
                        worker.state = States.PUT_EGG_ON_COOKER

        elif worker.state == States.PUT_EGG_ON_COOKER:
            if self.move_towards(controller, bot_id, kx, ky, other_bot_positions):
                # Using the NEW logic where place() starts cooking automatically
                if controller.place(bot_id, kx, ky):
                    if FoodType.MEAT.food_name in worker.current_order["required"]:
                        worker.state = States.WAIT_FOR_EGG
                    elif worker.has_plate:
                        worker.state = States.WAIT_FOR_EGG
                    else:
                        worker.state = States.BUY_PLATE

        elif worker.state == States.BUY_ONIONS:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy, other_bot_positions):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.ONIONS.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.ONIONS, sx, sy):
                        worker.state = States.PUT_ONIONS_ON_COUNTER

        elif worker.state == States.PUT_ONIONS_ON_COUNTER:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.place(bot_id, cx, cy):
                    worker.state = States.CHOP_ONIONS

        elif worker.state == States.CHOP_ONIONS:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.chop(bot_id, cx, cy):
                    worker.state = States.PICKUP_CHOPPED_ONIONS

        elif worker.state == States.PICKUP_CHOPPED_ONIONS:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.pickup(bot_id, cx, cy):
                    worker.state = States.STORE_CHOPPED_ONIONS

        elif worker.state == States.STORE_CHOPPED_ONIONS:
            if self.move_towards(controller, bot_id, box_x, box_y, other_bot_positions):
                if controller.place(bot_id, box_x, box_y):
                    if FoodType.MEAT.food_name in worker.current_order["required"]:
                        worker.state = States.BUY_MEAT
                    elif FoodType.EGG.food_name in worker.current_order["required"]:
                        worker.state = States.BUY_EGG
                    else:
                        worker.state = States.BUY_PLATE

        elif worker.state == States.BUY_PLATE:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy, other_bot_positions):
                if (
                    controller.get_team_money(controller.get_team())
                    >= ShopCosts.PLATE.buy_cost
                ):
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        worker.state = States.PUT_PLATE_ON_COUNTER

        elif worker.state == States.PUT_PLATE_ON_COUNTER:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.place(bot_id, cx, cy):
                    worker.has_plate = True
                    if FoodType.MEAT.food_name in worker.current_order["required"]:
                        worker.state = States.WAIT_FOR_MEAT
                    elif FoodType.EGG.food_name in worker.current_order["required"]:
                        worker.state = States.WAIT_FOR_EGG
                    elif FoodType.ONIONS.food_name in worker.current_order["required"]:
                        worker.state = States.RETRIEVE_BOX_ITEM
                    elif FoodType.NOODLES.food_name in worker.current_order["required"]:
                        worker.state = States.BUY_NOODLES
                    else:  # FoodType.SAUCE.food_name in worker.current_order["required"]:
                        worker.state = States.BUY_SAUCE

        elif worker.state == States.BUY_NOODLES:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy, other_bot_positions):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.NOODLES.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
                        worker.state = States.ADD_NOODLES_TO_PLATE

        elif worker.state == States.ADD_NOODLES_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    tile = controller.get_tile(controller.get_team(), cx, cy)
                    need = remaining_needed(worker.current_order, plate_counter_from_tile(tile, controller))
                    if need.get(FoodType.NOODLES.food_name, 0) > 0:
                        worker.state = States.BUY_NOODLES
                    elif need.get(FoodType.SAUCE.food_name, 0) > 0:
                        worker.state = States.BUY_SAUCE
                    else:
                        worker.state = States.PICKUP_PLATE

        elif worker.state == States.BUY_SAUCE:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy, other_bot_positions):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.SAUCE.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.SAUCE, sx, sy):
                        worker.state = States.ADD_SAUCE_TO_PLATE

        elif worker.state == States.ADD_SAUCE_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):          
                if controller.add_food_to_plate(bot_id, cx, cy):
                    tile = controller.get_tile(controller.get_team(), cx, cy)
                    need = remaining_needed(worker.current_order, plate_counter_from_tile(tile, controller))
                    if need.get(FoodType.SAUCE.food_name, 0) > 0:
                        worker.state = States.BUY_SAUCE
                    else:
                        worker.state = States.PICKUP_PLATE

        elif worker.state == States.WAIT_FOR_MEAT:
            if self.move_towards(controller, bot_id, kx, ky, other_bot_positions):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:
                        if controller.take_from_pan(bot_id, kx, ky):
                            worker.state = States.ADD_MEAT_TO_PLATE
                    elif food.cooked_stage == 2:
                        # trash
                        if controller.take_from_pan(bot_id, kx, ky):
                            worker.state = States.TRASH_ITEM
                else:
                    if bot_info.get("holding"):
                        # trash
                        worker.state = States.TRASH_ITEM
                    else:
                        # restart
                        worker.state = States.BUY_MEAT

        elif worker.state == States.ADD_MEAT_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    tile = controller.get_tile(controller.get_team(), cx, cy)
                    need = remaining_needed(worker.current_order, plate_counter_from_tile(tile, controller))

                    if need.get(FoodType.MEAT.food_name, 0) > 0:
                        worker.state = States.BUY_MEAT
                    elif need.get(FoodType.EGG.food_name, 0) > 0:
                        worker.state = States.BUY_EGG
                    elif need.get(FoodType.ONIONS.food_name, 0) > 0:
                        worker.state = States.RETRIEVE_BOX_ITEM
                    elif need.get(FoodType.NOODLES.food_name, 0) > 0:
                        worker.state = States.BUY_NOODLES
                    elif need.get(FoodType.SAUCE.food_name, 0) > 0:
                        worker.state = States.BUY_SAUCE
                    else:
                        worker.state = States.PICKUP_PLATE

        elif worker.state == States.WAIT_FOR_EGG:
            if self.move_towards(controller, bot_id, kx, ky, other_bot_positions):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:
                        if controller.take_from_pan(bot_id, kx, ky):
                            worker.state = States.ADD_EGG_TO_PLATE
                    elif food.cooked_stage == 2:
                        # trash
                        if controller.take_from_pan(bot_id, kx, ky):
                            worker.state = States.TRASH_ITEM
                else:
                    if bot_info.get("holding"):
                        # trash
                        worker.state = States.TRASH_ITEM
                    else:
                        # restart
                        worker.state = States.BUY_EGG

        elif worker.state == States.ADD_EGG_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    tile = controller.get_tile(controller.get_team(), cx, cy)
                    need = remaining_needed(worker.current_order, plate_counter_from_tile(tile, controller))
                    
                    if need.get(FoodType.EGG.food_name, 0) > 0:
                        worker.state = States.BUY_EGG
                    elif need.get(FoodType.ONIONS.food_name, 0) > 0:
                        worker.state = States.RETRIEVE_BOX_ITEM
                    elif need.get(FoodType.MEAT.food_name, 0) > 0:
                        worker.state = States.BUY_MEAT  # or WAIT_FOR_MEAT depending on your flow
                    elif need.get(FoodType.NOODLES.food_name, 0) > 0:
                        worker.state = States.BUY_NOODLES
                    elif need.get(FoodType.SAUCE.food_name, 0) > 0:
                        worker.state = States.BUY_SAUCE
                    else:
                        worker.state = States.PICKUP_PLATE

        elif worker.state == States.RETRIEVE_BOX_ITEM:
            if self.move_towards(controller, bot_id, box_x, box_y, other_bot_positions):
                if controller.pickup(bot_id, box_x, box_y):
                    worker.state = States.PLATE_BOX_ITEM

        elif worker.state == States.PLATE_BOX_ITEM:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    tile = controller.get_tile(controller.get_team(), cx, cy)
                    need = remaining_needed(worker.current_order, plate_counter_from_tile(tile, controller))

                    if need.get(FoodType.ONIONS.food_name, 0) > 0:
                        worker.state = States.RETRIEVE_BOX_ITEM
                    elif need.get(FoodType.NOODLES.food_name, 0) > 0:
                        worker.state = States.BUY_NOODLES
                    elif need.get(FoodType.SAUCE.food_name, 0) > 0:
                        worker.state = States.BUY_SAUCE
                    else:
                        worker.state = States.PICKUP_PLATE

        elif worker.state == States.PICKUP_PLATE:
            if self.move_towards(controller, bot_id, cx, cy, other_bot_positions):
                if controller.pickup(bot_id, cx, cy):
                    worker.state = States.SUBMIT_ORDER

        elif worker.state == States.SUBMIT_ORDER:
            submit_pos = self.find_nearest_tile(controller, bx, by, "SUBMIT")
            worker.has_plate = False
            if not submit_pos:
                return
            ux, uy = submit_pos
            if controller.get_turn() > worker.current_order["expires_turn"]:
                # order expired, trash plate items then discard plate
                worker.state = States.TRASH_PLATE
                return
            elif self.move_towards(controller, bot_id, ux, uy, other_bot_positions):
                if controller.submit(bot_id, ux, uy):
                    worker.state = States.INIT
                    self.fulfilled_orders.add(worker.current_order["order_id"])
                    self.claimed_orders.discard(worker.current_order["order_id"])
                    worker.current_order = None
                    # Reset workstation locations for next order
                    worker.counter_loc = None
                    worker.cooker_loc = None
                    worker.box_loc = None

        elif worker.state == States.TRASH_ITEM:
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if not trash_pos:
                return
            tx, ty = trash_pos
            if self.move_towards(controller, bot_id, tx, ty, other_bot_positions):
                if controller.trash(bot_id, tx, ty):
                    worker.state = States.INIT
                    if worker.current_order:
                        self.claimed_orders.discard(worker.current_order["order_id"])
                    worker.current_order = None
                    # Reset workstation locations for next order
                    worker.counter_loc = None
                    worker.cooker_loc = None
                    worker.box_loc = None

        elif worker.state == States.TRASH_PLATE:
            # Trash the items on the plate, then put the empty plate on a counter
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if not trash_pos:
                return
            tx, ty = trash_pos
            if self.move_towards(controller, bot_id, tx, ty, other_bot_positions):
                if controller.trash(bot_id, tx, ty):
                    # After trashing items, discard the empty plate
                    worker.state = States.DISCARD_PLATE

        elif worker.state == States.DISCARD_PLATE:
            # Put the empty plate on a counter (can't trash empty plates)
            # Find any counter to place it on
            px, py = worker.counter_loc
            if self.move_towards(controller, bot_id, px, py, other_bot_positions):
                if controller.place(bot_id, px, py):
                    worker.state = States.INIT
                    if worker.current_order:
                        self.claimed_orders.discard(worker.current_order["order_id"])
                    worker.current_order = None
                    # Reset workstation locations for next order
                    worker.counter_loc = None
                    worker.cooker_loc = None
                    worker.box_loc = None

    def run_sabotage(self, controller: RobotController):
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return
        
        m = controller.get_map(controller.get_enemy_team())
        bot0_id = my_bots[0]
        bot0_state = controller.get_bot_state(bot0_id)
        if not bot0_state:
            return
        bot0_x, bot0_y = bot0_state["x"], bot0_state["y"]
        
        if self.sabotage_state == SabotageState.MOVE_TO_PAN:
            for mx in range(m.width):
                for my in range(m.height):
                    if isinstance(m.tiles[mx][my].item, Pan):
                        # Pick up the pan
                        if self.move_towards(controller, bot0_id, mx, my):
                            if controller.pickup(bot0_id, mx, my):
                                self.sabotage_state = SabotageState.MOVE_TO_TRASH
                                return 
                                
        elif self.sabotage_state == SabotageState.MOVE_TO_TRASH:
            trash = self.find_nearest_tile(controller, bot0_x, bot0_y, "TRASH")
            if trash:
                if self.move_towards(controller, bot0_id, trash[0], trash[1]):
                    if controller.trash(bot0_id, trash[0], trash[1]):
                        print(f"Bot 0 threw away food in pan at turn {controller.get_turn()}")
                        self.sabotage_state = SabotageState.PLACE_PAN_ON_COUNTER
                        return
                    
        elif self.sabotage_state == SabotageState.PLACE_PAN_ON_COUNTER:
            counter = self.find_nearest_tile(controller, bot0_x, bot0_y, "COUNTER")
            if counter:
                if self.move_towards(controller, bot0_id, counter[0], counter[1]):
                    if controller.place(bot0_id, counter[0], counter[1]):
                        print(f"Bot 0 placed pan on counter at turn {controller.get_turn()}")
                        self.sabotage_state = SabotageState.MOVE_TO_PLATE
                        return
                        
        elif self.sabotage_state == SabotageState.MOVE_TO_PLATE:
            for mx in range(m.width):
                for my in range(m.height):
                    if isinstance(m.tiles[mx][my].item, Plate):
                        # Pick up the plate
                        if self.move_towards(controller, bot0_id, mx, my):
                            if controller.pickup(bot0_id, mx, my):
                                self.sabotage_state = SabotageState.MOVE_TO_TRASH
                                return
                                
        elif self.sabotage_state == SabotageState.MOVE_TO_TRASH:
            trash = self.find_nearest_tile(controller, bot0_x, bot0_y, "TRASH")
            if trash:
                if self.move_towards(controller, bot0_id, trash[0], trash[1]):
                    if controller.trash(bot0_id, trash[0], trash[1]):
                        print(f"Bot 0 threw away plate at turn {controller.get_turn()}")
                        self.sabotage_state = SabotageState.MOVE_TO_PAN
                        return
                    
        
        

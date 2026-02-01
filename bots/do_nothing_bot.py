import random
from collections import deque
from enum import Enum
from typing import List, Optional, Set, Tuple

from game_constants import FoodType, ShopCosts, Team, TileType
from item import Food, Pan, Plate
from robot_controller import RobotController


class States(Enum):
    INIT = 0
    BUY_PAN = 1
    BUY_MEAT = 2
    PUT_MEAT_ON_COUNTER = 3
    CHOP_MEAT = 4
    PICKUP_MEAT = 5
    PUT_MEAT_ON_COOKER = 6
    BUY_ONIONS = 7
    PUT_ONIONS_ON_COUNTER = 8
    CHOP_ONIONS = 9
    PICKUP_CHOPPED_ONIONS = 20
    STORE_CHOPPED_ONIONS = 10
    BUY_PLATE = 11
    PUT_PLATE_ON_COUNTER = 12
    BUY_NOODLES = 13
    ADD_NOODLES_TO_PLATE = 14
    BUY_SAUCE = 23
    ADD_SAUCE_TO_PLATE = 24
    WAIT_FOR_MEAT = 15
    ADD_MEAT_TO_PLATE = 16
    PICKUP_PLATE = 17
    SUBMIT_ORDER = 18
    TRASH_ITEM = 19
    RETRIEVE_BOX_ITEM = 21
    PLATE_BOX_ITEM = 22


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.counter_loc: Optional[tuple[int, int]] = None
        self.cooker_loc: Optional[tuple[int, int]] = None
        self.box_loc: Optional[tuple[int, int]] = None
        self.my_bot_id = None
        self.fulfilled_orders: set[int] = set()

        self.current_order = None
        self.placed_plate = None

        self.state = States.INIT

    def get_bfs_path(
        self, controller: RobotController, start: Tuple[int, int], target_predicate
    ) -> Optional[Tuple[int, int]]:
        queue = deque([(start, [])])
        visited = set([start])
        w, h = self.map.width, self.map.height

        while queue:
            (curr_x, curr_y), path = queue.popleft()
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
                        if controller.get_map().is_tile_walkable(nx, ny):
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None

    def move_towards(
        self, controller: RobotController, bot_id: int, target_x: int, target_y: int
    ) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        bx, by = bot_state["x"], bot_state["y"]

        def is_adjacent_to_target(x, y, tile):
            return max(abs(x - target_x), abs(y - target_y)) <= 1

        if is_adjacent_to_target(bx, by, None):
            return True
        step = self.get_bfs_path(controller, (bx, by), is_adjacent_to_target)
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
            return False
        return False

    def find_nearest_tile(
        self, controller: RobotController, bot_x: int, bot_y: int, tile_name: str
    ) -> Optional[Tuple[int, int]]:
        best_dist = 9999
        best_pos = None
        m = controller.get_map()
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == tile_name:
                    dist = max(abs(bot_x - x), abs(bot_y - y))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (x, y)
        return best_pos

    def next_order(self, controller: RobotController) -> Optional[dict]:
        self.current_order = None

    def play_turn(self, controller: RobotController):
        return

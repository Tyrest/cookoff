import itertools
import random
from collections import deque
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

from game_constants import FoodType, ShopCosts, Team, TileType
from item import Food, Pan, Plate
from robot_controller import RobotController
from time import time


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
    RETRIEVE_BOX_ITEM = auto()
    PLATE_BOX_ITEM = auto()


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.counter_loc: Optional[tuple[int, int]] = None
        self.cooker_loc: Optional[tuple[int, int]] = None
        self.box_loc: Optional[tuple[int, int]] = None
        self.my_bot_id = None
        self.fulfilled_orders: set[int] = set()

        self.current_order = None
        self.cooker_timer = None

        self.state = States.INIT
        self.last_state = None

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
                        if self.map.is_tile_walkable(nx, ny):
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
        """Find nearest tile of given type using BFS (actual walking distance)."""
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
    ) -> Optional[Tuple[int, int]]:
        """Find nearest tile matching any of the given types using BFS, excluding specified positions."""
        exclude = exclude or []
        queue = deque([(bot_x, bot_y, 0)])
        visited = {(bot_x, bot_y)}
        w, h = self.map.width, self.map.height

        while queue:
            cx, cy, dist = queue.popleft()
            # Check adjacent tiles for the target types
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in exclude:
                        tile = self.map.tiles[nx][ny]
                        if tile.tile_name in tile_names:
                            return (nx, ny)
            # Expand to walkable neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if self.map.is_tile_walkable(nx, ny):
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))
        return None

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

    def select_next_order(self, controller: RobotController):
        orders = controller.get_orders(controller.get_team())
        best_order = None
        best_value = -float("inf")
        team_money = controller.get_team_money(controller.get_team())

        for order in orders:
            if order["order_id"] in self.fulfilled_orders:
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
            if value > best_value and team_money >= ingredient_cost:
                best_value = value
                best_order = order

        self.current_order = best_order
        if self.current_order:
            print(f"Selected order: {self.current_order}")

    def play_turn(self, controller: RobotController):
        orders = controller.get_orders(controller.get_team())

        if self.current_order is None:
            self.select_next_order(controller)
            if self.current_order is None:
                return

        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return

        self.my_bot_id = my_bots[0]

        self.one_counter(controller, my_bots)

    def one_counter(
        self,
        controller: RobotController,
        my_bots,
    ) -> None:
        if self.last_state != self.state:
            print(f"t={controller.get_turn()} Current state: {self.state}")
            self.last_state = self.state

        bot_id = self.my_bot_id
        if bot_id is None:
            return

        bot_info = controller.get_bot_state(bot_id)
        if bot_info is None:
            return
        bx, by = bot_info["x"], bot_info["y"]

        if self.counter_loc is None:
            self.counter_loc = self.find_nearest_tile(controller, bx, by, "COUNTER")
        if self.cooker_loc is None:
            self.cooker_loc = self.find_nearest_tile(controller, bx, by, "COOKER")
        if self.counter_loc is None or self.cooker_loc is None:
            return
        if self.box_loc is None:
            self.box_loc = self.find_nearest_tile_of_types(
                controller,
                bx,
                by,
                ["BOX", "COUNTER"],
                exclude=[self.counter_loc, self.cooker_loc],
            )
        # print(f"Counter loc: {self.counter_loc}, Cooker loc: {self.cooker_loc}, Box loc: {self.box_loc}")

        if self.box_loc is None:
            return

        cx, cy = self.counter_loc
        kx, ky = self.cooker_loc
        box_x, box_y = self.box_loc

        if not self.current_order:
            return

        if self.state in [
            States.BUY_MEAT,
            States.BUY_PLATE,
            States.BUY_NOODLES,
        ] and bot_info.get("holding"):
            self.state = States.TRASH_ITEM

        if self.state is States.INIT:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if (
                FoodType.MEAT.food_name not in self.current_order["required"]
                and FoodType.EGG.food_name not in self.current_order["required"]
            ) or (tile and isinstance(tile.item, Pan)):
                if FoodType.ONIONS.food_name in self.current_order["required"]:
                    self.state = States.BUY_ONIONS
                elif FoodType.MEAT.food_name in self.current_order["required"]:
                    self.state = States.BUY_MEAT
                elif FoodType.EGG.food_name in self.current_order["required"]:
                    self.state = States.BUY_EGG
                else:
                    self.state = States.BUY_PLATE
            else:
                self.state = States.BUY_PAN

        elif self.state == States.BUY_PAN:
            holding = bot_info.get("holding")
            if holding:  # assume it's the pan
                if self.move_towards(controller, bot_id, kx, ky):
                    if controller.place(bot_id, kx, ky):
                        self.state = States.BUY_MEAT
            else:
                shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
                if not shop_pos:
                    return
                sx, sy = shop_pos
                if self.move_towards(controller, bot_id, sx, sy):
                    if (
                        controller.get_team_money(controller.get_team())
                        >= ShopCosts.PAN.buy_cost
                    ):
                        controller.buy(bot_id, ShopCosts.PAN, sx, sy)

        elif self.state == States.BUY_MEAT:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.MEAT.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.MEAT, sx, sy):
                        self.state = States.PUT_MEAT_ON_COUNTER

        elif self.state == States.PUT_MEAT_ON_COUNTER:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.state = States.CHOP_MEAT

        elif self.state == States.CHOP_MEAT:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.chop(bot_id, cx, cy):
                    self.state = States.PICKUP_MEAT

        elif self.state == States.PICKUP_MEAT:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.state = States.PUT_MEAT_ON_COOKER

        elif self.state == States.PUT_MEAT_ON_COOKER:
            if self.move_towards(controller, bot_id, kx, ky):
                # Using the NEW logic where place() starts cooking automatically
                if controller.place(bot_id, kx, ky):
                    self.state = States.BUY_PLATE

        elif self.state == States.BUY_EGG:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.EGG.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.EGG, sx, sy):
                        self.state = States.PUT_EGG_ON_COOKER

        elif self.state == States.PUT_EGG_ON_COOKER:
            if self.move_towards(controller, bot_id, kx, ky):
                # Using the NEW logic where place() starts cooking automatically
                if controller.place(bot_id, kx, ky):
                    if FoodType.MEAT.food_name in self.current_order["required"]:
                        self.state = States.WAIT_FOR_EGG
                    else:
                        self.state = States.BUY_PLATE

        elif self.state == States.BUY_ONIONS:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.ONIONS.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.ONIONS, sx, sy):
                        self.state = States.PUT_ONIONS_ON_COUNTER

        elif self.state == States.PUT_ONIONS_ON_COUNTER:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.state = States.CHOP_ONIONS

        elif self.state == States.CHOP_ONIONS:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.chop(bot_id, cx, cy):
                    self.state = States.PICKUP_CHOPPED_ONIONS

        elif self.state == States.PICKUP_CHOPPED_ONIONS:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.state = States.STORE_CHOPPED_ONIONS

        elif self.state == States.STORE_CHOPPED_ONIONS:
            if self.move_towards(controller, bot_id, box_x, box_y):
                if controller.place(bot_id, box_x, box_y):
                    if FoodType.MEAT.food_name in self.current_order["required"]:
                        self.state = States.BUY_MEAT
                    elif FoodType.EGG.food_name in self.current_order["required"]:
                        self.state = States.BUY_EGG
                    else:
                        self.state = States.BUY_PLATE

        elif self.state == States.BUY_PLATE:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if (
                    controller.get_team_money(controller.get_team())
                    >= ShopCosts.PLATE.buy_cost
                ):
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        self.state = States.PUT_PLATE_ON_COUNTER

        elif self.state == States.PUT_PLATE_ON_COUNTER:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    if FoodType.MEAT.food_name in self.current_order["required"]:
                        self.state = States.WAIT_FOR_MEAT
                    elif FoodType.EGG.food_name in self.current_order["required"]:
                        self.state = States.WAIT_FOR_EGG
                    elif FoodType.ONIONS.food_name in self.current_order["required"]:
                        self.state = States.RETRIEVE_BOX_ITEM
                    elif FoodType.NOODLES.food_name in self.current_order["required"]:
                        self.state = States.BUY_NOODLES
                    else:  # FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE

        elif self.state == States.BUY_NOODLES:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.NOODLES.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
                        self.state = States.ADD_NOODLES_TO_PLATE

        elif self.state == States.ADD_NOODLES_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    if FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE
                    else:
                        self.state = States.PICKUP_PLATE

        elif self.state == States.BUY_SAUCE:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if (
                    controller.get_team_money(controller.get_team())
                    >= FoodType.SAUCE.buy_cost
                ):
                    if controller.buy(bot_id, FoodType.SAUCE, sx, sy):
                        self.state = States.ADD_SAUCE_TO_PLATE

        elif self.state == States.ADD_SAUCE_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.state = States.PICKUP_PLATE

        elif self.state == States.WAIT_FOR_MEAT:
            if self.move_towards(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = States.ADD_MEAT_TO_PLATE
                    elif food.cooked_stage == 2:
                        # trash
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = States.TRASH_ITEM
                else:
                    if bot_info.get("holding"):
                        # trash
                        self.state = States.TRASH_ITEM
                    else:
                        # restart
                        self.state = States.BUY_MEAT

        elif self.state == States.ADD_MEAT_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    if FoodType.EGG.food_name in self.current_order["required"]:
                        self.state = States.BUY_EGG
                    elif FoodType.ONIONS.food_name in self.current_order["required"]:
                        self.state = States.RETRIEVE_BOX_ITEM
                    elif FoodType.NOODLES.food_name in self.current_order["required"]:
                        self.state = States.BUY_NOODLES
                    elif FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE
                    else:
                        self.state = States.PICKUP_PLATE

        elif self.state == States.WAIT_FOR_EGG:
            if self.move_towards(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(tile.item, Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = States.ADD_EGG_TO_PLATE
                    elif food.cooked_stage == 2:
                        # trash
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = States.TRASH_ITEM
                else:
                    if bot_info.get("holding"):
                        # trash
                        self.state = States.TRASH_ITEM
                    else:
                        # restart
                        self.state = States.BUY_EGG

        elif self.state == States.ADD_EGG_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    if FoodType.ONIONS.food_name in self.current_order["required"]:
                        self.state = States.RETRIEVE_BOX_ITEM
                    elif FoodType.NOODLES.food_name in self.current_order["required"]:
                        self.state = States.BUY_NOODLES
                    elif FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE
                    else:
                        self.state = States.PICKUP_PLATE

        elif self.state == States.RETRIEVE_BOX_ITEM:
            if self.move_towards(controller, bot_id, box_x, box_y):
                if controller.pickup(bot_id, box_x, box_y):
                    self.state = States.PLATE_BOX_ITEM

        elif self.state == States.PLATE_BOX_ITEM:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    if FoodType.NOODLES.food_name in self.current_order["required"]:
                        self.state = States.BUY_NOODLES
                    elif FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE
                    else:
                        self.state = States.PICKUP_PLATE

        elif self.state == States.PICKUP_PLATE:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.state = States.SUBMIT_ORDER

        elif self.state == States.SUBMIT_ORDER:
            submit_pos = self.find_nearest_tile(controller, bx, by, "SUBMIT")
            ux, uy = submit_pos
            if controller.get_turn() > self.current_order["expires_turn"]:
                # order expired, trash plate
                self.state = States.TRASH_ITEM
                return
            elif self.move_towards(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    self.state = States.INIT
                    self.fulfilled_orders.add(self.current_order["order_id"])
                    self.current_order = None

        elif self.state == States.TRASH_ITEM:
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if not trash_pos:
                return
            tx, ty = trash_pos
            if self.move_towards(controller, bot_id, tx, ty):
                if controller.trash(bot_id, tx, ty):
                    self.state = States.INIT
                    self.current_order = None

        for i in range(1, len(my_bots)):
            self.my_bot_id = my_bots[i]
            bot_id = self.my_bot_id

            bot_info = controller.get_bot_state(bot_id)
            if not bot_info:
                continue
            bx, by = bot_info["x"], bot_info["y"]

            possible = []
            for dx, dy in list(itertools.product([0, -1, 1], [0, -1, 1])):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = bx + dx, by + dy
                if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                    possible.append((dx, dy))

            if not possible:
                continue
            dx, dy = random.choice(possible)
            nx, ny = bx + dx, by + dy
            if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                controller.move(bot_id, dx, dy)
                continue

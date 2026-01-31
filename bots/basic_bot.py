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
    BUY_EGG = 7
    PUT_EGG_ON_COOKER = 8
    BUY_ONIONS = 9
    PUT_ONIONS_ON_COUNTER = 10
    CHOP_ONIONS = 11
    PICKUP_CHOPPED_ONIONS = 12
    STORE_CHOPPED_ONIONS = 13
    BUY_PLATE = 14
    PUT_PLATE_ON_COUNTER = 15
    BUY_NOODLES = 16
    ADD_NOODLES_TO_PLATE = 17
    BUY_SAUCE = 18
    ADD_SAUCE_TO_PLATE = 19
    WAIT_FOR_MEAT = 20
    ADD_MEAT_TO_PLATE = 21
    PICKUP_PLATE = 22
    SUBMIT_ORDER = 23
    TRASH_ITEM = 24
    RETRIEVE_BOX_ITEM = 25
    PLATE_BOX_ITEM = 26


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
                        if controller.get_map(controller.get_team()).is_tile_walkable(
                            nx, ny
                        ):
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
        m = controller.get_map(controller.get_team())
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
        orders = controller.get_orders(controller.get_team())

        if not self.current_order:
            for order in orders:
                if (
                    order["order_id"] not in self.fulfilled_orders
                    and FoodType.EGG.food_name not in order["required"]
                ):
                    self.current_order = order
                    print(f"Current order: {self.current_order}")
                    break
            else:
                return  # No available orders

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
        print(f"Current state: {self.state}")

        bot_id = self.my_bot_id
        if not bot_id:
            return

        bot_info = controller.get_bot_state(bot_id)
        if not bot_info:
            return
        bx, by = bot_info["x"], bot_info["y"]

        if self.counter_loc is None:
            self.counter_loc = self.find_nearest_tile(controller, bx, by, "COUNTER")
        if self.cooker_loc is None:
            self.cooker_loc = self.find_nearest_tile(controller, bx, by, "COOKER")
        if self.box_loc is None:
            self.box_loc = self.find_nearest_tile(controller, bx, by, "BOX")

        if not self.counter_loc or not self.cooker_loc or not self.box_loc:
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
                if FoodType.MEAT.food_name in self.current_order["required"]:
                    print("Going to buy meat")
                    self.state = States.BUY_MEAT
                elif FoodType.EGG.food_name in self.current_order["required"]:
                    print("Going to buy egg")
                    self.state = States.BUY_EGG
                elif FoodType.ONIONS.food_name in self.current_order["required"]:
                    print("Going to buy onions")
                    self.state = States.BUY_ONIONS
                else:
                    print("Going to buy plate")
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
                    if FoodType.ONIONS.food_name in self.current_order["required"]:
                        print("Going to buy onions")
                        self.state = States.BUY_ONIONS
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
                    print("Chopped onions stored in box")
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
                    if FoodType.NOODLES.food_name in self.current_order["required"]:
                        self.state = States.BUY_NOODLES
                    elif FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE
                    else:
                        self.state = States.PICKUP_PLATE

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

        # state 11: add noodles to plate
        elif self.state == States.ADD_NOODLES_TO_PLATE:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    if FoodType.MEAT.food_name in self.current_order["required"]:
                        self.state = States.WAIT_FOR_MEAT
                    elif FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE
                    elif FoodType.ONIONS.food_name in self.current_order["required"]:
                        self.state = States.RETRIEVE_BOX_ITEM
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
                    if FoodType.ONIONS.food_name in self.current_order["required"]:
                        self.state = States.RETRIEVE_BOX_ITEM
                    else:
                        self.state = States.PICKUP_PLATE

        # state 12: wait and take meat
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
                    if FoodType.SAUCE.food_name in self.current_order["required"]:
                        self.state = States.BUY_SAUCE
                    elif FoodType.ONIONS.food_name in self.current_order["required"]:
                        self.state = States.RETRIEVE_BOX_ITEM
                    else:
                        self.state = States.PICKUP_PLATE

        elif self.state == States.RETRIEVE_BOX_ITEM:
            if self.move_towards(controller, bot_id, box_x, box_y):
                if controller.pickup(bot_id, box_x, box_y):
                    self.state = States.PLATE_BOX_ITEM

        elif self.state == States.PLATE_BOX_ITEM:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.state = States.PICKUP_PLATE

        elif self.state == States.PICKUP_PLATE:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.state = States.SUBMIT_ORDER

        # state 15: submit
        elif self.state == States.SUBMIT_ORDER:
            submit_pos = self.find_nearest_tile(controller, bx, by, "SUBMIT")
            ux, uy = submit_pos
            if self.move_towards(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    self.state = States.INIT
                    self.fulfilled_orders.add(self.current_order["order_id"])
                    self.current_order = None

        # state 16: trash
        elif self.state == States.TRASH_ITEM:
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if not trash_pos:
                return
            tx, ty = trash_pos
            if self.move_towards(controller, bot_id, tx, ty):
                if controller.trash(bot_id, tx, ty):
                    self.state = States.BUY_MEAT  # restart
        for i in range(1, len(my_bots)):
            self.my_bot_id = my_bots[i]
            bot_id = self.my_bot_id

            bot_info = controller.get_bot_state(bot_id)
            if not bot_info:
                return
            bx, by = bot_info["x"], bot_info["y"]

            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            nx, ny = bx + dx, by + dy
            if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                controller.move(bot_id, dx, dy)
                return

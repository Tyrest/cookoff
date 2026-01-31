"""
Map Generator for Cookoff Game

Generates random playable maps with configurable parameters including:
- Map dimensions (fixed or random ranges)
- Tile placement (counters, cookers, sinks, etc.)
- Bot spawn points
- Switch configuration
- Order generation
- Batch generation of multiple maps

Usage:
    # Single map generation
    python map_generator.py [output_file] [--width W] [--height H] [--seed S]
    
    # Batch generation of multiple maps with random dimensions
    python map_generator.py --batch N [--output-dir DIR] [--min-width W] [--max-width W]
    
Examples:
    python map_generator.py map2.txt --width 20 --height 10 --seed 42
    python map_generator.py --batch 5                          # Generate 5 random maps
    python map_generator.py --batch 10 --output-dir maps/      # Generate 10 maps in maps/
    python map_generator.py --batch 3 --symmetric --seed 123   # 3 symmetric maps with seed
"""

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from enum import Enum


class TileChar(Enum):
    """Map tile characters matching the game's expected format."""
    FLOOR = '.'
    WALL = '#'
    COUNTER = 'C'
    COOKER = 'K'
    SINK = 'S'
    SINKTABLE = 'T'
    TRASH = 'R'
    SUBMIT = 'U'
    SHOP = '$'
    BOX = 'B'
    SPAWN = 'b'


# Available food types for orders
FOOD_TYPES = ['EGG', 'ONIONS', 'MEAT', 'NOODLES', 'SAUCE']


@dataclass
class MapConfig:
    """Configuration for map generation."""
    width: int = 16
    height: int = 8
    
    # Tile counts (minimum requirements)
    num_counters: int = 3
    num_cookers: int = 2
    num_sinks: int = 1
    num_sinktables: int = 1
    num_trash: int = 1
    num_submit: int = 1
    num_shops: int = 1
    num_boxes: int = 1
    num_spawns: int = 1
    
    # Order configuration
    num_orders: int = 5
    order_min_duration: int = 100
    order_max_duration: int = 300
    order_min_reward: int = 50
    order_max_reward: int = 200
    order_min_penalty: int = 1
    order_max_penalty: int = 10
    order_min_ingredients: int = 1
    order_max_ingredients: int = 3
    
    # Switch configuration
    switch_turn: int = 250
    switch_duration: int = 100
    
    # Total game turns (for order scheduling)
    total_turns: int = 500


class MapGenerator:
    """Generates random playable maps for the Cookoff game."""
    
    def __init__(self, config: MapConfig, seed: Optional[int] = None):
        self.config = config
        if seed is not None:
            random.seed(seed)
        
        # Initialize empty map grid
        self.grid: List[List[str]] = []
        self.available_positions: Set[Tuple[int, int]] = set()
        
    def generate(self) -> str:
        """Generate a complete map string."""
        self._initialize_grid()
        self._place_walls()
        self._place_required_tiles()
        self._place_spawn_points()
        
        # Build the complete map string
        map_str = self._grid_to_string()
        switch_str = self._generate_switch_config()
        orders_str = self._generate_orders()
        
        return f"{map_str}\n{switch_str}\n{orders_str}"
    
    def _initialize_grid(self):
        """Initialize grid with floor tiles."""
        self.grid = [
            [TileChar.FLOOR.value for _ in range(self.config.width)]
            for _ in range(self.config.height)
        ]
        
        # Track available interior positions (excluding border)
        self.available_positions = set()
        for y in range(1, self.config.height - 1):
            for x in range(1, self.config.width - 1):
                self.available_positions.add((x, y))
    
    def _place_walls(self):
        """Place border walls around the map."""
        # Top and bottom walls
        for x in range(self.config.width):
            self.grid[0][x] = TileChar.WALL.value
            self.grid[self.config.height - 1][x] = TileChar.WALL.value
        
        # Left and right walls
        for y in range(self.config.height):
            self.grid[y][0] = TileChar.WALL.value
            self.grid[y][self.config.width - 1] = TileChar.WALL.value
    
    def _get_edge_adjacent_positions(self) -> List[Tuple[int, int]]:
        """Get interior positions adjacent to walls (good for stations)."""
        edge_positions = []
        for x, y in self.available_positions:
            # Check if adjacent to a wall
            is_edge = (
                x == 1 or x == self.config.width - 2 or
                y == 1 or y == self.config.height - 2
            )
            if is_edge:
                edge_positions.append((x, y))
        return edge_positions
    
    def _get_interior_positions(self) -> List[Tuple[int, int]]:
        """Get positions not adjacent to walls (good for counters/walkways)."""
        interior = []
        for x, y in self.available_positions:
            if (x > 1 and x < self.config.width - 2 and
                y > 1 and y < self.config.height - 2):
                interior.append((x, y))
        return interior
    
    def _place_tile(self, tile_char: str, prefer_edge: bool = True) -> Optional[Tuple[int, int]]:
        """Place a tile at a random available position."""
        if not self.available_positions:
            return None
        
        # Choose position based on preference
        if prefer_edge:
            candidates = self._get_edge_adjacent_positions()
            if not candidates:
                candidates = list(self.available_positions)
        else:
            candidates = list(self.available_positions)
        
        if not candidates:
            return None
            
        pos = random.choice(candidates)
        x, y = pos
        self.grid[y][x] = tile_char
        self.available_positions.discard(pos)
        return pos
    
    def _place_tiles_in_cluster(self, tile_char: str, count: int, 
                                 start_pos: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Place multiple tiles in a cluster for better gameplay."""
        placed = []
        
        if start_pos is None:
            # Pick a random starting position
            if not self.available_positions:
                return placed
            start_pos = random.choice(list(self.available_positions))
        
        # BFS-style placement for clustering
        to_try = [start_pos]
        tried = set()
        
        while len(placed) < count and to_try:
            pos = to_try.pop(0)
            if pos in tried or pos not in self.available_positions:
                continue
            tried.add(pos)
            
            x, y = pos
            self.grid[y][x] = tile_char
            self.available_positions.discard(pos)
            placed.append(pos)
            
            # Add neighbors to try
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if neighbor in self.available_positions and neighbor not in tried:
                    to_try.append(neighbor)
        
        return placed
    
    def _place_required_tiles(self):
        """Place all required game tiles."""
        # Place stations along edges (more realistic kitchen layout)
        # Shop and Submit should be accessible
        self._place_tile(TileChar.SHOP.value, prefer_edge=True)
        self._place_tile(TileChar.SUBMIT.value, prefer_edge=True)
        
        # Place cooking stations
        for _ in range(self.config.num_cookers):
            self._place_tile(TileChar.COOKER.value, prefer_edge=True)
        
        # Place sink and sinktable near each other
        sink_pos = self._place_tile(TileChar.SINK.value, prefer_edge=True)
        if sink_pos:
            # Try to place sinktable adjacent to sink
            x, y = sink_pos
            placed_sinktable = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.available_positions:
                    self.grid[ny][nx] = TileChar.SINKTABLE.value
                    self.available_positions.discard((nx, ny))
                    placed_sinktable = True
                    break
            if not placed_sinktable:
                self._place_tile(TileChar.SINKTABLE.value, prefer_edge=True)
        else:
            self._place_tile(TileChar.SINKTABLE.value, prefer_edge=True)
        
        # Place trash
        for _ in range(self.config.num_trash):
            self._place_tile(TileChar.TRASH.value, prefer_edge=True)
        
        # Place boxes
        for _ in range(self.config.num_boxes):
            self._place_tile(TileChar.BOX.value, prefer_edge=True)
        
        # Place counters (can be scattered or clustered)
        self._place_tiles_in_cluster(TileChar.COUNTER.value, self.config.num_counters)
    
    def _place_spawn_points(self):
        """Place bot spawn points in accessible locations."""
        # Get interior positions for spawns (not on edges)
        interior = self._get_interior_positions()
        
        for _ in range(self.config.num_spawns):
            if interior:
                pos = random.choice(interior)
                x, y = pos
                self.grid[y][x] = TileChar.SPAWN.value
                self.available_positions.discard(pos)
                interior.remove(pos)
            else:
                # Fallback to any available position
                self._place_tile(TileChar.SPAWN.value, prefer_edge=False)
    
    def _grid_to_string(self) -> str:
        """Convert the grid to a string representation."""
        return '\n'.join(''.join(row) for row in self.grid)
    
    def _generate_switch_config(self) -> str:
        """Generate the SWITCH configuration line."""
        return f"SWITCH: turn={self.config.switch_turn} duration={self.config.switch_duration}"
    
    def _generate_orders(self) -> str:
        """Generate random orders for the game."""
        orders = ["ORDERS:"]
        
        # Distribute orders across the game duration
        time_per_order = self.config.total_turns // (self.config.num_orders + 1)
        
        for i in range(self.config.num_orders):
            # Calculate start time with some randomization
            base_start = i * time_per_order
            start = max(0, base_start + random.randint(-time_per_order // 4, time_per_order // 4))
            
            # Random duration
            duration = random.randint(
                self.config.order_min_duration,
                self.config.order_max_duration
            )
            
            # Random ingredients (1-3 ingredients per order)
            num_ingredients = random.randint(
                self.config.order_min_ingredients,
                self.config.order_max_ingredients
            )
            ingredients = random.sample(FOOD_TYPES, min(num_ingredients, len(FOOD_TYPES)))
            required = ','.join(ingredients)
            
            # Random reward and penalty (scale with ingredients)
            base_reward = random.randint(
                self.config.order_min_reward,
                self.config.order_max_reward
            )
            reward = base_reward * num_ingredients
            
            penalty = random.randint(
                self.config.order_min_penalty,
                self.config.order_max_penalty
            )
            
            order_line = f"start={start}  duration={duration}  required={required}  reward={reward}  penalty={penalty}"
            orders.append(order_line)
        
        return '\n'.join(orders)


def create_symmetric_map(config: MapConfig, seed: Optional[int] = None) -> str:
    """
    Generate a horizontally symmetric map for fair competitive play.
    Both teams get mirrored layouts.
    """
    if seed is not None:
        random.seed(seed)
    
    half_width = config.width // 2
    grid = [[TileChar.FLOOR.value for _ in range(config.width)] for _ in range(config.height)]
    
    # Place border walls
    for x in range(config.width):
        grid[0][x] = TileChar.WALL.value
        grid[config.height - 1][x] = TileChar.WALL.value
    for y in range(config.height):
        grid[y][0] = TileChar.WALL.value
        grid[y][config.width - 1] = TileChar.WALL.value
    
    # Available positions on left half
    left_positions = set()
    for y in range(1, config.height - 1):
        for x in range(1, half_width):
            left_positions.add((x, y))
    
    def place_symmetric(tile_char: str, positions: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Place a tile and its mirror."""
        if not positions:
            return None
        pos = random.choice(list(positions))
        x, y = pos
        mirror_x = config.width - 1 - x
        
        grid[y][x] = tile_char
        grid[y][mirror_x] = tile_char
        positions.discard(pos)
        positions.discard((mirror_x, y))
        return pos
    
    # Place required tiles symmetrically
    tiles_to_place = [
        (TileChar.SHOP.value, 1),
        (TileChar.SUBMIT.value, 1),
        (TileChar.COOKER.value, config.num_cookers),
        (TileChar.SINK.value, config.num_sinks),
        (TileChar.SINKTABLE.value, config.num_sinktables),
        (TileChar.TRASH.value, config.num_trash),
        (TileChar.BOX.value, config.num_boxes),
        (TileChar.COUNTER.value, config.num_counters),
    ]
    
    for tile_char, count in tiles_to_place:
        for _ in range(count):
            place_symmetric(tile_char, left_positions)
    
    # Place spawn point(s) in center area
    center_positions = [(x, y) for x, y in left_positions 
                        if half_width - 3 <= x <= half_width]
    if center_positions:
        for _ in range(config.num_spawns):
            if center_positions:
                pos = random.choice(center_positions)
                x, y = pos
                grid[y][x] = TileChar.SPAWN.value
                left_positions.discard(pos)
                center_positions.remove(pos)
    
    # Build output
    map_str = '\n'.join(''.join(row) for row in grid)
    switch_str = f"SWITCH: turn={config.switch_turn} duration={config.switch_duration}"
    
    # Generate orders
    generator = MapGenerator(config, seed)
    orders_str = generator._generate_orders()
    
    return f"{map_str}\n{switch_str}\n{orders_str}"


def generate_simple_map(width: int = 16, height: int = 8, seed: Optional[int] = None) -> str:
    """
    Quick function to generate a simple playable map.
    
    Args:
        width: Map width (default 16)
        height: Map height (default 8)
        seed: Random seed for reproducibility
    
    Returns:
        Complete map string ready to save to file
    """
    config = MapConfig(width=width, height=height)
    generator = MapGenerator(config, seed)
    return generator.generate()


def generate_batch(
    count: int,
    output_dir: str = '.',
    min_width: int = 12,
    max_width: int = 24,
    min_height: int = 6,
    max_height: int = 14,
    seed: Optional[int] = None,
    symmetric: bool = False,
    prefix: str = 'map'
) -> List[str]:
    """
    Generate multiple random maps with random dimensions.
    
    Args:
        count: Number of maps to generate
        output_dir: Directory to save maps (default: current directory)
        min_width: Minimum map width (default: 12)
        max_width: Maximum map width (default: 24)
        min_height: Minimum map height (default: 6)
        max_height: Maximum map height (default: 14)
        seed: Random seed for reproducibility
        symmetric: Whether to generate symmetric maps
        prefix: Filename prefix (default: 'map')
    
    Returns:
        List of generated file paths
    """
    import os
    
    if seed is not None:
        random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # Find the next available map number
    existing_nums = set()
    for f in os.listdir(output_dir):
        if f.startswith(prefix) and f.endswith('.txt'):
            try:
                num = int(f[len(prefix):-4])
                existing_nums.add(num)
            except ValueError:
                pass
    
    next_num = max(existing_nums, default=0) + 1
    
    for i in range(count):
        # Random dimensions
        width = random.randint(min_width, max_width)
        height = random.randint(min_height, max_height)
        
        # Scale tile counts based on map size
        area = width * height
        base_area = 16 * 8  # Reference area
        scale = area / base_area
        
        config = MapConfig(
            width=width,
            height=height,
            num_counters=max(2, int(3 * scale)),
            num_cookers=max(1, int(2 * scale)),
            num_sinks=max(1, int(1 * scale)),
            num_sinktables=max(1, int(1 * scale)),
            num_trash=max(1, int(1 * scale)),
            num_boxes=max(1, int(1 * scale)),
            num_spawns=max(1, int(1 * scale)),
            num_orders=random.randint(3, 8),
        )
        
        # Generate map
        if symmetric:
            map_content = create_symmetric_map(config)
        else:
            generator = MapGenerator(config)
            map_content = generator.generate()
        
        # Save to file
        filename = f"{prefix}{next_num + i}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(map_content)
        
        generated_files.append(filepath)
        print(f"Generated: {filename} ({width}x{height})")
    
    return generated_files


def main():
    """Command-line interface for map generation."""
    parser = argparse.ArgumentParser(
        description='Generate random maps for the Cookoff game',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python map_generator.py map2.txt
  python map_generator.py map2.txt --width 20 --height 10
  python map_generator.py map2.txt --seed 42 --symmetric
  python map_generator.py --preview  # Preview without saving
  python map_generator.py --batch 5  # Generate 5 random maps
  python map_generator.py --batch 10 --min-width 15 --max-height 12
        """
    )
    
    parser.add_argument('output', nargs='?', default=None,
                        help='Output file path (default: generated_map.txt)')
    parser.add_argument('--width', type=int, default=16,
                        help='Map width (default: 16)')
    parser.add_argument('--height', type=int, default=8,
                        help='Map height (default: 8)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--symmetric', action='store_true',
                        help='Generate a horizontally symmetric map')
    parser.add_argument('--preview', action='store_true',
                        help='Print map to stdout without saving')
    
    # Tile count options
    parser.add_argument('--counters', type=int, default=3,
                        help='Number of counters (default: 3)')
    parser.add_argument('--cookers', type=int, default=2,
                        help='Number of cookers (default: 2)')
    parser.add_argument('--spawns', type=int, default=1,
                        help='Number of spawn points (default: 1)')
    
    # Order options
    parser.add_argument('--orders', type=int, default=5,
                        help='Number of orders to generate (default: 5)')
    
    # Batch generation options
    parser.add_argument('--batch', type=int, default=None,
                        help='Generate multiple maps with random dimensions')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for batch generation (default: current dir)')
    parser.add_argument('--min-width', type=int, default=12,
                        help='Minimum width for batch generation (default: 12)')
    parser.add_argument('--max-width', type=int, default=24,
                        help='Maximum width for batch generation (default: 24)')
    parser.add_argument('--min-height', type=int, default=6,
                        help='Minimum height for batch generation (default: 6)')
    parser.add_argument('--max-height', type=int, default=14,
                        help='Maximum height for batch generation (default: 14)')
    parser.add_argument('--prefix', type=str, default='map',
                        help='Filename prefix for batch generation (default: map)')
    
    args = parser.parse_args()
    
    # Batch generation mode
    if args.batch is not None:
        files = generate_batch(
            count=args.batch,
            output_dir=args.output_dir,
            min_width=args.min_width,
            max_width=args.max_width,
            min_height=args.min_height,
            max_height=args.max_height,
            seed=args.seed,
            symmetric=args.symmetric,
            prefix=args.prefix,
        )
        print(f"\nGenerated {len(files)} maps in '{args.output_dir}'")
        return
    
    # Single map generation mode
    output = args.output or 'generated_map.txt'
    
    # Build configuration
    config = MapConfig(
        width=args.width,
        height=args.height,
        num_counters=args.counters,
        num_cookers=args.cookers,
        num_spawns=args.spawns,
        num_orders=args.orders,
    )
    
    # Generate map
    if args.symmetric:
        map_content = create_symmetric_map(config, args.seed)
    else:
        generator = MapGenerator(config, args.seed)
        map_content = generator.generate()
    
    # Output
    if args.preview:
        print(map_content)
        print(f"\n--- Preview mode: not saved ---")
    else:
        with open(output, 'w') as f:
            f.write(map_content)
        print(f"Map generated and saved to: {output}")
        print(f"Dimensions: {args.width}x{args.height}")
        if args.seed is not None:
            print(f"Seed: {args.seed}")


if __name__ == '__main__':
    main()

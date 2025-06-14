import pygame
import math
import heapq
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import copy

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
HEX_RADIUS = 30
GRID_WIDTH = 10
GRID_HEIGHT = 6

# Colors
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (128, 128, 128),
    'DARK_GRAY': (64, 64, 64),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'YELLOW': (255, 255, 0),
    'PURPLE': (128, 0, 128),
    'ORANGE': (255, 165, 0),
    'CYAN': (0, 255, 255),
    'PINK': (255, 192, 203),
    'BROWN': (139, 69, 19),
    'LIGHT_BLUE': (173, 216, 230),
    'LIGHT_GREEN': (144, 238, 144)
}

class CellType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    TRAP1 = 2  # Increase gravity (double energy)
    TRAP2 = 3  # Decrease speed (double steps)
    TRAP3 = 4  # Move 2 cells in last direction
    TRAP4 = 5  # Remove uncollected treasures
    REWARD1 = 6  # Decrease gravity (half energy)
    REWARD2 = 7  # Increase speed (half steps)
    TREASURE = 8
    ENTRY = 9

@dataclass
class GameState:
    position: Tuple[int, int]   # Current location
    collected_treasures: Set[Tuple[int, int]] # Treasures gathered
    collected_rewards: Set[Tuple[int, int]] # Rewards gathered
    energy_multiplier: float # Current energy cost modifier
    speed_multiplier: float # Current speed modifier
    last_direction: Optional[Tuple[int, int]] # For TRAP3 effect
    total_energy: float # Cumulative energy spent
    total_steps: int # Cumulative steps taken
    treasures_removed: bool

    def __lt__(self, other):
        return (self.total_energy + self.total_steps) < (other.total_energy + other.total_steps)

class OptimizedTreasureHuntSolver:
    def __init__(self, grid):
        self.grid = grid
        self.all_treasures = copy.deepcopy(grid.treasures)
        self.all_rewards = copy.deepcopy(grid.rewards)
        
    def hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate hexagonal distance between two positions"""
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Convert to cube coordinates for accurate hex distance
        def hex_to_cube(row, col):
            x = col
            z = row - (col - (col & 1)) // 2
            y = -x - z
            return x, y, z
        
        x1, y1, z1 = hex_to_cube(r1, c1)
        x2, y2, z2 = hex_to_cube(r2, c2)
        
        return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2
    
    def apply_trap_effect(self, state: GameState, trap_type: CellType) -> GameState:
        """Apply the effect of stepping on a trap"""
        new_state = copy.deepcopy(state)
        
        if trap_type == CellType.TRAP1:  # Increase gravity
            new_state.energy_multiplier *= 2
        elif trap_type == CellType.TRAP2:  # Decrease speed
            new_state.speed_multiplier *= 2
        elif trap_type == CellType.TRAP3:  # Move 2 cells in last direction
            if new_state.last_direction:
                dr, dc = new_state.last_direction
                new_row = new_state.position[0] + 2 * dr
                new_col = new_state.position[1] + 2 * dc
                # Check bounds and obstacles
                if (0 <= new_row < self.grid.height and 
                    0 <= new_col < self.grid.width and 
                    self.grid.get_cell(new_row, new_col) != CellType.OBSTACLE):
                    new_state.position = (new_row, new_col)
        elif trap_type == CellType.TRAP4:  # Remove uncollected treasures
            new_state.treasures_removed = True
            
        return new_state
    
    def apply_reward_effect(self, state: GameState, reward_type: CellType) -> GameState:
        """Apply the effect of stepping on a reward"""
        new_state = copy.deepcopy(state)
        
        if reward_type == CellType.REWARD1:  # Decrease gravity
            new_state.energy_multiplier = max(0.25, new_state.energy_multiplier * 0.5)
        elif reward_type == CellType.REWARD2:  # Increase speed
            new_state.speed_multiplier = max(0.25, new_state.speed_multiplier * 0.5)
            
        return new_state
    
    def get_valid_moves(self, state: GameState) -> List[Tuple[Tuple[int, int], float, int]]:
        """Get valid moves from current state with their costs, avoiding traps"""
        moves = []
        current_row, current_col = state.position

        for next_row, next_col in self.grid.get_hex_neighbors(current_row, current_col):
            cell_type = self.grid.get_cell(next_row, next_col)

            # Avoid all traps
            if cell_type in [CellType.TRAP1, CellType.TRAP2, CellType.TRAP3, CellType.TRAP4]:
                continue

            if cell_type == CellType.OBSTACLE:
                continue

            # Normal movement cost
            base_energy = 1.0
            energy_cost = base_energy * state.energy_multiplier
            steps_cost = max(1, int(state.speed_multiplier))

            moves.append(((next_row, next_col), energy_cost, steps_cost))

        return moves
    
    def heuristic(self, state: GameState) -> float:
        """Improved heuristic function that considers strategic value"""
        uncollected_treasures = self.all_treasures - state.collected_treasures
        uncollected_rewards = self.all_rewards - state.collected_rewards
        
        if state.treasures_removed:
            uncollected_treasures = self.all_treasures
        
        all_uncollected = uncollected_treasures | uncollected_rewards
        
        if not all_uncollected:
            return 0
        
        current_pos = state.position
        
        # Calculate minimum spanning tree cost approximation
        if len(all_uncollected) == 1:
            target = next(iter(all_uncollected))
            return self.hex_distance(current_pos, target) * state.energy_multiplier
        
        # For multiple targets, use improved estimation
        distances = []
        for target in all_uncollected:
            dist = self.hex_distance(current_pos, target)
            # Prioritize rewards by reducing their heuristic cost
            if target in uncollected_rewards:
                dist *= 0.5  # Strong preference for collecting rewards first
            distances.append(dist)
        
        # Estimate minimum cost considering current multipliers
        min_distance = min(distances) if distances else 0
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        # Strategic penalties and bonuses
        strategy_cost = 0
        
        # Heavy penalty for high multipliers (encourages collecting rewards)
        if state.energy_multiplier > 1:
            strategy_cost += (state.energy_multiplier - 1) * 5
        if state.speed_multiplier > 1:
            strategy_cost += (state.speed_multiplier - 1) * 3
            
        # Bonus for having good multipliers
        if state.energy_multiplier < 1:
            strategy_cost -= (1 - state.energy_multiplier) * 2
        if state.speed_multiplier < 1:
            strategy_cost -= (1 - state.speed_multiplier) * 2
        
        # Additional cost for remaining items
        remaining_cost = len(all_uncollected) * avg_distance * 0.3
        
        return min_distance + remaining_cost + strategy_cost
    
    def is_goal_state(self, state: GameState) -> bool:
        """Goal: all treasures collected, rewards optional."""
        treasures_collected = (state.treasures_removed and 
                         len(state.collected_treasures) >= len(self.all_treasures)) or \
                         (state.collected_treasures == self.all_treasures)
        return treasures_collected
    
    def solve_optimized_astar(self) -> Optional[List[Tuple[int, int]]]:
        """Optimized A* search with better heuristics and pruning"""
        start_state = GameState(
            position=self.grid.entry_point,
            collected_treasures=set(),
            collected_rewards=set(),
            energy_multiplier=1.0,
            speed_multiplier=1.0,
            last_direction=None,
            total_energy=0.0,
            total_steps=0,
            treasures_removed=False
        )
        
        print(f"Starting optimized search from {start_state.position}")
        print(f"Need to collect {len(self.all_treasures)} treasures and {len(self.all_rewards)} rewards")
        
        # Priority queue: (f_score, g_score, state, path)
        open_set = [(self.heuristic(start_state), 0, start_state, [start_state.position])]
        
        # Better state tracking - only keep best score for each state configuration
        best_scores = {}
        
        iterations = 0
        max_iterations = 100000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            if iterations % 10000 == 0:
                print(f"Search iteration {iterations}, queue size: {len(open_set)}")
            
            f_score, g_score, current_state, path = heapq.heappop(open_set)
            
            # Create state key for tracking
            state_key = (
                current_state.position,
                tuple(sorted(current_state.collected_treasures)),
                tuple(sorted(current_state.collected_rewards)),
                round(current_state.energy_multiplier, 2),
                round(current_state.speed_multiplier, 2),
                current_state.treasures_removed
            )
            
            # Skip if we've seen this state with better cost
            total_cost = current_state.total_energy + current_state.total_steps
            if state_key in best_scores and best_scores[state_key] <= total_cost:
                continue
                
            best_scores[state_key] = total_cost
            
            if self.is_goal_state(current_state):
                print("Optimized solution found!")
                print(f"Total energy: {current_state.total_energy:.2f}")
                print(f"Total steps: {current_state.total_steps}")
                print(f"Combined cost: {total_cost:.2f}")
                print(f"Path length: {len(path)} moves")
                print(f"Iterations: {iterations}")
                return path
            
            # Generate next states
            for (next_row, next_col), energy_cost, steps_cost in self.get_valid_moves(current_state):
                new_state = copy.deepcopy(current_state)
                new_state.position = (next_row, next_col)
                new_state.total_energy += energy_cost
                new_state.total_steps += steps_cost
                new_state.last_direction = (next_row - current_state.position[0], 
                                          next_col - current_state.position[1])
                
                cell_type = self.grid.get_cell(next_row, next_col)
                
                # Handle cell effects
                if cell_type == CellType.TREASURE and (next_row, next_col) not in new_state.collected_treasures:
                    new_state.collected_treasures.add((next_row, next_col))
                elif cell_type in [CellType.REWARD1, CellType.REWARD2] and (next_row, next_col) not in new_state.collected_rewards:
                    new_state.collected_rewards.add((next_row, next_col))
                    new_state = self.apply_reward_effect(new_state, cell_type)
                elif cell_type in [CellType.TRAP1, CellType.TRAP2, CellType.TRAP3, CellType.TRAP4]:
                    new_state = self.apply_trap_effect(new_state, cell_type)
                
                new_g_score = new_state.total_energy + new_state.total_steps
                new_f_score = new_g_score + self.heuristic(new_state)
                
                # Pruning: skip if this path is getting too expensive
                if new_g_score > 50:  # Reasonable upper bound
                    continue
                
                new_path = path + [(next_row, next_col)]
                heapq.heappush(open_set, (new_f_score, new_g_score, new_state, new_path))
        
        print(f"Search completed after {iterations} iterations")
        return None
    
    def validate_path(self, path: List[Tuple[int, int]]) -> bool:
        """Validate that path doesn't pass through obstacles"""
        for i in range(len(path)-1):
            current = path[i]
            next_pos = path[i+1]
            
            if next_pos not in self.grid.get_hex_neighbors(current[0], current[1]):
                return False
                
            if self.grid.get_cell(next_pos[0], next_pos[1]) == CellType.OBSTACLE:
                return False
        return True

class HexGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[CellType.EMPTY for _ in range(width)] for _ in range(height)]
        self.treasures = set()
        self.rewards = set()
        self.entry_point = (0, 0)
        
    def set_cell(self, row: int, col: int, cell_type: CellType):
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row][col] = cell_type
            if cell_type == CellType.TREASURE:
                self.treasures.add((row, col))
            elif cell_type in [CellType.REWARD1, CellType.REWARD2]:
                self.rewards.add((row, col))
            elif cell_type == CellType.ENTRY:
                self.entry_point = (row, col)
    
    def get_cell(self, row: int, col: int) -> CellType:
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row][col]
        return CellType.OBSTACLE
    
    def get_hex_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get hexagonal neighbors for even-columns-higher layout"""
        neighbors = []
        
        if col % 2 == 0:  # Even column (higher)
            offsets = [
                (0, -1), (0, 1),   # NW, NE
                (-1, 0), (1, 0),   # N, S
                (1, -1), (1, 1)    # SW, SE
            ]
        else:  # Odd column (lower)
            offsets = [
                (-1, -1), (-1, 1), # NW, NE
                (-1, 0), (1, 0),   # N, S
                (0, -1), (0, 1)    # SW, SE
            ]
        
        for dr, dc in offsets:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                neighbors.append((new_row, new_col))
        
        return neighbors

class HexVisualizer:
    def __init__(self, grid: HexGrid):
        self.grid = grid
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Optimized Hexagonal Treasure Hunt - A* Pathfinding")
        
    def hex_to_pixel(self, row: int, col: int) -> Tuple[float, float]:
        """Convert hex coordinates to pixel coordinates"""
        x = HEX_RADIUS * 3/2 * col + 100
        y = HEX_RADIUS * math.sqrt(3) * (row - 0.5 * (col % 2)) + 100
        return x, y
    
    def draw_hexagon(self, surface, center: Tuple[float, float], color: Tuple[int, int, int], 
                    outline_color: Tuple[int, int, int] = None):
        """Draw a hexagon at the given center"""
        points = []
        for i in range(6):
            angle = math.pi / 3 * i
            x = center[0] + HEX_RADIUS * math.cos(angle)
            y = center[1] + HEX_RADIUS * math.sin(angle)
            points.append((x, y))
        
        pygame.draw.polygon(surface, color, points)
        if outline_color:
            pygame.draw.polygon(surface, outline_color, points, 2)
    
    def draw_grid(self, path: List[Tuple[int, int]] = None, show_move_numbers: bool = False):
        """Draw grid with path"""
        self.screen.fill(COLORS['WHITE'])
        font = pygame.font.Font(None, 20)
        
        # Draw all hexes
        for row in range(self.grid.height):
            for col in range(self.grid.width):
                center = self.hex_to_pixel(row, col)
                cell_type = self.grid.get_cell(row, col)
                color = self.get_cell_color(cell_type)
                self.draw_hexagon(self.screen, center, color, COLORS['GRAY'])
        
        if path:
            path_points = []
            
            # Draw path segments
            for i in range(len(path)):
                row, col = path[i]
                center = self.hex_to_pixel(row, col)
                path_points.append(center)
                
                # Highlight path cells
                self.draw_hexagon(self.screen, center, COLORS['CYAN'], COLORS['BLUE'])
                
                # Draw move numbers
                if show_move_numbers and i > 0:
                    text = font.render(str(i), True, COLORS['BLACK'])
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)
            
            # Draw connecting lines
            if len(path_points) > 1:
                for i in range(len(path_points)-1):
                    pygame.draw.line(self.screen, COLORS['BLUE'], path_points[i], path_points[i+1], 3)
        
        self.draw_legend()
        
    def get_cell_color(self, cell_type: CellType) -> Tuple[int, int, int]:
        """Get color for each cell type"""
        color_map = {
            CellType.EMPTY: COLORS['WHITE'],
            CellType.OBSTACLE: COLORS['BLACK'],
            CellType.TRAP1: COLORS['RED'],
            CellType.TRAP2: COLORS['DARK_GRAY'],
            CellType.TRAP3: COLORS['PURPLE'],
            CellType.TRAP4: COLORS['BROWN'],
            CellType.REWARD1: COLORS['LIGHT_GREEN'],
            CellType.REWARD2: COLORS['LIGHT_BLUE'],
            CellType.TREASURE: COLORS['YELLOW'],
            CellType.ENTRY: COLORS['GREEN']
        }
        return color_map.get(cell_type, COLORS['WHITE'])
    
    def draw_legend(self):
        """Draw legend explaining the symbols"""
        legend_items = [
            ("Entry", COLORS['GREEN']),
            ("Treasure", COLORS['YELLOW']),
            ("Obstacle", COLORS['BLACK']),
            ("Trap 1 (↑Energy)", COLORS['RED']),
            ("Trap 2 (↓Speed)", COLORS['DARK_GRAY']),
            ("Trap 3 (Push)", COLORS['PURPLE']),
            ("Trap 4 (Remove)", COLORS['BROWN']),
            ("Reward 1 (↓Energy)", COLORS['LIGHT_GREEN']),
            ("Reward 2 (↑Speed)", COLORS['LIGHT_BLUE']),
            ("Optimized Path", COLORS['CYAN'])
        ]
        
        font = pygame.font.Font(None, 24)
        y_offset = 20
        
        for i, (text, color) in enumerate(legend_items):
            pygame.draw.rect(self.screen, color, (WINDOW_WIDTH - 200, y_offset + i * 30, 20, 20))
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (WINDOW_WIDTH - 170, y_offset + i * 30))

def create_sample_world() -> HexGrid:
    """Create corrected world with all rewards and treasures"""
    grid = HexGrid(GRID_WIDTH, GRID_HEIGHT)
    
    # Row 0
    grid.set_cell(0, 0, CellType.ENTRY)    # E
    grid.set_cell(0, 4, CellType.REWARD1)  # !1
    
    # Row 1
    grid.set_cell(1, 1, CellType.TRAP2)    # 2
    grid.set_cell(1, 3, CellType.TRAP4)    # 4
    grid.set_cell(1, 4, CellType.TREASURE) # T
    grid.set_cell(1, 6, CellType.TRAP3)    # 3
    grid.set_cell(1, 8, CellType.OBSTACLE) # #
    
    # Row 2
    grid.set_cell(2, 2, CellType.OBSTACLE) # #
    grid.set_cell(2, 4, CellType.OBSTACLE) # #
    grid.set_cell(2, 7, CellType.REWARD2)  # !2
    grid.set_cell(2, 8, CellType.TRAP1)    # 1
    
    # Row 3
    grid.set_cell(3, 0, CellType.OBSTACLE) # #
    grid.set_cell(3, 1, CellType.REWARD1)  # !1
    grid.set_cell(3, 3, CellType.OBSTACLE) # #
    grid.set_cell(3, 5, CellType.TRAP3)    # 3
    grid.set_cell(3, 6, CellType.OBSTACLE) # #
    grid.set_cell(3, 7, CellType.TREASURE) # T
    grid.set_cell(3, 9, CellType.TREASURE) # T
    
    # Row 4
    grid.set_cell(4, 2, CellType.TRAP2)    # 2
    grid.set_cell(4, 3, CellType.TREASURE) # T
    grid.set_cell(4, 4, CellType.OBSTACLE) # #
    grid.set_cell(4, 6, CellType.OBSTACLE) # #
    grid.set_cell(4, 7, CellType.OBSTACLE) # #
    
    # Row 5
    grid.set_cell(5, 5, CellType.REWARD2)  # !2
    
    return grid

def main():
    # Initialize
    grid = create_sample_world()
    solver = OptimizedTreasureHuntSolver(grid)
    visualizer = HexVisualizer(grid)
    
    # Print grid information
    print(f"Grid created with {len(grid.treasures)} treasures and {len(grid.rewards)} rewards")
    print(f"Treasures at: {grid.treasures}")
    print(f"Rewards at: {grid.rewards}")
    
    # Solve with optimized algorithm
    print("Solving with optimized A* algorithm...")
    solution_path = solver.solve_optimized_astar()
    
    if solution_path:
        print(f"Optimized solution found in {len(solution_path)-1} moves!")
        print("Path:", solution_path)
        
        # Verify all items are collected
        treasures_in_path = set()
        rewards_in_path = set()
        for pos in solution_path:
            if pos in grid.treasures:
                treasures_in_path.add(pos)
            if pos in grid.rewards:
                rewards_in_path.add(pos)
        
        print(f"Treasures collected: {len(treasures_in_path)}/{len(grid.treasures)}")
        print(f"Rewards collected: {len(rewards_in_path)}/{len(grid.rewards)}")
    else:
        print("No solution found!")
        return
    
    # Visualization loop
    clock = pygame.time.Clock()
    show_solution = False
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show_solution = not show_solution
        
        # Draw appropriate view
        if show_solution:
            visualizer.draw_grid(solution_path, show_move_numbers=True)
            font = pygame.font.Font(None, 36)
            text = font.render(f"Optimized A* Solution: {len(solution_path)-1} moves", True, COLORS['BLACK'])
            visualizer.screen.blit(text, (20, 20))
        else:
            visualizer.draw_grid()
            font = pygame.font.Font(None, 36)
            text = font.render("Press SPACE to show optimized A* solution", True, COLORS['BLACK'])
            visualizer.screen.blit(text, (20, 20))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
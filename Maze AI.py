import pygame
import math
import heapq
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
import copy

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
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
    position: Tuple[int, int]
    collected_treasures: Set[Tuple[int, int]]
    energy_multiplier: float  # Affects energy consumption
    speed_multiplier: float   # Affects steps needed
    last_direction: Optional[Tuple[int, int]]
    total_energy: float
    total_steps: int
    treasures_removed: bool  # Track if Trap4 was activated

    def __lt__(self, other):
        # Define how to compare two GameStates
        # Here we just compare positions as a simple solution
        return self.position < other.position

class HexGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[CellType.EMPTY for _ in range(width)] for _ in range(height)]
        self.treasures = set()
        self.entry_point = (0, 0)
        
    def set_cell(self, row: int, col: int, cell_type: CellType):
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row][col] = cell_type
            if cell_type == CellType.TREASURE:
                self.treasures.add((row, col))
            elif cell_type == CellType.ENTRY:
                self.entry_point = (row, col)
    
    def get_cell(self, row: int, col: int) -> CellType:
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row][col]
        return CellType.OBSTACLE  # Out of bounds treated as obstacle

    def get_hex_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get hexagonal neighbors for odd-columns-higher layout"""
        neighbors = []
        
        # Modified offsets for even-columns-higher layout
        if col % 2 == 0:  # Even column (lower)
            offsets = [
                (0, -1), (0, 1),   # NW, NE
                (-1, 0), (1, 0),     # N, S
                (1, -1), (1, 1)       # SW, SE
            ]
        else:  # Odd column (higher)
            offsets = [
                (-1, -1), (-1, 1),   # NW, NE
                (-1, 0), (1, 0),     # N, S
                (0, -1), (0, 1)       # SW, SE
            ]
        
        for dr, dc in offsets:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                neighbors.append((new_row, new_col))
        
        return neighbors

class TreasureHuntSolver:
    ODD_Q_NEIGHBORS = [
        (-1, 0),  # N
        (0, 1),   # NE
        (1, 1),   # SE
        (1, 0),   # S
        (1, -1),  # SW
        (0, -1)   # NW
    ]
    
    EVEN_Q_NEIGHBORS = [
        (-1, 0),  # N
        (-1, 1),  # NE
        (0, 1),   # SE
        (1, 0),   # S
        (0, -1),  # SW
        (-1, -1)  # NW
    ]
    
    def __init__(self, grid: HexGrid):
        self.grid = grid
        self.all_treasures = copy.deepcopy(grid.treasures)
    
    def get_direction_index(self, from_row, from_col, to_row, to_col):
        deltas = (to_row - from_row, to_col - from_col)
        neighbors = self.ODD_Q_NEIGHBORS if from_col % 2 == 1 else self.EVEN_Q_NEIGHBORS
        for idx, (dr, dc) in enumerate(neighbors):
            if (dr, dc) == deltas:
                return idx
        return None  # not a direct neighbor
    
    def get_neighbor_in_direction(self, row, col, dir_idx):
        neighbors = self.ODD_Q_NEIGHBORS if col % 2 == 1 else self.EVEN_Q_NEIGHBORS
        dr, dc = neighbors[dir_idx]
        return row + dr, col + dc
    
    def is_valid(self, row, col):
        return (
            0 <= row < self.grid.height and
            0 <= col < self.grid.width and
            self.grid.get_cell(row, col) != CellType.OBSTACLE
        )

    def apply_trap_effect(self, state: GameState, trap_type: CellType) -> GameState:
        """Apply the effect of stepping on a trap"""
        new_state = copy.deepcopy(state)
        
        if trap_type == CellType.TRAP1:  # Increase gravity
            new_state.energy_multiplier *= 2
        elif trap_type == CellType.TRAP2:  # Decrease speed
            new_state.speed_multiplier *= 2
        elif trap_type == CellType.TRAP3:  # Move 2 cells in last direction
            current_row, current_col = new_state.position
            if new_state.last_direction:
                dir_idx = new_state.last_direction
                row1, col1 = self.get_neighbor_in_direction(current_row, current_col, dir_idx)
                row2, col2 = self.get_neighbor_in_direction(row1, col1, dir_idx)
            
                valid1 = self.is_valid(row1, col1)
                valid2 = self.is_valid(row2, col2)
            
                if valid1 and valid2:
                    new_state.position = (row2, col2)
                elif valid1:
                    new_state.position = (row1, col1)
                # else: no move
        elif trap_type == CellType.TRAP4:  # Remove uncollected treasures
            new_state.treasures_removed = True
            
        return new_state
    
    def apply_reward_effect(self, state: GameState, reward_type: CellType) -> GameState:
        """Apply the effect of stepping on a reward"""
        new_state = copy.deepcopy(state)
        
        if reward_type == CellType.REWARD1:  # Decrease gravity
            new_state.energy_multiplier *= 0.5
        elif reward_type == CellType.REWARD2:  # Increase speed
            new_state.speed_multiplier *= 0.5
            
        return new_state
    
    def get_valid_moves(self, state: GameState) -> List[Tuple[Tuple[int, int], float, int]]:
        """Get valid moves from current state with their costs"""
        moves = []
        current_row, current_col = state.position
        
        for next_row, next_col in self.grid.get_hex_neighbors(current_row, current_col):
            cell_type = self.grid.get_cell(next_row, next_col)
            
            if cell_type == CellType.OBSTACLE:
                continue
                
            # Calculate movement cost
            base_energy = 1.0
            energy_cost = base_energy * state.energy_multiplier
            steps_cost = max(1, int(state.speed_multiplier))
            
            moves.append(((next_row, next_col), energy_cost, steps_cost))
            
        return moves
    
    def heuristic(self, state: GameState) -> float:
        """Heuristic function for A* search"""
        if state.treasures_removed:
            # If treasures are removed, just get to any treasure location
            available_treasures = self.all_treasures
        else:
            available_treasures = self.all_treasures - state.collected_treasures
        
        if not available_treasures:
            return 0
        
        # Manhattan distance to nearest uncollected treasure
        min_distance = float('inf')
        current_row, current_col = state.position
        
        for treasure_row, treasure_col in available_treasures:
            # Hexagonal distance approximation
            distance = max(abs(current_row - treasure_row), 
                          abs(current_col - treasure_col))
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def is_goal_state(self, state: GameState) -> bool:
        """Check if all treasures are collected"""
        # If Trap4 was activated but not all treasures collected → fail
        if state.treasures_removed and state.collected_treasures != self.all_treasures:
            return False
        
        # All treasures collected → goal achieved
        return state.collected_treasures == self.all_treasures
    
    def solve_astar(self) -> Optional[List[Tuple[int, int]]]:
        """Solve using A* search algorithm with strict adjacency checking"""
        start_state = GameState(
            position=self.grid.entry_point,
            collected_treasures=set(),
            energy_multiplier=1.0,
            speed_multiplier=1.0,
            last_direction=None,
            total_energy=0.0,
            total_steps=0,
            treasures_removed=False
        )
        print("Initial state:", start_state.energy_multiplier)
        
        # Priority queue: (f_score, g_score, state, path)
        open_set = [(self.heuristic(start_state), 0, start_state, [start_state.position])]
        closed_set = set()
        
        while open_set:
            f_score, g_score, current_state, path = heapq.heappop(open_set)
            
            # Create state key for closed set
            state_key = (
                current_state.position,
                tuple(sorted(current_state.collected_treasures)),
                current_state.energy_multiplier,
                current_state.speed_multiplier,
                current_state.treasures_removed
            )
            
            if state_key in closed_set:
                continue
                
            closed_set.add(state_key)
            
            if self.is_goal_state(current_state):
                # Validate the final path
                if self.validate_path(path):
                    return path, current_state
                continue
            
            # Generate next states
            for (next_row, next_col), energy_cost, steps_cost in self.get_valid_moves(current_state):
                new_state = copy.deepcopy(current_state)
                new_state.position = (next_row, next_col)
                new_state.total_energy += energy_cost
                new_state.total_steps += steps_cost
                # Set last_direction with hex-specific adjustment
                current_row, current_col = current_state.position
                
                # Compute direction index using helper
                dir_idx = self.get_direction_index(current_row, current_col, next_row, next_col)
                if dir_idx is not None:
                    new_state.last_direction = dir_idx
                else:
                    # This should not happen if moves come from valid neighbors
                    new_state.last_direction = None

                cell_type = self.grid.get_cell(next_row, next_col)
                
                # Handle cell effects
                if cell_type == CellType.TREASURE and (next_row, next_col) not in new_state.collected_treasures:
                    new_state.collected_treasures.add((next_row, next_col))
                elif cell_type in [CellType.TRAP1, CellType.TRAP2, CellType.TRAP3, CellType.TRAP4]:
                    new_state = self.apply_trap_effect(new_state, cell_type)
                elif cell_type in [CellType.REWARD1, CellType.REWARD2]:
                    new_state = self.apply_reward_effect(new_state, cell_type)
                
                # Apply extra penalty for triggering TRAP4 before completing treasure collection
                trap_penalty = 0.0
                if cell_type == CellType.TRAP4 and new_state.collected_treasures != self.all_treasures:
                    trap_penalty = 100.0  # Make it very costly
                
                new_g_score = g_score + energy_cost + steps_cost + trap_penalty
                new_f_score = new_g_score + self.heuristic(new_state)
                
                new_path = path + [(next_row, next_col)]
                heapq.heappush(open_set, (new_f_score, new_g_score, new_state, new_path))
            
        return None  # No solution found
        
    def validate_path(self, path: List[Tuple[int, int]]) -> bool:
        """Strict validation that path doesn't pass through obstacles"""
        for i in range(len(path)-1):
            current = path[i]
            next_pos = path[i+1]
    
            # Must be adjacent or a TRAP3 jump
            if next_pos not in self.grid.get_hex_neighbors(current[0], current[1]):
                # Check if this is a TRAP3 jump
                row_diff = next_pos[0] - current[0]
                col_diff = next_pos[1] - current[1]
    
                if abs(row_diff) > 1 or abs(col_diff) > 1:
                    return False
    
                # Ensure it was a TRAP3 cell
                if self.grid.get_cell(current[0], current[1]) != CellType.TRAP3:
                    return False
    
                # Calculate intermediate cell
                mid_row = (current[0] + next_pos[0]) // 2
                mid_col = (current[1] + next_pos[1]) // 2
    
                # Ensure intermediate and final positions are valid
                if self.grid.get_cell(mid_row, mid_col) == CellType.OBSTACLE:
                    return False
                if self.grid.get_cell(next_pos[0], next_pos[1]) == CellType.OBSTACLE:
                    return False
    
            # Ensure target cell is not an obstacle
            if self.grid.get_cell(next_pos[0], next_pos[1]) == CellType.OBSTACLE:
                return False
    
        return True

class HexVisualizer:
    def __init__(self, grid: HexGrid):
        self.grid = grid
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hexagonal Treasure Hunt")
        
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
        """Draw grid with path validation"""
        self.screen.fill(COLORS['WHITE'])
        font = pygame.font.Font(None, 24)
        
        # Draw all hexes
        for row in range(self.grid.height):
            for col in range(self.grid.width):
                center = self.hex_to_pixel(row, col)
                cell_type = self.grid.get_cell(row, col)
                color = self.get_cell_color(cell_type)
                self.draw_hexagon(self.screen, center, color, COLORS['GRAY'])
        
        if path:
            path_points = []
            valid_path = True
            
            # First validate the entire path
            solver = TreasureHuntSolver(self.grid)
            if not solver.validate_path(path):
                valid_path = False
                # Draw warning text
                warning_font = pygame.font.Font(None, 36)
                warning = warning_font.render("INVALID PATH DETECTED!", True, COLORS['RED'])
                self.screen.blit(warning, (WINDOW_WIDTH//2 - 150, 20))
            
            # Draw path segments with validity check
            for i in range(len(path)):
                row, col = path[i]
                center = self.hex_to_pixel(row, col)
                path_points.append(center)
                
                # Highlight path cells
                cell_color = COLORS['RED'] if not valid_path else COLORS['CYAN']
                self.draw_hexagon(self.screen, center, cell_color, COLORS['BLUE'])
                
                # Draw move numbers
                if show_move_numbers and i > 0:
                    text = font.render(str(i), True, COLORS['BLACK'])
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)
            
            # Draw connecting lines with segment validation
            if len(path_points) > 1:
                for i in range(len(path_points)-1):
                    current = path[i]
                    next_pos = path[i+1]
                    line_color = COLORS['BLUE']
                    
                    # Check if this segment is valid
                    if next_pos not in self.grid.get_hex_neighbors(current[0], current[1]):
                        line_color = COLORS['RED']
                    elif self.grid.get_cell(next_pos[0], next_pos[1]) == CellType.OBSTACLE:
                        line_color = COLORS['RED']
                    
                    pygame.draw.line(self.screen, line_color, path_points[i], path_points[i+1], 3)
        
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
            ("Path", COLORS['CYAN'])
        ]
        
        font = pygame.font.Font(None, 24)
        y_offset = 20
        
        for i, (text, color) in enumerate(legend_items):
            # Draw color box
            pygame.draw.rect(self.screen, color, (WINDOW_WIDTH - 200, y_offset + i * 30, 20, 20))
            # Draw text
            text_surface = font.render(text, True, COLORS['BLACK'])
            self.screen.blit(text_surface, (WINDOW_WIDTH - 170, y_offset + i * 30))

def create_sample_world() -> HexGrid:
    """Create world matching the exact layout provided"""
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
    solver = TreasureHuntSolver(grid)
    visualizer = HexVisualizer(grid)
    
    # Solve
    print("Solving...")
    solution = solver.solve_astar()  # Returns (path, final_state)
    
    if solution:
        solution_path, final_state = solution
        print(f"Solution found in {len(solution_path)-1} moves!")
        print("Path:", solution_path)
        print(f"Total energy used: {final_state.total_energy:.2f}")
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
            # Add status text
            font = pygame.font.Font(None, 36)
            text = font.render(f"Solution: {len(solution_path)-1} moves", True, COLORS['BLACK'])
            visualizer.screen.blit(text, (20, 20))
        else:
            visualizer.draw_grid()  # Original view
            font = pygame.font.Font(None, 36)
            text = font.render("Press SPACE to show solution", True, COLORS['BLACK'])
            visualizer.screen.blit(text, (20, 20))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
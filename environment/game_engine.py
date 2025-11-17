import os
import random
import math

from .levels import LEVELS, LEVEL_ORDER, LEVEL_MAX_POINTS
from .maze import Maze, PELLET, POWER, PACMAN, GHOST, WALL, DEFAULT_MAP
from .entities import Pacman, Ghost
from ai_modules.controller import HybridController


TILE_SIZE = 28
HIGHSCORE_FILE = "highscore.txt"


# ----------------------------------------------------------
# Persistent Highscore (Level-wise) - MODULE LEVEL FUNCTIONS
# ----------------------------------------------------------
def load_highscore_for_level(level_name):
    """Load highscore for a specific level"""
    filename = f"highscore_{level_name}.txt"
    if not os.path.exists(filename):
        return 0
    try:
        with open(filename, "r") as f:
            return int(f.read().strip())
    except:
        return 0

def save_highscore_for_level(level_name, score):
    """Save highscore for a specific level"""
    filename = f"highscore_{level_name}.txt"
    try:
        with open(filename, "w") as f:
            f.write(str(score))
    except:
        pass


class GameEngine:

    # ----------------------------------------------------------
    def __init__(self, map_lines=None, level_name=None):
        # Level management
        self.current_level_index = 0
        self.current_level_name = LEVEL_ORDER[0]  # Start with beginner
        self.current_variation = 0
        self.all_levels_complete = False
        
        # Persistent highscores (level-wise)
        self.highscores = self.load_all_highscores()
        
        # Load map
        if map_lines is None:
            if level_name is None:
                level_name = self.current_level_name
            # Randomly select variation
            variations = LEVELS[level_name]
            self.current_variation = random.randint(0, len(variations) - 1)
            map_lines = variations[self.current_variation]
        
        self.original_map = [row[:] for row in map_lines]

        self.maze = Maze(map_lines)

        self.tile_size = TILE_SIZE
        self.width_px = self.maze.width * TILE_SIZE
        self.height_px = self.maze.height * TILE_SIZE

        # Game objects
        self.pellets = set()
        self.power_pellets = set()
        self.pacman = None
        self.ghosts = []

        self._load_from_map()

        # AI controller
        self.controller = HybridController()

        # Game state
        self.running = True
        self.game_over = False
        self.win = False
        self.step_time = 0


    # ----------------------------------------------------------
    # LEVEL MANAGEMENT METHODS (INSIDE CLASS)
    # ----------------------------------------------------------
    def load_all_highscores(self):
        """Load highscores for all levels"""
        return {
            level: load_highscore_for_level(level) 
            for level in LEVEL_ORDER
        }

    def get_current_highscore(self):
        """Get highscore for current level"""
        return self.highscores.get(self.current_level_name, 0)

    def save_current_highscore(self):
        """Save highscore for current level if beaten"""
        current_score = self.pacman.score
        current_high = self.get_current_highscore()
        
        if current_score > current_high:
            self.highscores[self.current_level_name] = current_score
            save_highscore_for_level(self.current_level_name, current_score)

    def load_next_level(self):
        """Load the next level in progression"""
        self.current_level_index += 1
        
        if self.current_level_index >= len(LEVEL_ORDER):
            # Game completed!
            self.all_levels_complete = True
            return False
        
        self.current_level_name = LEVEL_ORDER[self.current_level_index]
        variations = LEVELS[self.current_level_name]
        self.current_variation = random.randint(0, len(variations) - 1)
        map_lines = variations[self.current_variation]
        
        # Reset game with new map
        self.original_map = [row[:] for row in map_lines]
        self.maze = Maze(map_lines)  # ADDED: Create new maze first
        
        # ADDED: Update dimensions
        self.width_px = self.maze.width * TILE_SIZE
        self.height_px = self.maze.height * TILE_SIZE
        
        self.reset()
        return True

    def reset_to_level(self, level_index=None):
        """Reset current level or specific level"""
        if level_index is not None:
            self.current_level_index = level_index
            self.current_level_name = LEVEL_ORDER[level_index]
        
        variations = LEVELS[self.current_level_name]
        self.current_variation = random.randint(0, len(variations) - 1)
        map_lines = variations[self.current_variation]
        self.original_map = [row[:] for row in map_lines]
        
        self.maze = Maze(map_lines)  # ADDED: Create new maze first
        
        # ADDED: Update dimensions
        self.width_px = self.maze.width * TILE_SIZE
        self.height_px = self.maze.height * TILE_SIZE
        
        self.reset()

    # ----------------------------------------------------------
    def _load_from_map(self):
        for y, row in enumerate(self.maze.raw):
            for x, ch in enumerate(row):

                if ch == '.':
                    self.pellets.add((x, y))
                    self.maze.raw[y][x] = ' '

                elif ch == 'o':
                    self.power_pellets.add((x, y))
                    self.maze.raw[y][x] = ' '

                elif ch == 'P':
                    self.pacman = Pacman(y, x)
                    self.maze.raw[y][x] = ' '

                elif ch == 'G':
                    self.ghosts.append(Ghost(y, x))
                    self.maze.raw[y][x] = ' '


        # Default pacman location
        if self.pacman is None:
            cx = self.maze.width // 2   # x (col)
            cy = self.maze.height // 2  # y (row)
            self.pacman = Pacman(cy, cx)  # pass (row, col)


        # Pixel positions
        px, py = self.maze.tile_center(self.pacman.tx, self.pacman.ty, self.tile_size)
        self.pacman.set_pixel_pos(px, py)

        for g in self.ghosts:
            gx, gy = self.maze.tile_center(g.tx, g.ty, self.tile_size)
            g.set_pixel_pos(gx, gy)



    # ----------------------------------------------------------
    # Reset game without touching highscore
    # ----------------------------------------------------------
    def reset(self):
        self.maze = Maze(self.original_map)

        # ADDED: Update dimensions based on new maze
        self.width_px = self.maze.width * TILE_SIZE
        self.height_px = self.maze.height * TILE_SIZE

        self.pellets = set()
        self.power_pellets = set()
        self.ghosts = []
        self.pacman = None

        self._load_from_map()

        self.running = True
        self.game_over = False
        self.win = False
        self.step_time = 0
        
        # Reset speeds to default
        if self.pacman:
            self.pacman.move_delay = self.pacman.normal_move_delay
        
        for g in self.ghosts:
            g.move_delay = g.normal_move_delay
            g.state = "normal"



    # ----------------------------------------------------------
    def handle_keydown(self, key):
        import pygame

        if key == pygame.K_UP: self.pacman.set_intent(0,-1)
        elif key == pygame.K_DOWN: self.pacman.set_intent(0,1)
        elif key == pygame.K_LEFT: self.pacman.set_intent(-1,0)
        elif key == pygame.K_RIGHT: self.pacman.set_intent(1,0)
        elif key == pygame.K_a: self.pacman.toggle_autopilot()
        elif key == pygame.K_SPACE: self.reset()
        elif key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            if self.game_over and self.win:
                if not self.load_next_level():
                    # All levels complete - restart from beginning
                    self.reset_to_level(0)



    # ----------------------------------------------------------
    def update(self, dt):
        if self.game_over:
            return

        self.step_time += dt

        # ------------------------------------------------------
        # WIN condition → save highscore
        # ------------------------------------------------------
        if not self.pellets and not self.power_pellets:
            self.win = True
            self.running = False
            self.game_over = True
            self.save_current_highscore()
            return


        # ----------------- AUTOPILOT AI -----------------------
        if self.pacman.autopilot:
            escape = self._runaway_from_threat()

            if escape:
                self.pacman.set_intent(*escape)
            else:
                chase = self._nearest_vulnerable_ghost_direction()

                if chase:
                    self.pacman.set_intent(*chase)
                else:
                    dx, dy = self.controller.choose_action(self)

                    # fallback if AI stuck
                    if (dx, dy) == (0,0) or self.maze.is_wall(self.pacman.tx+dx, self.pacman.ty+dy):
                        dx, dy = self._greedy_step_to_nearest_pellet()

                    self.pacman.set_intent(dx, dy)


        # ------------------------------------------------------
        # PACMAN MOVEMENT
        # ------------------------------------------------------
        self.pacman.time_since_move += dt

        if self.pacman.time_since_move >= self.pacman.move_delay:
            self.pacman.time_since_move = 0

            dx, dy = self.pacman.direction

            nx = self.pacman.tx + dx
            ny = self.pacman.ty + dy

            if not self.maze.is_wall(nx, ny):
                self.pacman.set_tile(nx, ny)
                px, py = self.maze.tile_center(nx, ny, self.tile_size)
                self.pacman.set_pixel_pos(px, py)

                self.pacman.animation_timer += dt
                if self.pacman.animation_timer >= self.pacman.animation_speed:
                    self.pacman.animation_timer = 0
                    self.pacman.mouth_open = not self.pacman.mouth_open



        # ------------------------------------------------------
        # PELLET CONSUMPTION
        # ------------------------------------------------------
        pos = (self.pacman.tx, self.pacman.ty)

        if pos in self.pellets:
            self.pellets.remove(pos)
            self.pacman.score += 10

        if pos in self.power_pellets:
            self.power_pellets.remove(pos)
            self.pacman.score += 25
            self.pacman.move_delay = self.pacman.boost_move_delay

            for g in self.ghosts:
                g.state = "vulnerable"
                g.vulnerable_timer = 5
                g.move_delay = g.vulnerable_move_delay  # SLOW DOWN GHOST



        # ------------------------------------------------------
        # GHOST MOVEMENT
        # ------------------------------------------------------
        for g in self.ghosts:
            g.prev_tx, g.prev_ty = g.tx, g.ty

            g.time_since_move += dt

            if g.state == "vulnerable":
                g.vulnerable_timer -= dt
                if g.vulnerable_timer <= 0:
                    g.state = "normal"
                    g.move_delay = g.normal_move_delay  # RESET SPEED

            if all(gg.state == "normal" for gg in self.ghosts):
                    self.pacman.move_delay = self.pacman.normal_move_delay

            if g.time_since_move < g.move_delay:
                continue

            g.time_since_move = 0

            dxg, dyg = g.choose_move(self.maze, pos)

            ngx, ngy = g.tx + dxg, g.ty + dyg
            if not self.maze.is_wall(ngx, ngy):
                g.set_tile(ngx, ngy)
                gx, gy = self.maze.tile_center(ngx, ngy, self.tile_size)
                g.set_pixel_pos(gx, gy)



        # ------------------------------------------------------
        # COLLISION CHECK
        # ------------------------------------------------------
        for g in self.ghosts:

            # direct collision
            if (g.tx, g.ty) == (self.pacman.tx, self.pacman.ty):
                self._handle_ghost_collision(g)

            # swap collision
            if (g.prev_tx, g.prev_ty) == (self.pacman.tx, self.pacman.ty) and \
               (self.pacman.prev_tx, self.pacman.prev_ty) == (g.tx, g.ty):
                self._handle_ghost_collision(g)



    # ----------------------------------------------------------
    def _handle_ghost_collision(self, g):

        # Pac-Man eats ghost
        if g.state == "vulnerable":
            self.pacman.score += 50

            g.set_tile(1, 1)
            g.set_pixel_pos(*self.maze.tile_center(1, 1, self.tile_size))
            g.state = "normal"
            g.move_delay = g.normal_move_delay  # RESET SPEED when eaten
            g.vulnerable_timer = 0
            
            if all(gg.state == "normal" for gg in self.ghosts):
                self.pacman.move_delay = self.pacman.normal_move_delay
            return
        # --------------------
        # PACMAN DIES
        # --------------------
        self.running = False
        self.game_over = True
        self.win = False
        self.save_current_highscore()

    # ----------------------------------------------------------
    def get_state_snapshot(self):
        return {
            "width": self.maze.width,
            "height": self.maze.height,
            "tile_size": self.tile_size,
            "pacman": self.pacman,
            "ghosts": self.ghosts,
            "pellets": self.pellets,
            "power_pellets": self.power_pellets,
            "maze": self.maze
        }



    # ----------------------------------------------------------
    # AI HELPERS
    # ----------------------------------------------------------
    def _runaway_from_threat(self, danger_radius=2):
        threats = [(g.tx, g.ty) for g in self.ghosts if g.state == "normal"]
        if not threats:
            return None

        px, py = self.pacman.tx, self.pacman.ty

        close = [(gx, gy) for gx, gy in threats
                 if abs(gx - px) + abs(gy - py) <= danger_radius]

        if not close:
            return None

        best_dirs = []
        best_dist = -1

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = px + dx, py + dy
            if self.maze.is_wall(nx, ny):
                continue

            min_d = min(abs(gx - nx) + abs(gy - ny) for gx, gy in close)

            if min_d > best_dist:
                best_dist = min_d
                best_dirs = [(dx, dy)]
            elif min_d == best_dist:
                best_dirs.append((dx, dy))

        return random.choice(best_dirs) if best_dirs else None



    def _nearest_vulnerable_ghost_direction(self):
        vuln = [(g.tx, g.ty) for g in self.ghosts if g.state == "vulnerable"]
        if not vuln:
            return None

        dist = self.maze.bfs_distance_grid(vuln)
        start = (self.pacman.tx, self.pacman.ty)

        if start not in dist:
            return None

        curr = dist[start]

        best = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = start[0]+dx, start[1]+dy
            if not self.maze.is_wall(nx, ny) and (nx, ny) in dist:
                if dist[(nx, ny)] < curr:
                    best.append((dx, dy))

        return random.choice(best) if best else None



    def _greedy_step_to_nearest_pellet(self):
        targets = list(self.pellets | self.power_pellets)
        if not targets:
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = self.pacman.tx + dx, self.pacman.ty + dy
                if not self.maze.is_wall(nx, ny):
                    return (dx, dy)
            return (0, 0)

        dist = self.maze.bfs_distance_grid(targets)
        start = (self.pacman.tx, self.pacman.ty)

        if start not in dist:
            return (0,0)

        curr = dist[start]
        candidates = []

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = start[0]+dx, start[1]+dy
            if self.maze.is_wall(nx, ny): continue
            if (nx, ny) in dist:
                candidates.append(((dx, dy), dist[(nx, ny)], (nx, ny)))

        if not candidates:
            return (0,0)

        better = [c for c in candidates if c[1] < curr]
        pick = better if better else candidates

        safe = [c for c in pick if not self._is_tile_dangerous(c[2][0], c[2][1])]
        if safe:
            pick = safe

        pick.sort(key=lambda x: x[1])
        best = [c for c in pick if c[1] == pick[0][1]]

        return random.choice(best)[0]


    def _is_tile_dangerous(self, tx, ty, danger_radius=2):
        for g in self.ghosts:
            if g.state == "normal":
                if abs(g.tx - tx) + abs(g.ty - ty) <= danger_radius:
                    return True
        return False

# AI-Pacman Engine

## Project Title & Objective

**Title:** AI-Powered Pacman Game with Intelligent Pathfinding and Decision-Making

**Objective:**
This project implements a fully functional Pacman game with an intelligent AI system that autonomously navigates the maze, avoids threats, and optimizes pellet collection. The primary goal is to demonstrate the application of artificial intelligence techniques including rule-based systems, feature extraction, and pathfinding algorithms (BFS) in a classic game environment. The project emphasizes clean architecture, modularity, and extensibility for future AI enhancements.

---

## Demo

<p align="center">
  <!-- <img src="assets/demo_video_win_pacman.gif" width="500" /> -->
  <img src="assets/demo-video.gif" width="500" />
</p>

---

## Dataset Details

### Game Environment

- Grid-based maze structure (tile-based system)
- Default map size: 15x7 tiles (28 pixels per tile)
- Configurable maze layouts using ASCII representation

### Game State Features

- Pacman position (tile coordinates)
- Ghost positions and states (normal/vulnerable)
- Pellet locations (normal and power pellets)
- Distance metrics (Manhattan distance to nearest ghost/pellet)
- Score and game status

### Map Representation

```
Legend:
- '%' = Wall
- '.' = Normal pellet (+10 points)
- 'o' = Power pellet (+25 points)
- 'P' = Pacman spawn point
- 'G' = Ghost spawn point
- ' ' = Empty space
```

### Scoring System

- Normal pellet consumption: +10 points
- Power pellet consumption: +25 points
- Vulnerable ghost eaten: +50 points
- Persistent high score tracking across sessions

---

## Algorithm / Model Used

### AI Decision-Making Architecture

#### 1. Hybrid Controller System

Hierarchical decision-making with three priority levels:

---

#### **Priority 1: Threat Avoidance (Defensive)**

- Algorithm: Distance-based threat detection with greedy escape
- Trigger: Any normal ghost within 2-tile Manhattan distance
- Method: Evaluates all four directions, selects move that maximizes distance from nearest threat
- Complexity: O(g × 4)

---

#### **Priority 2: Opportunity Pursuit (Offensive)**

- Algorithm: Breadth-First Search (BFS)
- Trigger: Vulnerable ghosts detected
- Method: Multi-source BFS + gradient descent
- Complexity: O(w × h)
- Speed boost: movement delay reduced from 0.25s → 0.45s

---

#### **Priority 3: Resource Collection (Default)**

- Algorithm: BFS with safety filtering
- Method:

  1. BFS from all pellets
  2. Remove tiles within 2 tiles of ghosts
  3. Select nearest safe pellet

- Fallback: Random valid move

---

### Feature Extraction System

```python
Features = {
    "min_ghost_dist": int,
    "min_pellet_dist": int,
    "pellet_count": int,
    "ghost_positions": [(x, y)],
    "pacman": (x, y)
}
```

---

### BFS Pathfinding Algorithm

- Multi-source BFS
- Returns optimal distance grid
- Handles pellets + vulnerable ghosts
- O(V + E) complexity
- Ideal for unweighted grid mazes

---

### Ghost AI System

#### Movement Algorithm

- Random valid direction
- No backtracking (no 180° flip)
- No pathfinding (keeps unpredictability)

#### Collision Detection

- Tile collision check
- Swap-collision handling
- Tracks previous tile positions

---

## Results

### Performance Metrics

#### AI Effectiveness

- Avoids ghosts reliably
- Good pellet collection efficiency
- Actively hunts vulnerable ghosts
- Dynamically adapts to changing environment

#### Technical Performance

- 30 FPS steady
- <16ms per frame updates (delta-time)
- BFS <5ms in typical mazes
- Low memory usage

#### Gameplay Observations

- Intelligent risk assessment
- Balanced offense + defense
- High score optimization
- Smooth animation

---

### Visual Demonstrations

The game features:

- Direction-aware Pacman with mouth animation
- 3D-effect walls
- Pulsing power pellets
- Wavy ghost sprites with expressive eyes
- Vulnerable ghost flashing effect
- HUD with score + high score

---

### Architecture Quality

- **Modularity:** AI, engine, renderer separated
- **Extensibility:** Easy to add new AI logic
- **Maintainability:** Clear documentation + structured modules

---

## Conclusion

This project successfully demonstrates intelligent AI behavior in a Pacman environment using:

1. Rule-based hierarchical decision-making
2. BFS pathfinding
3. Feature engineering
4. Clean state management

### Key Achievements

- Fully autonomous gameplay
- Robust AI that handles edge cases
- Modular and extendable architecture
- Strong software engineering principles

### Technical Insights

- BFS ideal for grid games
- Priority-based decision architecture is effective
- Feature extraction simplifies game-state complexity
- Time-based animation ensures hardware consistency

---

## Future Scope

### AI Enhancements

- Q-learning for adaptive behavior
- Neural networks using current features
- DQN (end-to-end learning)
- A\* pathfinding
- Minimax + MCTS
- Predictive ghost modeling
- Dynamic danger radius
- Multi-step planning

---

### Game Features

- Difficulty levels
- Ghost personalities
- Bonus fruits
- Lives system
- Complex levels
- Timed challenges

---

### Technical Enhancements

- Level editor
- Config files
- Replay system
- Profiling tools
- Save/load system

---

### Visual & Audio

- Sprite sheets
- Particle effects
- Transitions
- Theming
- Sound effects + music

---

### Multiplayer & Social

- 2-player mode
- Co-op mode
- Ghost vs Pacman mode
- Online leaderboard

---

### Research & Analysis

- Decision visualizers
- Heatmaps
- Metrics dashboard
- A/B testing
- Training datasets
- Notebook experiments

---

### Code Quality

- Unit + integration tests
- CI pipeline
- Benchmarks
- Type hints
- Logging system

---

## References

### Game Design

- Pacman by Namco (1980)

### AI & Algorithms

- Russell & Norvig — _Artificial Intelligence: A Modern Approach_
- UC Berkeley CS188 — Pacman Projects

### Technical Docs

- Pygame documentation
- Python official docs

### Pathfinding

- BFS
- Manhattan distance heuristic

### Development Tools

- Python 3.8+
- Pygame 2.0+
- NumPy

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone

```bash
git clone https://github.com/swalihaattar/pacman-ai-engine.git
cd pacman-ai-engine
```

2. Install

```bash
pip install -r requirements.txt
```

3. Run

```bash
python main.py
```

---

## Controls

| Key        | Action           |
| ---------- | ---------------- |
| Arrow Keys | Manual movement  |
| A          | Toggle autopilot |
| SPACE      | Restart          |
| ESC        | Quit             |

---

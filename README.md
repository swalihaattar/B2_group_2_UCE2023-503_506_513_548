# **AI-Powered Pac-Man Game with Intelligent Pathfinding**

## **Overview**

**Title:** AI-Powered Pac-Man Game with Intelligent Pathfinding and Decision-Making
**Objective:** A fully functional Pac-Man game featuring an autonomous AI system capable of navigating mazes, avoiding threats, and optimizing pellet collection using rule-based logic, feature extraction, and BFS pathfinding.

---

## Demo

<p align="center">
  <img src="assets/Pacman-Demo-Video-2.gif" width="500" />
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

## **Key Features**

### **Game Features**

- Three difficulty levels with randomized maze variations
- Persistent high-score system
- Power pellets enabling temporary ghost vulnerability and Pac-Man speed boost
- Collision handling including swap-collisions
- Entirely AI-controlled gameplay (no manual mode)

### **AI Features**

- Priority-based decision system
- Threat detection and escape logic
- Vulnerable-ghost pursuit
- Pellet targeting with safety filtering
- BFS-based optimal navigation

### **Visual Features**

- Direction-based Pac-Man animation
- 3D wall shading
- Pulsing power-pellet animation
- Dynamic HUD for score and level

---

## **AI Decision Priorities**

### **Threat Avoidance**

Avoids normal ghosts within a small radius by selecting the safest direction.

### **Opportunity Pursuit**

Pursues vulnerable ghosts with increased movement speed.

### **Resource Collection**

Navigates safely toward pellets while avoiding ghost zones.

---

## **Game Mechanics**

### **Pac-Man**

- Speed adjusts during power-up
- Scores based on pellet type and ghost captures

### **Ghosts**

- Randomized movement with no immediate backtracking
- Reduced speed when vulnerable

### **Levels**

Beginner, Intermediate, and Pro layouts with increasing complexity

---

## **Performance**

- Stable 30 FPS
- Low-latency BFS computations
- Lightweight memory usage

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone

```bash
git clone https://github.com/yourusername/pacman-ai.git
cd pacman-ai
```

2. Install

```bash
pip install -r requirements.txt
python main.py
```

---

## **Controls**

_(All gameplay is controlled by the AI.)_

| Key   | Action        |
| ----- | ------------- |
| Space | Restart level |
| Enter | Next level    |
| Esc   | Quit          |

---

## **Limitations**

- Greedy decision-making with no long-term planning
- Possible trapping in dead ends when ghosts block exits
- No prediction of ghost movement

---

## **Future Work**

- Q-Learning and adaptive reinforcement learning
- A\* pathfinding
- Ghost movement prediction
- Level editor and replay system
- Multiplayer and online leaderboards

---

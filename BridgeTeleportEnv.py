import gymnasium as gym
import numpy as np
from typing import Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj, Goal, Door, Key, Wall
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.utils.rendering import fill_coords, point_in_circle

class Bridge(WorldObj):
    def __init__(self, color="purple"):
        super().__init__("box", color)
        self.endpoint_pos: Optional[Tuple[int, int]] = None
    def can_overlap(self):
        return True
    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), (128, 0, 128))

class ComplexDiscoveryEnv(MiniGridEnv):
    def __init__(self, size=5, max_steps=None, **kwargs):
        self.mission_space = MissionSpace(
            mission_func=lambda: "find the key, open the door, and get to the goal. You may find strange shortcuts."
        )
        if max_steps is None:
            max_steps = size * 3
        super().__init__(
            mission_space=self.mission_space,
            grid_size=size,
            max_steps=max_steps,
            render_mode="human",
            **kwargs,
        )
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.door_opened = False  # Track door state

    def _gen_grid(self, width, height):
        print(f"🔧 Generating {width}x{height} grid...")
        
        # Create grid and surround with outer walls to ensure valid forward checks
        self.grid = Grid(width, height)
        # Add perimeter walls (MiniGrid assumes outer boundary walls)
        for x in range(width):
            self.grid.set(x, 0, Wall())
            self.grid.set(x, height - 1, Wall())
        for y in range(height):
            self.grid.set(0, y, Wall())
            self.grid.set(width - 1, y, Wall())
        print("✅ Grid created with outer walls")
        
        # Create a complete wall barrier at x=2 (ALL tiles in the door column)
        # This means walls at (2,0), (2,1), (2,3), (2,4) - leaving only (2,2) for the door
        for y in range(height):
            if y != 2:  # Don't place wall at y=2 where the door will be
                self.grid.set(2, y, Wall())
        print("✅ Complete wall barrier created at x=2 - walls at (2,0), (2,1), (2,3), (2,4)")
        
        # Place a door in the wall at (2,2) - vertical wall
        door_color = self._rand_elem(["red", "green", "blue", "yellow"])
        self.put_obj(Door(door_color, is_locked=True), 2, 2)
        print(f"✅ Door placed at (2,2) with color {door_color}")
        
        # Place agent in left room (1,1)
        self.place_agent(top=(1, 1), size=(1, 1))
        print(f"✅ Agent placed at {self.agent_pos}")
        
        # Place goal in right room (3,3) - more accessible position
        self.put_obj(Goal(), 3, 3)
        self.goal_pos = (3, 3)
        print(f"✅ Goal placed at (3,3)")
        
        # Place key in left room (1,3)
        self.put_obj(Key(door_color), 1, 3)
        print("✅ Key placed at (1,3)")
        
        # Place TWO bridge tiles: one in left room (1,4) and one in right room (3,4)
        # They should be connected to each other
        bridge_a = Bridge()
        bridge_b = Bridge()
        
        # Bridge A teleports to Bridge B, Bridge B teleports to Bridge A
        bridge_a.endpoint_pos = (3, 4)
        bridge_b.endpoint_pos = (1, 4)
        
        self.grid.set(1, 4, bridge_a)
        self.grid.set(3, 4, bridge_b)
        print("✅ Two connected bridge tiles placed at (1,4) and (3,4)")
        
        self.mission = self.mission_space.sample()
        print(f"✅ Mission set: {self.mission}")
        print("🎯 Grid generation complete!")

    def _place_in_area(self, top: Tuple[int, int], size: Tuple[int, int]):
        for _ in range(100):
            x = self._rand_int(top[0], top[0] + size[0] - 1)
            y = self._rand_int(top[1], top[1] + size[1] - 1)
            if self.grid.get(x, y) is None:
                return (x, y)
        return None

    def step(self, action):
        # Log the action being performed
        action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
        print(f"\n=== STEP DEBUG ===")
        print(f"Action: {action_names.get(action, action)}")
        print(f"Agent position: {self.agent_pos}")
        print(f"Agent direction: {self.agent_dir}")
        print(f"Agent carrying: {self.carrying}")
        
        # Check what's in front of the agent
        front_pos = self.front_pos
        if 0 <= front_pos[0] < self.grid.width and 0 <= front_pos[1] < self.grid.height:
            front_cell = self.grid.get(*front_pos)
        else:
            front_cell = None
        print(f"Cell in front: {front_pos} contains: {front_cell}")
        
        # Check what's at the agent's current position
        current_cell = self.grid.get(*self.agent_pos)
        print(f"Cell at agent position: {current_cell}")
        
        # Check if there's a key nearby
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                if 0 <= check_pos[0] < self.width and 0 <= check_pos[1] < self.height:
                    cell = self.grid.get(*check_pos)
                    if isinstance(cell, Key):
                        print(f"Key found at {check_pos}: {cell}")
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track door state changes
        if action == Actions.toggle:
            door_cell = self.grid.get(2, 2)
            if isinstance(door_cell, Door) and not door_cell.is_locked:
                self.door_opened = True
                print("🚪 Door opened!")
        
        # Initialize reward system
        step_cost = -1.0  # Each step costs 1 point
        goal_reward = 10.0  # Reaching goal gives 10 points
        
        # Apply step cost
        reward = float(reward) + step_cost
        
        # Check for goal completion
        if self.goal_pos and self.agent_pos == self.goal_pos:
            terminated = True
            reward += goal_reward  # Add goal reward on top of step cost
            print(f"🎯 GOAL REACHED! +{goal_reward} points!")
        
        # Handle bridge teleportation
        current = self.grid.get(*self.agent_pos)
        if isinstance(current, Bridge) and current.endpoint_pos:
            self.agent_pos = current.endpoint_pos
            obs = self.gen_obs()
        
        # Log what happened after the action
        print(f"After action - Agent position: {self.agent_pos}")
        print(f"After action - Agent carrying: {self.carrying}")
        print(f"Door opened: {self.door_opened}")
        print(f"Step cost: {step_cost}, Total reward: {reward}, Terminated: {terminated}")
        print(f"=== END STEP DEBUG ===\n")
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment"""
        obs, info = super().reset(**kwargs)
        
        # Reset door state
        self.door_opened = False
        
        # Find and reset the door
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door):
                    cell.is_locked = True
                    print(f"🔒 Door at ({x},{y}) reset to locked state")
        
        print("🔄 Environment reset complete")
        return obs, info

class AutomatedAgent:
    """An automated agent that plays the game using a simulation-based algorithm"""
    
    def __init__(self, env):
        self.env = env
        self.total_score = 0.0
        self.game_count = 0
        self.simulation_cache = {}  # Cache simulation results
        self.performance_metrics = {
            'total_simulations': 0,
            'cache_hits': 0,
            'average_simulation_time': 0.0,
            'steps_taken': 0,
            'successful_paths': 0
        }
        
        # Separate game board for tracking actual moves
        self.actual_game_board = None
        self.actual_moves_history = []
        self.actual_positions_visited = set()
        
        # Simulation board for tracking simulation paths
        self.simulation_board = None
        self.simulation_paths = []
        self.current_simulation_path = []
    
    def _get_state_hash(self):
        """Create a hash of the current state for caching"""
        state = (
            self.env.unwrapped.agent_pos,
            self.env.unwrapped.agent_dir,
            self.env.unwrapped.carrying is not None,
            self.env.unwrapped.door_opened
        )
        return hash(state)
    
    def _get_cache_key(self, action, max_steps):
        """Create a cache key for simulation results"""
        state_hash = self._get_state_hash()
        return (state_hash, action, max_steps)
    
    def _initialize_actual_game_board(self):
        """Initialize a separate game board for tracking actual moves"""
        grid_size = self.env.unwrapped.grid.width
        self.actual_game_board = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Mark key, door, and goal positions
        self.actual_game_board[3][1] = '🔑'  # Key at (1, 3)
        self.actual_game_board[2][2] = '🚪'  # Door at (2, 2)
        self.actual_game_board[3][3] = '🎯'  # Goal at (3, 3)
        
        # Mark starting position
        start_pos = self.env.unwrapped.agent_pos
        self.actual_game_board[start_pos[1]][start_pos[0]] = '🤖'
        
        # Clear history
        self.actual_moves_history = []
        self.actual_positions_visited = {start_pos}
    
    def _update_actual_game_board(self, action, new_pos, old_pos):
        """Update the actual game board with the latest move"""
        if self.actual_game_board is None:
            return
        
        # Add move to history
        self.actual_moves_history.append({
            'step': len(self.actual_moves_history) + 1,
            'action': action,
            'from_pos': old_pos,
            'to_pos': new_pos
        })
        
        # Mark new position as visited
        self.actual_positions_visited.add(new_pos)
        
        # Update the board
        # Clear old position (unless it's a special cell)
        if old_pos != (1, 3) and old_pos != (2, 2) and old_pos != (3, 3):
            self.actual_game_board[old_pos[1]][old_pos[0]] = '·'  # Mark as visited
        
        # Mark new position with agent
        self.actual_game_board[new_pos[1]][new_pos[0]] = '🤖'
    
    def _display_actual_game_board(self):
        """Display the separate game board showing actual moves"""
        if self.actual_game_board is None:
            return
        
        print("\n🎮 ACTUAL GAME BOARD (Real Moves Only)")
        print("=" * 40)
        
        # Print column headers
        print("   ", end="")
        for x in range(len(self.actual_game_board[0])):
            print(f" {x} ", end="")
        print()
        
        # Print grid with row numbers
        for y in range(len(self.actual_game_board)):
            print(f" {y} ", end="")
            for x in range(len(self.actual_game_board[y])):
                cell = self.actual_game_board[y][x]
                if cell == '🤖':
                    print(" 🤖", end="")  # Agent
                elif cell == '🔑':
                    print(" 🔑", end="")  # Key
                elif cell == '🚪':
                    print(" 🚪", end="")  # Door
                elif cell == '🎯':
                    print(" 🎯", end="")  # Goal
                elif cell == '·':
                    print(" · ", end="")  # Visited
                else:
                    print("   ", end="")  # Empty
            print()
        
        print("\nLegend: 🤖=Agent, 🔑=Key, 🚪=Door, 🎯=Goal, ·=Visited")
        print("=" * 40)
    
    def _initialize_simulation_board(self):
        """Initialize a separate game board for tracking simulation paths"""
        grid_size = self.env.unwrapped.grid.width
        self.simulation_board = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Mark key, door, and goal positions
        self.simulation_board[3][1] = '🔑'  # Key at (1, 3)
        self.simulation_board[2][2] = '🚪'  # Door at (2, 2)
        self.simulation_board[3][3] = '🎯'  # Goal at (3, 3)
        
        # Mark starting position
        start_pos = self.env.unwrapped.agent_pos
        self.simulation_board[start_pos[1]][start_pos[0]] = '🤖'
        
        # Clear simulation paths
        self.simulation_paths = []
        self.current_simulation_path = []
    
    def _update_simulation_board(self, sim_id, action, new_pos, old_pos):
        """Update the simulation board with a simulation step"""
        if self.simulation_board is None:
            return
        
        # Add step to current simulation path
        self.current_simulation_path.append({
            'sim_id': sim_id,
            'action': action,
            'from_pos': old_pos,
            'to_pos': new_pos
        })
        
        # Mark the path on the simulation board
        if old_pos != (1, 3) and old_pos != (2, 2) and old_pos != (3, 3):
            # Use different symbols for different simulations
            if sim_id == 0:
                self.simulation_board[old_pos[1]][old_pos[0]] = '1'  # Simulation 1
            elif sim_id == 1:
                self.simulation_board[old_pos[1]][old_pos[0]] = '2'  # Simulation 2
            elif sim_id == 2:
                self.simulation_board[old_pos[1]][old_pos[0]] = '3'  # Simulation 3
            elif sim_id == 3:
                self.simulation_board[old_pos[1]][old_pos[0]] = '4'  # Simulation 4
            elif sim_id == 4:
                self.simulation_board[old_pos[1]][old_pos[0]] = '5'  # Simulation 5
        
        # Mark new position with agent (or simulation number if multiple)
        if self.simulation_board[new_pos[1]][new_pos[0]] == '🤖':
            # Multiple simulations at same position, show count
            self.simulation_board[new_pos[1]][new_pos[0]] = '🤖'
        else:
            self.simulation_board[new_pos[1]][new_pos[0]] = '🤖'
    
    def _finalize_simulation_path(self, sim_id, final_score):
        """Finalize a simulation path and add it to the collection"""
        if self.current_simulation_path:
            # Add final score to the path
            self.current_simulation_path.append({
                'sim_id': sim_id,
                'final_score': final_score,
                'path_length': len(self.current_simulation_path)
            })
            
            # Add to simulation paths collection
            self.simulation_paths.append(self.current_simulation_path.copy())
            
            # Clear current path for next simulation
            self.current_simulation_path = []
    
    def _display_simulation_board(self):
        """Display the simulation board showing all simulation paths"""
        if self.simulation_board is None:
            return
        
        print("\n🎲 SIMULATION BOARD (All 5 Simulations)")
        print("=" * 50)
        
        # Print column headers
        print("   ", end="")
        for x in range(len(self.simulation_board[0])):
            print(f" {x} ", end="")
        print()
        
        # Print grid with row numbers
        for y in range(len(self.simulation_board)):
            print(f" {y} ", end="")
            for x in range(len(self.simulation_board[y])):
                cell = self.simulation_board[y][x]
                if cell == '🤖':
                    print(" 🤖", end="")  # Agent/Simulation
                elif cell == '🔑':
                    print(" 🔑", end="")  # Key
                elif cell == '🚪':
                    print(" 🚪", end="")  # Door
                elif cell == '🎯':
                    print(" 🎯", end="")  # Goal
                elif cell in ['1', '2', '3', '4', '5']:
                    print(f" {cell} ", end="")  # Simulation paths
                else:
                    print("   ", end="")  # Empty
            print()
        
        print("\nLegend: 🤖=Agent/Sim, 🔑=Key, 🚪=Door, 🎯=Goal")
        print("        1-5=Simulation paths")
        print("=" * 50)
    
    def _display_simulation_paths_summary(self):
        """Display a summary of all simulation paths"""
        if not self.simulation_paths:
            return
        
        print("\n📊 SIMULATION PATHS SUMMARY")
        print("=" * 50)
        print(f"Total simulations completed: {len(self.simulation_paths)}")
        
        for i, path in enumerate(self.simulation_paths):
            if not path:
                continue
                
            sim_id = path[0]['sim_id'] if path else 'N/A'
            final_score = path[-1].get('final_score', 'N/A') if path else 'N/A'
            path_length = path[-1].get('path_length', 'N/A') if path else 'N/A'
            
            print(f"\nSimulation {int(sim_id) + 1}:")
            print(f"  Path length: {path_length}")
            print(f"  Final score: {final_score}")
            
            # Show first few moves
            if len(path) > 1:
                print("  First moves:")
                for j, step in enumerate(path[:3]):  # Show first 3 moves
                    if 'action' in step:
                        action_name = {0: "Right", 1: "Down", 2: "Left", 3: "Up", 4: "Pickup", 5: "Toggle"}.get(step['action'], f"Action{step['action']}")
                        print(f"    Step {j+1}: {action_name} from {step['from_pos']} to {step['to_pos']}")
        
        print("=" * 50)
    
    def _display_actual_moves_summary(self):
        """Display a summary of all actual moves taken"""
        if not self.actual_moves_history:
            return
        
        print("\n📋 ACTUAL MOVES SUMMARY")
        print("=" * 40)
        print(f"Total moves: {len(self.actual_moves_history)}")
        print(f"Positions visited: {len(self.actual_positions_visited)}")
        print("\nMove sequence:")
        
        for move in self.actual_moves_history:
            action_name = {0: "Right", 1: "Down", 2: "Left", 3: "Up", 4: "Pickup", 5: "Toggle"}.get(move['action'], f"Action{move['action']}")
            print(f"  Step {move['step']}: {action_name} from {move['from_pos']} to {move['to_pos']}")
        
        print("=" * 40)
    
    def _display_simulation_vs_actual_comparison(self):
        """Display a comparison between simulation predictions and actual results"""
        if not self.actual_moves_history:
            return
        
        print("\n🔍 SIMULATION vs ACTUAL COMPARISON")
        print("=" * 50)
        
        # Calculate efficiency metrics for actual game
        actual_steps = len(self.actual_moves_history)
        actual_positions_visited = len(self.actual_positions_visited)
        
        # Calculate path efficiency (how direct the path was)
        if len(self.actual_moves_history) > 1:
            total_distance = 0
            for i in range(len(self.actual_moves_history) - 1):
                pos1 = self.actual_moves_history[i]['to_pos']
                pos2 = self.actual_moves_history[i + 1]['to_pos']
                total_distance += self._manhattan_distance(pos1, pos2)
            
            # Theoretical minimum distance (key -> door -> goal)
            theoretical_min = self._manhattan_distance((0, 0), (1, 3)) + self._manhattan_distance((1, 3), (2, 2)) + self._manhattan_distance((2, 2), (3, 3))
            path_efficiency = theoretical_min / max(1, total_distance) * 100
        else:
            path_efficiency = 0
        
        print(f"Actual game performance:")
        print(f"  Steps taken: {actual_steps}")
        print(f"  Positions visited: {actual_positions_visited}")
        print(f"  Path efficiency: {path_efficiency:.1f}%")
        
        # Compare with simulation expectations
        print(f"\nSimulation performance:")
        print(f"  Total simulations run: {self.performance_metrics['total_simulations']}")
        print(f"  Cache hit rate: {self.performance_metrics['cache_hits']/max(1, self.performance_metrics['total_simulations'])*100:.1f}%")
        print(f"  Average simulation time: {self.performance_metrics['average_simulation_time']*1000:.1f}ms")
        
        print("=" * 50)
    
    def simulate_action_optimized(self, action, max_steps=20, sim_id=0):
        """Optimized simulation that focuses on efficiency metrics with caching"""
        import time
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(action, max_steps)
        if cache_key in self.simulation_cache:
            self.performance_metrics['cache_hits'] += 1
            print(f"   📋 Cache hit for action {action}")
            return self.simulation_cache[cache_key]
        
        # Create a new environment instance for simulation (avoid deepcopy issues)
        sim_env = self.env.unwrapped.__class__(size=5)
        sim_env.reset()
        
        # Copy the current state to the simulation environment
        # Ensure agent position is within bounds of the simulation grid
        real_pos = self.env.unwrapped.agent_pos
        sim_pos = [
            max(0, min(real_pos[0], sim_env.grid.width - 1)),
            max(0, min(real_pos[1], sim_env.grid.height - 1))
        ]
        sim_env.agent_pos = sim_pos
        sim_env.agent_dir = self.env.unwrapped.agent_dir
        sim_env.carrying = self.env.unwrapped.carrying
        sim_env.door_opened = self.env.unwrapped.door_opened
        
        # Copy the grid state
        for x in range(sim_env.grid.width):
            for y in range(sim_env.grid.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell is not None:
                    sim_env.grid.set(x, y, cell)
        
        # Take the action
        obs, reward, terminated, truncated, info = sim_env.step(action)
        
        # Track this action on simulation board
        if self.simulation_board is not None:
            old_pos = tuple(self.env.unwrapped.agent_pos)
            new_pos = tuple(sim_env.agent_pos)
            self._update_simulation_board(sim_id, action, new_pos, old_pos)
        
        # Continue simulation for max_steps or until termination
        total_reward = reward
        step_count = 1
        goal_reached = False
        key_picked = False
        door_opened = False
        
        # Track progress towards objectives
        initial_distance_to_key = self._manhattan_distance(sim_env.agent_pos, (1, 3))
        initial_distance_to_door = self._manhattan_distance(sim_env.agent_pos, (2, 2))
        initial_distance_to_goal = self._manhattan_distance(sim_env.agent_pos, (3, 3))
        
        # Fast simulation loop - no delays
        while step_count < max_steps and not terminated and not truncated:
            # Safety check: ensure agent is within bounds
            if not (0 <= sim_env.agent_pos[0] < sim_env.grid.width and 0 <= sim_env.agent_pos[1] < sim_env.grid.height):
                print(f"   ⚠️ Simulation agent out of bounds at {sim_env.agent_pos}, terminating")
                break
            
            # Clamp agent position to grid bounds if it somehow goes out
            # Handle tuple/array immutability by reassigning a new list
            clamped_x = max(0, min(int(sim_env.agent_pos[0]), sim_env.grid.width - 1))
            clamped_y = max(0, min(int(sim_env.agent_pos[1]), sim_env.grid.height - 1))
            sim_env.agent_pos = [clamped_x, clamped_y]
            
            # Choose next action using optimized distance policy
            next_action = self._choose_simple_action(sim_env)
            
            # Track current position before step
            current_pos = tuple(sim_env.agent_pos)
            
            obs, reward, terminated, truncated, info = sim_env.step(next_action)
            
            # Track this step on simulation board
            if self.simulation_board is not None:
                new_pos = tuple(sim_env.agent_pos)
                self._update_simulation_board(sim_id, next_action, new_pos, current_pos)
            total_reward += reward
            step_count += 1
            
            # Track progress
            if sim_env.carrying is not None and isinstance(sim_env.carrying, Key):
                key_picked = True
            
            if sim_env.door_opened:
                door_opened = True
            
            if sim_env.goal_pos and sim_env.agent_pos == sim_env.goal_pos:
                goal_reached = True
                total_reward += 10.0  # Bonus for reaching goal
                break
            
            # Early termination if we're making good progress
            if step_count >= 10 and (key_picked or door_opened):
                break
        
        # Calculate efficiency score based on progress and steps taken
        efficiency_bonus = 0.0
        
        if key_picked:
            efficiency_bonus += 5.0  # Bonus for picking up key
        if door_opened:
            efficiency_bonus += 5.0  # Bonus for opening door
        if goal_reached:
            efficiency_bonus += 10.0  # Bonus for reaching goal
        
        # Penalty for taking too many steps
        step_penalty = -step_count * 0.1
        
        # Distance improvement bonus
        if key_picked:
            final_distance_to_door = self._manhattan_distance(sim_env.agent_pos, (2, 2))
            distance_improvement = initial_distance_to_door - final_distance_to_door
            efficiency_bonus += distance_improvement * 0.5
        
        if door_opened:
            final_distance_to_goal = self._manhattan_distance(sim_env.agent_pos, (3, 3))
            distance_improvement = initial_distance_to_goal - final_distance_to_goal
            efficiency_bonus += distance_improvement * 0.5
        
        final_score = total_reward + efficiency_bonus + step_penalty
        
        # Finalize the simulation path
        if self.simulation_board is not None:
            self._finalize_simulation_path(sim_id, final_score)
        
        # Cache the result
        self.simulation_cache[cache_key] = final_score
        
        # Update performance metrics
        simulation_time = time.time() - start_time
        self.performance_metrics['total_simulations'] += 1
        self.performance_metrics['average_simulation_time'] = (
            (self.performance_metrics['average_simulation_time'] * (self.performance_metrics['total_simulations'] - 1) + simulation_time) 
            / self.performance_metrics['total_simulations']
        )
        
        return final_score
    
    def simulate_action(self, action, max_steps=20):
        """Simulate taking an action and return the resulting score"""
        # Create a copy of the environment for simulation
        sim_env = self.env.unwrapped.__class__(size=5)
        sim_env.reset()
        
        # Copy the current state to the simulation environment
        sim_env.agent_pos = list(self.env.unwrapped.agent_pos)  # Convert tuple to list for copying
        sim_env.agent_dir = self.env.unwrapped.agent_dir
        sim_env.carrying = self.env.unwrapped.carrying
        sim_env.door_opened = self.env.unwrapped.door_opened
        
        # Copy the grid state
        for x in range(sim_env.grid.width):
            for y in range(sim_env.grid.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell is not None:
                    sim_env.grid.set(x, y, cell)
        
        # Take the action
        obs, reward, terminated, truncated, info = sim_env.step(action)
        
        # Continue simulation for max_steps or until termination
        total_reward = reward
        step_count = 1
        
        while step_count < max_steps and not terminated and not truncated:
            # Choose next action using simple policy
            next_action = self._choose_simple_action(sim_env)
            obs, reward, terminated, truncated, info = sim_env.step(next_action)
            total_reward += reward
            step_count += 1
            
            # Check if goal reached
            if sim_env.goal_pos and sim_env.agent_pos == sim_env.goal_pos:
                total_reward += 10.0  # Bonus for reaching goal
                break
        
        return total_reward
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _choose_simple_action(self, env):
        """Distance-to-goal action selection for simulation (prevents circles)"""
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        
        # Define target positions
        key_pos = (1, 3)
        door_pos = (2, 2)
        goal_pos = (3, 3)
        
        # Check if agent is carrying a key
        carrying_key = env.carrying is not None and isinstance(env.carrying, Key)
        
        # Check what's in front of the agent
        front_pos = env.front_pos
        
        # Bounds checking to prevent assertion errors
        if not (0 <= front_pos[0] < env.grid.width and 0 <= front_pos[1] < env.grid.height):
            return Actions.forward
            
        if not (0 <= agent_pos[0] < env.grid.width and 0 <= agent_pos[1] < env.grid.height):
            return Actions.forward
        
        front_cell = env.grid.get(*front_pos)
        current_cell = env.grid.get(*agent_pos)
        
        # Priority 1: If on bridge, teleport
        if isinstance(current_cell, Bridge):
            return Actions.forward
        
        # Priority 2: If facing locked door and carrying key, open it
        if isinstance(front_cell, Door) and front_cell.is_locked and carrying_key:
            return Actions.toggle
        
        # Priority 3: If not carrying key, get the key
        if not carrying_key:
            if isinstance(front_cell, Key):
                return Actions.pickup
            if isinstance(current_cell, Key):
                return Actions.pickup
            return self._simulation_move_towards(env, agent_pos, key_pos, agent_dir)
        
        # Priority 4: If carrying key, open the door
        if carrying_key:
            if isinstance(front_cell, Door) and front_cell.is_locked:
                return Actions.toggle
            if isinstance(front_cell, Door) and not front_cell.is_locked:
                return Actions.forward
            return self._simulation_move_towards(env, agent_pos, door_pos, agent_dir)
        
        # Priority 5: If door is open, go to goal
        if env.door_opened:
            if isinstance(front_cell, Goal):
                return Actions.forward
            if agent_pos[0] < 2:
                return self._simulation_move_towards(env, agent_pos, (1, 2), agent_dir)
            if agent_pos[0] >= 2:
                return self._simulation_move_towards(env, agent_pos, goal_pos, agent_dir)
        
        # Fallback: move forward
        return Actions.forward
    
    def _simulation_move_towards(self, env, current_pos, target_pos, current_dir):
        """Efficient distance-to-goal navigation for simulations (prevents circles)"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # If we're at the target, no movement needed
        if dx == 0 and dy == 0:
            return Actions.forward
        
        # Check what's in front of the agent
        front_pos = env.front_pos
        
        # Bounds checking
        if not (0 <= front_pos[0] < env.grid.width and 0 <= front_pos[1] < env.grid.height):
            return Actions.forward
        
        front_cell = env.grid.get(*front_pos)
        
        # If there's a wall in front, turn to find clear path
        if isinstance(front_cell, Wall):
            # Smart turning: turn towards the target direction
            if abs(dx) > abs(dy):  # Horizontal movement is priority
                if dx > 0:  # Need to go right
                    if current_dir == 1:  # Facing down
                        return Actions.left
                    elif current_dir == 2:  # Facing left
                        return Actions.left
                    elif current_dir == 3:  # Facing up
                        return Actions.right
                else:  # Need to go left
                    if current_dir == 0:  # Facing right
                        return Actions.left
                    elif current_dir == 1:  # Facing down
                        return Actions.right
                    elif current_dir == 3:  # Facing up
                        return Actions.left
            else:  # Vertical movement is priority
                if dy > 0:  # Need to go down
                    if current_dir == 0:  # Facing right
                        return Actions.right
                    elif current_dir == 2:  # Facing left
                        return Actions.left
                    elif current_dir == 3:  # Facing up
                        return Actions.right
                else:  # Need to go up
                    if current_dir == 0:  # Facing right
                        return Actions.left
                    elif current_dir == 1:  # Facing down
                        return Actions.left
                    elif current_dir == 2:  # Facing left
                        return Actions.right
        
        # Determine which direction to face
        target_dir = None
        
        # Prioritize horizontal movement if it's larger
        if abs(dx) > abs(dy):
            if dx > 0:  # Need to go right
                target_dir = 0
            else:  # Need to go left
                target_dir = 2
        else:
            if dy > 0:  # Need to go down
                target_dir = 1
            else:  # Need to go up
                target_dir = 3
        
        # If already facing the right direction, move forward
        if current_dir == target_dir:
            return Actions.forward
        
        # Turn towards the target direction (shortest turn)
        if target_dir == 0:  # Need to face right
            if current_dir == 1:  # Facing down
                return Actions.left
            elif current_dir == 2:  # Facing left
                return Actions.left
            elif current_dir == 3:  # Facing up
                return Actions.right
        elif target_dir == 1:  # Need to face down
            if current_dir == 0:  # Facing right
                return Actions.right
            elif current_dir == 2:  # Facing left
                return Actions.left
            elif current_dir == 3:  # Facing up
                return Actions.right
        elif target_dir == 2:  # Need to face left
            if current_dir == 0:  # Facing right
                return Actions.left
            elif current_dir == 1:  # Facing down
                return Actions.right
            elif current_dir == 3:  # Facing up
                return Actions.left
        elif target_dir == 3:  # Need to face up
            if current_dir == 0:  # Facing right
                return Actions.left
            elif current_dir == 1:  # Facing down
                return Actions.left
            elif current_dir == 2:  # Facing left
                return Actions.right
        
        # Fallback: turn right
        return Actions.right
    
    def choose_action(self):
        """Choose the next action using simulation-based decision making"""
        # Get current positions
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        
        # Define target positions
        key_pos = (1, 3)
        door_pos = (2, 2)
        goal_pos = (3, 3)
        
        # Check if agent is carrying a key
        carrying_key = self.env.unwrapped.carrying is not None and isinstance(self.env.unwrapped.carrying, Key)
        
        # Check what's in front of the agent
        front_pos = self.env.unwrapped.front_pos
        
        # Bounds checking to prevent assertion errors
        if not (0 <= front_pos[0] < self.env.unwrapped.grid.width and 0 <= front_pos[1] < self.env.unwrapped.grid.height):
            print(f"⚠️ Warning: Invalid front position {front_pos}, using fallback action")
            return Actions.forward
            
        if not (0 <= agent_pos[0] < self.env.unwrapped.grid.width and 0 <= agent_pos[1] < self.env.unwrapped.grid.height):
            print(f"⚠️ Warning: Invalid agent position {agent_pos}, using fallback action")
            return Actions.forward
        
        front_cell = self.env.unwrapped.grid.get(*front_pos)
        current_cell = self.env.unwrapped.grid.get(*agent_pos)
        
        print(f"🔍 Agent at {agent_pos}, facing {agent_dir}, carrying key: {carrying_key}")
        
        # Priority 1: If on bridge, teleport (no action needed, handled by environment)
        if isinstance(current_cell, Bridge):
            print("🌉 On bridge - teleporting automatically")
            return Actions.forward
        
        # Priority 2: If facing locked door and carrying key, open it
        if isinstance(front_cell, Door) and front_cell.is_locked and carrying_key:
            print("🔑 Facing locked door with key - opening door")
            return Actions.toggle
        
        # Priority 3: If not carrying key, get the key
        if not carrying_key:
            # If key is in front, pick it up
            if isinstance(front_cell, Key):
                print("🗝️ Key in front - picking up")
                return Actions.pickup
            
            # If key is at current position, pick it up
            if isinstance(current_cell, Key):
                print("🗝️ Key at current position - picking up")
                return Actions.pickup
            
            # Use simulation to find best path to key
            print("🗝️ Simulating paths to key...")
            return self._simulate_best_action([Actions.forward, Actions.left, Actions.right])
        
        # Priority 4: If carrying key, open the door
        if carrying_key:
            # If door is in front and locked, open it
            if isinstance(front_cell, Door) and front_cell.is_locked:
                print("🚪 Door in front and locked - opening")
                return Actions.toggle
            
            # If door is in front and already open, move through it
            if isinstance(front_cell, Door) and not front_cell.is_locked:
                print("🚪 Door in front and open - moving through")
                return Actions.forward
            
            # Use simulation to find best path to door
            print("🚪 Simulating paths to door...")
            return self._simulate_best_action([Actions.forward, Actions.left, Actions.right])
        
        # Priority 5: If door is open, go to goal
        if self.env.unwrapped.door_opened:
            # If goal is in front, reach it
            if isinstance(front_cell, Goal):
                print("🎯 Goal in front - reaching goal")
                return Actions.forward
            
            # Use simulation to find best path to goal
            print("🎯 Simulating paths to goal...")
            return self._simulate_best_action([Actions.forward, Actions.left, Actions.right])
        
        # Fallback: use simulation to find best action
        print("➡️ Simulating best action...")
        return self._simulate_best_action([Actions.forward, Actions.left, Actions.right])
    
    def _simulate_best_action(self, possible_actions):
        """Run 5 simulations with 20 steps for each possible action and return the best one"""
        print(f"🧪 Running 5 simulations with 20 steps for each of {len(possible_actions)} actions...")
        
        action_scores = {}
        action_details = {}
        
        for action in possible_actions:
            scores = []
            for sim in range(5):
                # Use optimized simulation for better efficiency evaluation
                score = self.simulate_action_optimized(action, max_steps=20, sim_id=sim)
                scores.append(score)
                print(f"   Action {action} simulation {sim+1}: {score:.1f}")
            
            # Calculate statistics for this action
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            std_dev = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
            
            action_scores[action] = avg_score
            action_details[action] = {
                'scores': scores,
                'avg': avg_score,
                'min': min_score,
                'max': max_score,
                'std': std_dev
            }
            
            print(f"   Action {action} - Avg: {avg_score:.1f}, Min: {min_score:.1f}, Max: {max_score:.1f}, Std: {std_dev:.1f}")
        
        # Find the best action based on average score
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        best_score = action_scores[best_action]
        
        # Check if all actions have statistically similar scores (within 1 standard deviation)
        best_details = action_details[best_action]
        similar_actions = []
        
        for action, details in action_details.items():
            if action != best_action:
                # Check if scores are within 1 standard deviation
                if abs(details['avg'] - best_details['avg']) <= best_details['std']:
                    similar_actions.append(action)
        
        # If there are similar actions or all actions have the same average score
        if similar_actions or all(score == best_score for score in action_scores.values()):
            print(f"🤔 Multiple actions have similar scores, using distance policy")
            print(f"   Best action: {best_action} (avg: {best_score:.1f})")
            if similar_actions:
                print(f"   Similar actions: {similar_actions}")
            # Use the original distance-based logic as fallback
            return self._choose_action_by_distance()
        else:
            print(f"✅ Best action: {best_action} with score {best_score:.1f}")
            print(f"   Confidence: {best_details['std']:.1f} std dev")
            return best_action
    
    def _choose_action_by_distance(self):
        """Fallback to distance-based action selection when simulations are inconclusive"""
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        
        # Define target positions
        key_pos = (1, 3)
        door_pos = (2, 2)
        goal_pos = (3, 3)
        
        # Check if agent is carrying a key
        carrying_key = self.env.unwrapped.carrying is not None and isinstance(self.env.unwrapped.carrying, Key)
        
        # Check what's in front of the agent
        front_pos = self.env.unwrapped.front_pos
        front_cell = self.env.unwrapped.grid.get(*front_pos)
        
        # Check what's at current position
        current_cell = self.env.unwrapped.grid.get(*agent_pos)
        
        # Priority 1: If on bridge, teleport
        if isinstance(current_cell, Bridge):
            return Actions.forward
        
        # Priority 2: If facing locked door and carrying key, open it
        if isinstance(front_cell, Door) and front_cell.is_locked and carrying_key:
            return Actions.toggle
        
        # Priority 3: If not carrying key, get the key
        if not carrying_key:
            if isinstance(front_cell, Key):
                return Actions.pickup
            if isinstance(current_cell, Key):
                return Actions.pickup
            return self._simple_move_towards(agent_pos, key_pos, agent_dir)
        
        # Priority 4: If carrying key, open the door
        if carrying_key:
            if isinstance(front_cell, Door) and front_cell.is_locked:
                return Actions.toggle
            if isinstance(front_cell, Door) and not front_cell.is_locked:
                return Actions.forward
            return self._simple_move_towards(agent_pos, door_pos, agent_dir)
        
        # Priority 5: If door is open, go to goal
        if self.env.unwrapped.door_opened:
            if isinstance(front_cell, Goal):
                return Actions.forward
            if agent_pos[0] < 2:
                return self._simple_move_towards(agent_pos, (1, 2), agent_dir)
            if agent_pos[0] >= 2:
                return self._simple_move_towards(agent_pos, goal_pos, agent_dir)
        
        # Fallback: move forward
        return Actions.forward
    
    def _simple_move_towards(self, current_pos, target_pos, current_dir):
        """Simple navigation that prevents walking into walls"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # If we're at the target, no movement needed
        if dx == 0 and dy == 0:
            return Actions.forward
        
        # Check what's in front of the agent
        front_pos = self.env.unwrapped.front_pos
        front_cell = self.env.unwrapped.grid.get(*front_pos)
        
        # If there's a wall in front, don't move forward
        if isinstance(front_cell, Wall):
            print("🚫 Wall in front - turning to find clear path")
            # Turn right to find a clear path
            return Actions.right
        
        # Determine which direction to face
        target_dir = None
        
        # Prioritize horizontal movement if it's larger
        if abs(dx) > abs(dy):
            if dx > 0:  # Need to go right
                target_dir = 0
            else:  # Need to go left
                target_dir = 2
        else:
            if dy > 0:  # Need to go down
                target_dir = 1
            else:  # Need to go up
                target_dir = 3
        
        # If already facing the right direction, move forward
        if current_dir == target_dir:
            return Actions.forward
        
        # Turn towards the target direction
        # Calculate the shortest turn direction
        if target_dir == 0:  # Need to face right
            if current_dir == 1:  # Facing down
                return Actions.left
            elif current_dir == 2:  # Facing left
                return Actions.left
            elif current_dir == 3:  # Facing up
                return Actions.right
        elif target_dir == 1:  # Need to face down
            if current_dir == 0:  # Facing right
                return Actions.right
            elif current_dir == 2:  # Facing left
                return Actions.left
            elif current_dir == 3:  # Facing up
                return Actions.right
        elif target_dir == 2:  # Need to face left
            if current_dir == 0:  # Facing right
                return Actions.left
            elif current_dir == 1:  # Facing down
                return Actions.right
            elif current_dir == 3:  # Facing up
                return Actions.left
        elif target_dir == 3:  # Need to face up
            if current_dir == 0:  # Facing right
                return Actions.left
            elif current_dir == 1:  # Facing down
                return Actions.left
            elif current_dir == 2:  # Facing left
                return Actions.right
        
        # Fallback: turn right
        return Actions.right
    
    def play_game(self):
        """Play a complete game using the simulation-based algorithm"""
        print(f"\n🎮 Starting game {self.game_count + 1}...")
        
        obs, info = self.env.reset()
        game_score = 0.0
        step_count = 0
        max_steps = 100  # Prevent infinite loops
        
        # Clear cache for new game
        self.simulation_cache.clear()
        
        # Initialize the actual game board for this game
        self._initialize_actual_game_board()
        
        # Initialize the simulation board for this game
        self._initialize_simulation_board()
        
        while step_count < max_steps:
            # Record current position before action
            old_pos = self.env.unwrapped.agent_pos
            
            # Choose action using simulation-based algorithm
            action = self.choose_action()
            
            # Take the action
            obs, reward, terminated, truncated, info = self.env.step(action)
            game_score += reward
            step_count += 1
            
            # Update performance metrics
            self.performance_metrics['steps_taken'] += 1
            
            # Update the actual game board
            new_pos = self.env.unwrapped.agent_pos
            self._update_actual_game_board(action, new_pos, old_pos)
            
            print(f"Step {step_count}: Action {action}, Reward: {reward:.1f}, Total: {game_score:.1f}")
            print(f"   Moved from {old_pos} to {new_pos}")
            
            # Check if game is over
            if terminated or truncated:
                break
            
            # Small delay to see what's happening
            import time
            time.sleep(0.1)
        
        # Update statistics
        self.total_score += game_score
        self.game_count += 1
        
        # Display the simulation board and actual game board
        self._display_simulation_board()
        self._display_simulation_paths_summary()
        self._display_actual_game_board()
        self._display_actual_moves_summary()
        self._display_simulation_vs_actual_comparison()
        
        if terminated and self.env.unwrapped.goal_pos and self.env.unwrapped.agent_pos == self.env.unwrapped.goal_pos:
            self.performance_metrics['successful_paths'] += 1
            print(f"🎯 Game {self.game_count} completed successfully! Final score: {game_score:.1f}")
            return True
        else:
            print(f"❌ Game {self.game_count} failed. Final score: {game_score:.1f}")
            return False
    
    def display_performance_metrics(self):
        """Display comprehensive performance metrics"""
        print("\n📊 PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Games played: {self.game_count}")
        print(f"Total score: {self.total_score:.1f}")
        print(f"Average score per game: {self.total_score/max(1, self.game_count):.1f}")
        print(f"Success rate: {self.performance_metrics['successful_paths']/max(1, self.game_count)*100:.1f}%")
        print(f"Total steps taken: {self.performance_metrics['steps_taken']}")
        print(f"Average steps per game: {self.performance_metrics['steps_taken']/max(1, self.game_count):.1f}")
        print(f"Total simulations run: {self.performance_metrics['total_simulations']}")
        print(f"Cache hits: {self.performance_metrics['cache_hits']}")
        print(f"Cache hit rate: {self.performance_metrics['cache_hits']/max(1, self.performance_metrics['total_simulations'])*100:.1f}%")
        print(f"Average simulation time: {self.performance_metrics['average_simulation_time']*1000:.1f}ms")
        print("=" * 50)

    def run_single_game_with_analysis(self):
        """Run a single game with detailed step-by-step analysis"""
        print(f"\n🔬 DETAILED GAME ANALYSIS")
        print("=" * 60)
        
        obs, info = self.env.reset()
        game_score = 0.0
        step_count = 0
        max_steps = 100
        
        # Clear cache for new game
        self.simulation_cache.clear()
        
        # Initialize the actual game board for this game
        self._initialize_actual_game_board()
        
        # Initialize the simulation board for this game
        self._initialize_simulation_board()
        
        # Track decision making
        decisions = []
        
        while step_count < max_steps:
            print(f"\n--- STEP {step_count + 1} ---")
            print(f"Agent position: {self.env.unwrapped.agent_pos}")
            print(f"Agent direction: {self.env.unwrapped.agent_dir}")
            print(f"Carrying: {self.env.unwrapped.carrying}")
            print(f"Door opened: {self.env.unwrapped.door_opened}")
            
            # Record current position before action
            old_pos = self.env.unwrapped.agent_pos
            
            # Choose action using simulation-based algorithm
            action = self.choose_action()
            
            # Record decision
            decisions.append({
                'step': step_count + 1,
                'position': list(self.env.unwrapped.agent_pos),
                'action': action,
                'score_before': game_score
            })
            
            # Take the action
            obs, reward, terminated, truncated, info = self.env.step(action)
            game_score += reward
            step_count += 1
            
            # Update performance metrics
            self.performance_metrics['steps_taken'] += 1
            
            # Update the actual game board
            new_pos = self.env.unwrapped.agent_pos
            self._update_actual_game_board(action, new_pos, old_pos)
            
            print(f"Action taken: {action}, Reward: {reward:.1f}, Total score: {game_score:.1f}")
            print(f"   Moved from {old_pos} to {new_pos}")
            
            # Check if game is over
            if terminated or truncated:
                break
            
            # Small delay to see what's happening
            import time
            time.sleep(0.2)
        
        # Final analysis
        print(f"\n--- FINAL ANALYSIS ---")
        print(f"Game completed in {step_count} steps")
        print(f"Final score: {game_score:.1f}")
        
        # Display the simulation board and actual game board
        self._display_simulation_board()
        self._display_simulation_paths_summary()
        self._display_actual_game_board()
        self._display_actual_moves_summary()
        self._display_simulation_vs_actual_comparison()
        
        if terminated and self.env.unwrapped.goal_pos and self.env.unwrapped.agent_pos == self.env.unwrapped.goal_pos:
            self.performance_metrics['successful_paths'] += 1
            print("🎯 SUCCESS: Goal reached!")
            success = True
        else:
            print("❌ FAILURE: Goal not reached")
            success = False
        
        # Analyze decision making
        print(f"\n--- DECISION ANALYSIS ---")
        for i, decision in enumerate(decisions):
            if i < len(decisions) - 1:
                next_decision = decisions[i + 1]
                position_change = (
                    next_decision['position'][0] - decision['position'][0],
                    next_decision['position'][1] - decision['position'][1]
                )
                print(f"Step {decision['step']}: Action {decision['action']} moved from {decision['position']} to {next_decision['position']} (change: {position_change})")
        
        return success, game_score, step_count
    
    def compare_simulation_strategies(self, num_trials=3):
        """Compare different simulation strategies"""
        print(f"\n🧪 COMPARING SIMULATION STRATEGIES")
        print("=" * 60)
        
        strategies = {
            '5_sims_20_steps': {'sims': 5, 'steps': 20},
            '3_sims_30_steps': {'sims': 3, 'steps': 30},
            '10_sims_10_steps': {'sims': 10, 'steps': 10}
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            print(f"\nTesting strategy: {strategy_name}")
            print(f"Parameters: {params['sims']} simulations, {params['steps']} steps each")
            
            # Temporarily modify simulation parameters
            original_simulate = self._simulate_best_action
            
            def custom_simulate(possible_actions, sims=params['sims'], steps=params['steps']):
                print(f"🧪 Running {sims} simulations with {steps} steps for each of {len(possible_actions)} actions...")
                
                action_scores = {}
                action_details = {}
                
                for action in possible_actions:
                    scores = []
                    for sim in range(sims):
                        score = self.simulate_action_optimized(action, max_steps=steps)
                        scores.append(score)
                        print(f"   Action {action} simulation {sim+1}: {score:.1f}")
                    
                    # Calculate statistics for this action
                    avg_score = sum(scores) / len(scores)
                    min_score = min(scores)
                    max_score = max(scores)
                    std_dev = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
                    
                    action_scores[action] = avg_score
                    action_details[action] = {
                        'scores': scores,
                        'avg': avg_score,
                        'min': min_score,
                        'max': max_score,
                        'std': std_dev
                    }
                    
                    print(f"   Action {action} - Avg: {avg_score:.1f}, Min: {min_score:.1f}, Max: {max_score:.1f}, Std: {std_dev:.1f}")
                
                # Find the best action based on average score
                best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
                best_score = action_scores[best_action]
                
                print(f"✅ Best action: {best_action} with score {best_score:.1f}")
                return best_action
            
            # Replace the simulation method temporarily
            self._simulate_best_action = custom_simulate
            
            # Test the strategy
            strategy_scores = []
            strategy_steps = []
            
            for trial in range(num_trials):
                print(f"\nTrial {trial + 1}:")
                success, score, steps = self.run_single_game_with_analysis()
                strategy_scores.append(score)
                strategy_steps.append(steps)
                
                # Reset for next trial
                self.env.reset()
                self.simulation_cache.clear()
            
            # Restore original method
            self._simulate_best_action = original_simulate
            
            # Calculate strategy performance
            avg_score = sum(strategy_scores) / len(strategy_scores)
            avg_steps = sum(strategy_steps) / len(strategy_steps)
            success_rate = sum(1 for s in strategy_scores if s > 0) / len(strategy_scores)
            
            results[strategy_name] = {
                'avg_score': avg_score,
                'avg_steps': avg_steps,
                'success_rate': success_rate,
                'scores': strategy_scores,
                'steps': strategy_steps
            }
            
            print(f"\nStrategy {strategy_name} results:")
            print(f"  Average score: {avg_score:.1f}")
            print(f"  Average steps: {avg_steps:.1f}")
            print(f"  Success rate: {success_rate*100:.1f}%")
        
        # Final comparison
        print(f"\n--- STRATEGY COMPARISON ---")
        print("=" * 60)
        for strategy_name, result in results.items():
            print(f"{strategy_name:20} | Score: {result['avg_score']:6.1f} | Steps: {result['avg_steps']:5.1f} | Success: {result['success_rate']*100:5.1f}%")
        
        return results

def main():
    env_id = 'MiniGrid-ComplexDiscovery-v0'
    try:
        gym.register(id=env_id, entry_point=ComplexDiscoveryEnv)
    except gym.error.Error:
        pass

    env = gym.make(env_id)
    
    # Create automated agent
    agent = AutomatedAgent(env)
    
    print("🤖 AUTOMATED AGENT MODE")
    print("="*50)
    print("The agent will play automatically using a simulation-based algorithm.")
    print("Each step costs -1 point, reaching the goal gives +10 points.")
    print("="*50)
    
    # Choose mode
    print("\nChoose mode:")
    print("1. Play multiple games with current algorithm")
    print("2. Run single game with detailed analysis")
    print("3. Compare different simulation strategies")
    print("4. Run performance benchmark")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except:
        choice = "1"  # Default to option 1
    
    if choice == "1":
        # Play multiple games
        num_games = 3
        successful_games = 0
        
        for i in range(num_games):
            success = agent.play_game()
            if success:
                successful_games += 1
            
            if i < num_games - 1:  # Don't wait after the last game
                print(f"\n🔄 Starting next game in 3 seconds...")
                import time
                time.sleep(3)
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"Games played: {num_games}")
        print(f"Successful games: {successful_games}")
        print(f"Success rate: {successful_games/num_games*100:.1f}%")
        print(f"Total score across all games: {agent.total_score:.1f}")
        print(f"Average score per game: {agent.total_score/num_games:.1f}")
        agent.display_performance_metrics()
        
    elif choice == "2":
        # Run single game with detailed analysis
        success, score, steps = agent.run_single_game_with_analysis()
        print(f"\n🎯 Game completed: {'SUCCESS' if success else 'FAILURE'}")
        print(f"Final score: {score:.1f}")
        print(f"Steps taken: {steps}")
        agent.display_performance_metrics()
        
    elif choice == "3":
        # Compare simulation strategies
        results = agent.compare_simulation_strategies(num_trials=2)
        print(f"\n📊 Strategy comparison completed!")
        agent.display_performance_metrics()
        
    elif choice == "4":
        # Performance benchmark
        print("\n🏃 Running performance benchmark...")
        import time
        
        start_time = time.time()
        success, score, steps = agent.run_single_game_with_analysis()
        end_time = time.time()
        
        print(f"\n⏱️ PERFORMANCE BENCHMARK RESULTS:")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Steps per second: {steps / (end_time - start_time):.2f}")
        print(f"Simulations per second: {agent.performance_metrics['total_simulations'] / (end_time - start_time):.2f}")
        agent.display_performance_metrics()
        
    else:
        print("Invalid choice. Running default mode (multiple games)...")
        # Run default mode
        success = agent.play_game()
        agent.display_performance_metrics()

if __name__ == "__main__":
    main()




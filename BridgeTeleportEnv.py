import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, List
import random
import copy
import time

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj, Goal, Door, Key, Wall
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.utils.rendering import fill_coords, point_in_circle

# --- אובייקטים חדשים של הסביבה ---

class GoodBridge(WorldObj):
    """גשר טוב: משגר את הסוכן קרוב יותר למטרה."""
    def __init__(self):
        super().__init__("ball", "green") # שינוי לייצוג ייחודי (כדור ירוק)
        self.start_pos: Optional[Tuple[int, int]] = None
        self.end_pos: Optional[Tuple[int, int]] = None
    def can_overlap(self):
        return True
    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), (0, 255, 0))

class BadBridge(WorldObj):
    """גשר רע: משגר את הסוכן חזרה להתחלה."""
    def __init__(self):
        super().__init__("ball", "red") # שינוי לייצוג ייחודי (כדור אדום)
        self.start_pos: Optional[Tuple[int, int]] = None
        self.end_pos: Optional[Tuple[int, int]] = None
    def can_overlap(self):
        return True
    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), (255, 0, 0))


# --- סביבה מותאמת אישית ---

class ComplexDiscoveryEnv(MiniGridEnv):
    """סביבה הכוללת את הגשרים החדשים."""
    def __init__(self, size=5, max_steps=None, **kwargs):
        self.mission_space = MissionSpace(
            mission_func=lambda: "מצא את המפתח, פתח את הדלת, והגע למטרה. גלה אובייקטים חדשים."
        )
        if max_steps is None:
            max_steps = size * 6
        
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.start_pos: Optional[Tuple[int, int]] = None
        
        super().__init__(
            mission_space=self.mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        for y in range(height):
            if y != 2: self.grid.set(2, y, Wall())
        
        door_color = self._rand_elem(["red", "green", "blue", "yellow"])
        self.put_obj(Door(door_color, is_locked=True), 2, 2)
        
        self.start_pos = (1, 1)
        self.place_agent(top=self.start_pos, size=(1, 1))
        # --- FIX 1 ---
        self.agent_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        
        self.put_obj(Goal(), 3, 3)
        self.goal_pos = (3, 3)
        
        self.put_obj(Key(door_color), 1, 3)
        
        good_bridge = GoodBridge()
        good_bridge.start_pos = (1, 4)
        good_bridge.end_pos = (3, 1)
        self.put_obj(good_bridge, *good_bridge.start_pos)

        bad_bridge = BadBridge()
        bad_bridge.start_pos = (0, 3)
        bad_bridge.end_pos = self.start_pos
        self.put_obj(bad_bridge, *bad_bridge.start_pos)
        
        self.mission = self.mission_space.sample()

    def step(self, action):
        fwd_cell = None
        
        if action == Actions.forward:
            fwd_pos = self.front_pos
            if 0 <= fwd_pos[0] < self.width and 0 <= fwd_pos[1] < self.height:
                fwd_cell = self.grid.get(*fwd_pos)

        obs, reward, terminated, truncated, info = super().step(action)
        # --- FIX 2 ---
        self.agent_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))

        if action == Actions.forward and isinstance(fwd_cell, (GoodBridge, BadBridge)):
             # --- FIX 3 ---
             if fwd_cell.end_pos:
                self.agent_pos = (int(fwd_cell.end_pos[0]), int(fwd_cell.end_pos[1]))
             obs = self.gen_obs()

        if self.agent_pos == self.goal_pos:
            terminated = True
            reward = 1.0 - 0.9 * (self.step_count / self.max_steps)
        else:
            reward = -0.01
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # --- FIX 4 ---
        self.agent_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        return obs, info
# --- סוכן אוטומטי עם למידה ותכנון מחדש ---

class RePlanningAgent:
    """סוכן שבונה מפה מנטלית, לומד ומתכנן מחדש את דרכו."""
    def __init__(self, env):
        self.env = env
        self.simulation_cache = {}
        self._create_initial_mental_map()
        
        self.known_object_functions = {
            'wall': True, 'door': True, 'key': True, 'goal': True,
            'ball': False
        }
        self.learned_effects = {}
        print("--- המוח של הסוכן נוצר ---")
        print(f"ידע מולד על תפקודים: {self.known_object_functions}")

    def _create_initial_mental_map(self):
        """יוצר מפה מנטלית התחלתית עם ידע בסיסי על אובייקטים סטנדרטיים."""
        print("--- יוצר מפה מנטלית התחלתית עם ידע בסיסי... ---")
        self.mental_map = Grid(self.env.width, self.env.height)
        known_types = ['wall', 'door', 'key', 'goal']
        
        for j in range(self.env.height):
            for i in range(self.env.width):
                cell = self.env.grid.get(i, j)
                if cell and cell.type in known_types:
                    self.mental_map.set(i, j, cell)
        print("--- מפה מנטלית התחלתית נוצרה. ---")
    
    def _display_mental_map(self):
        """מדפיס ייצוג טקסטואלי של המפה המנטלית של הסוכן."""
        print("\n--- המפה המנטלית הנוכחית של הסוכן ---")
        agent_pos = self.env.agent_pos
        if agent_pos is None:
            print(" (מיקום הסוכן עדיין לא ידוע)")
            return

        for j in range(self.mental_map.height):
            row_str = ""
            for i in range(self.mental_map.width):
                cell = self.mental_map.get(i, j)
                if i == agent_pos[0] and j == agent_pos[1]:
                    row_str += " A "
                elif cell is None:
                    row_str += " . "
                elif cell.type == 'wall':
                    row_str += "[W]"
                elif cell.type == 'door':
                    row_str += "[D]"
                elif cell.type == 'key':
                    row_str += " K "
                elif cell.type == 'goal':
                    row_str += " G "
                else:
                    row_str += " ? "
            print(row_str)
        print("-------------------------------------\n")


    def update_mental_map(self, obs):
        """מעדכן את המפה המנטלית על סמך התצפית הנוכחית."""
        print("--- מעדכן מפה מנטלית על סמך תצפית... ---")
        agent_pos = np.array(self.env.agent_pos)
        
        image = obs['image']
        view_size = image.shape[0]

        view_grid = Grid(view_size, view_size)
        for i in range(view_size):
            for j in range(view_size):
                obj_type, color, state = image[i, j]
                if obj_type > 0:
                    v = WorldObj.decode(obj_type, color, state)
                    if v:
                        if v.type == 'ball' and v.color == 'green':
                            v = GoodBridge()
                        elif v.type == 'ball' and v.color == 'red':
                            v = BadBridge()
                        view_grid.set(i, j, v)

        rotated_grid = view_grid
        for _ in range(self.env.agent_dir):
            rotated_grid = rotated_grid.rotate_left()

        top_left = self.env.agent_pos - self.env.dir_vec * (view_size - 1) + self.env.right_vec * (view_size // 2)

        for j_view in range(view_grid.height):
            for i_view in range(view_grid.width):
                cell = rotated_grid.get(i_view, j_view)
                if cell:
                    abs_x = top_left[0] + i_view
                    abs_y = top_left[1] + j_view

                    if 0 <= abs_x < self.env.width and 0 <= abs_y < self.env.height:
                        if self.mental_map.get(abs_x, abs_y) is None:
                            self.mental_map.set(abs_x, abs_y, cell)
                            print(f"  - גילוי! אובייקט חדש '{cell.type}' נוסף למפה במיקום ({abs_x}, {abs_y})")

    def learn_from_interaction(self, old_pos, new_pos, action):
        """לומד את האפקט של אובייקט לאחר אינטראקציה."""
        if action != Actions.forward: return

        old_pos_tuple = tuple(map(int, old_pos))
        obj = self.env.grid.get(*old_pos_tuple)
        
        if obj and isinstance(obj, (GoodBridge, BadBridge)):
            expected_new_pos = getattr(obj, 'end_pos', None)
            if new_pos == expected_new_pos:
                if old_pos_tuple not in self.learned_effects:
                    print(f"--- Eureka! למדתי על '{obj.type}' במיקום {old_pos_tuple} ---")
                    print(f"    הוא משגר ל-{new_pos}")
                    self.learned_effects[old_pos_tuple] = new_pos
                    self.known_object_functions[obj.type] = True

    def is_action_legal(self, action, env):
        """בודק אם פעולה היא חוקית בסביבה הנתונה."""
        front_pos = tuple(map(int, env.front_pos))
        
        if not (0 <= front_pos[0] < env.width and 0 <= front_pos[1] < env.height):
            if action == Actions.forward: return False
            return True

        fwd_cell = env.grid.get(*front_pos)

        if action == Actions.forward:
            return fwd_cell is None or fwd_cell.can_overlap()
        
        elif action == Actions.pickup:
            return fwd_cell is not None and fwd_cell.can_pickup() and env.carrying is None

        elif action == Actions.toggle:
            return fwd_cell is not None and fwd_cell.type == 'door'

        return True

    def _choose_simple_action(self, env, mental_grid):
        """מדיניות פשוטה להנחיית סימולציות, עם בדיקת חוקיות."""
        agent_pos = tuple(map(int, env.agent_pos))
        agent_dir = env.agent_dir
        
        key_pos, door_pos, goal_pos = self._find_objects_in_map(mental_grid)

        has_key = isinstance(env.carrying, Key)
        door = mental_grid.get(*door_pos) if door_pos else None
        is_door_open = isinstance(door, Door) and not door.is_locked

        target_pos = goal_pos
        if key_pos and not has_key: target_pos = key_pos
        elif door_pos and not is_door_open: target_pos = door_pos
        
        if target_pos is None: return Actions.forward

        front_pos = tuple(map(int, env.front_pos))
        if self.is_action_legal(Actions.pickup, env) and front_pos == key_pos and not has_key: return Actions.pickup
        if self.is_action_legal(Actions.toggle, env) and front_pos == door_pos and has_key and not is_door_open: return Actions.toggle

        dir_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        best_dot = -float('inf')
        best_turn = Actions.left

        possible_moves = []
        for turn_action in [None, Actions.left, Actions.right]:
            final_action = turn_action if turn_action is not None else Actions.forward
            if self.is_action_legal(final_action, env):
                possible_moves.append(turn_action)

        if not possible_moves: return Actions.left

        for turn_action in possible_moves:
            current_dir = agent_dir
            if turn_action == Actions.left: current_dir = (agent_dir + 3) % 4
            elif turn_action == Actions.right: current_dir = (agent_dir + 1) % 4

            target_vec = (target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1])
            agent_fwd_vec = dir_vectors[current_dir]
            dot = sum(tv * av for tv, av in zip(target_vec, agent_fwd_vec))

            if dot > best_dot:
                best_dot = dot
                best_turn = turn_action if turn_action is not None else Actions.forward
        
        return best_turn

    def _find_objects_in_map(self, grid):
        key_pos, door_pos, goal_pos = None, None, None
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell:
                    if isinstance(cell, Key): key_pos = (i, j)
                    elif isinstance(cell, Door): door_pos = (i, j)
                    elif isinstance(cell, Goal): goal_pos = (i, j)
        return key_pos, door_pos, goal_pos

    def simulate_step(self, sim_env, action):
        """
        מבצע צעד סימולציה בטוח לגמרי:
        - מגן על מקרי קצה בקצוות המפה.
        - מטפל ידנית בפניות מחוץ לגבול.
        - מיישם אפקטי גשרים נלמדים.
        """
        fwd_pos = sim_env.front_pos
        is_fwd_out_of_bounds = not (0 <= fwd_pos[0] < sim_env.width and 0 <= fwd_pos[1] < sim_env.height)

        # 🟢 מקרה 1: פנייה מחוץ לגבול (נפוץ מאוד שגורם לקריסה)
        if is_fwd_out_of_bounds and action in [Actions.left, Actions.right]:
            if action == Actions.left:
                sim_env.agent_dir = (sim_env.agent_dir - 1) % 4
            else:  # Actions.right
                sim_env.agent_dir = (sim_env.agent_dir + 1) % 4

            sim_env.step_count += 1
            obs = sim_env.gen_obs()
            reward = -0.01
            terminated = False
            truncated = sim_env.step_count >= sim_env.max_steps
            info = {}
            return obs, reward, terminated, truncated, info

        # 🟢 מקרה 2: פעולה אחרת כשהסוכן מחוץ לגבול → נחזיר צעד דמיוני
        if is_fwd_out_of_bounds:
            sim_env.step_count += 1
            obs = sim_env.gen_obs()
            reward = -0.01
            terminated = False
            truncated = sim_env.step_count >= sim_env.max_steps
            info = {"warning": "action_out_of_bounds"}
            return obs, reward, terminated, truncated, info

        # 🟢 מקרה 3: צעד רגיל בתוך גבולות המפה
        fwd_pos_tuple = tuple(map(int, sim_env.front_pos))
        obs, reward, terminated, truncated, info = sim_env.step(action)

        # אפקט גשרים נלמדים
        if action == Actions.forward and fwd_pos_tuple in self.learned_effects:
            sim_env.agent_pos = self.learned_effects[fwd_pos_tuple]
            obs = sim_env.gen_obs()

        return obs, reward, terminated, truncated, info

    def simulate_trajectory(self, first_action, max_steps=20):
        sim_env = ComplexDiscoveryEnv(size=self.env.unwrapped.width, render_mode=None)
        sim_env.grid = copy.deepcopy(self.mental_map)
        sim_env.agent_pos = tuple(map(int, self.env.unwrapped.agent_pos))
        sim_env.agent_dir = self.env.unwrapped.agent_dir
        sim_env.carrying = self.env.unwrapped.carrying
        sim_env.step_count = self.env.unwrapped.step_count

        action_path = [first_action]
        obs, reward, terminated, truncated, info = self.simulate_step(sim_env, first_action)
        total_reward = reward

        for _ in range(max_steps - 1):
            if terminated or truncated: break
            next_action = self._choose_simple_action(sim_env, sim_env.grid)
            action_path.append(next_action)
            obs, reward, terminated, truncated, info = self.simulate_step(sim_env, next_action)
            total_reward += reward
        
        return total_reward, action_path

    def plan_best_trajectory(self):
        """מריץ סימולציות רק על פעולות חוקיות ומחזיר את המסלול הטוב ביותר."""
        all_actions = [Actions.left, Actions.right, Actions.forward, Actions.pickup, Actions.toggle]
        legal_actions = [a for a in all_actions if self.is_action_legal(a, self.env)]
        
        if not legal_actions:
            return [Actions.left]

        best_path = []
        best_score = -float('inf')

        print("\n--- הסוכן מתכנן מסלול (רק מפעולות חוקיות)... ---")
        for action in legal_actions:
            score, path = self.simulate_trajectory(action)
            print(f"  - סימולציית מסלול המתחיל ב-'{action.name}': ניקוד={score:.2f}, אורך={len(path)}")
            if score > best_score:
                best_score = score
                best_path = path
        
        print(f"--- המסלול הטוב ביותר תוכנן (מתחיל ב: {best_path[0].name}, ניקוד: {best_score:.2f}) ---\n")
        return best_path

# --- לולאת המשחק הראשית ---

def main():
    """מאתחל ומריץ את המשחק עם הסוכן הלומד במחזורי תכנון מחדש."""
    env = ComplexDiscoveryEnv(render_mode="human")
    
    obs, info = env.reset()
    agent = RePlanningAgent(env)
    
    agent.update_mental_map(obs)
    env.render()
    
    print("--- מתחיל משחק ---")
    print(f"משימה: {env.mission}")
    time.sleep(2)

    total_reward = 0
    current_plan = []
    
    for i in range(env.max_steps):
        if i % 3 == 0:
            print(f"\n==================== [שלב {i+1}] מחזור תכנון חדש ====================")
            print("הסוכן עוצר כדי לחשוב מחדש על סמך הידע המעודכן...")
            agent._display_mental_map()
            time.sleep(2)
            current_plan = agent.plan_best_trajectory()
            
            if not current_plan:
                # --- תיקון שגיאת תחביר ---
                print("❌ כישלון: הסוכן לא הצליח למצוא מסלול חוקי.")
                break

        action_index_in_plan = i % 3
        if action_index_in_plan >= len(current_plan):
            print("🏁 מידע: התוכנית הנוכחית הסתיימה, מתחילים תכנון מחדש.")
            continue

        action = current_plan[action_index_in_plan]
        
        old_pos = env.agent_pos
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos
        total_reward += reward
        
        agent.update_mental_map(obs)
        agent._display_mental_map()
        agent.learn_from_interaction(old_pos, new_pos, action)

        env.render()
        
        print(f"\nצעד כולל {i + 1}: מבצע '{action.name}' (צעד {action_index_in_plan + 1}/3 בתוכנית הנוכחית)...")
        print(f"  - אחרי:  מיקום={new_pos}, כיוון={env.agent_dir}, ניקוד={reward:.2f}, ניקוד כולל={total_reward:.2f}")
        time.sleep(0.5)

        if terminated:
            print(f"\n✅ הצלחה: המטרה הושגה ב-{i + 1} צעדים!")
            break
        if truncated:
            print(f"\n❌ כישלון: הגעה למספר הצעדים המרבי.")
            break
            
    if not terminated and not truncated:
         print("\n🏁 מידע: המשחק הסתיים.")

    print("--- המשחק הסתיים ---")
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    main()

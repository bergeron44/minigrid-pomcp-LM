import gymnasium as gym
import minigrid
import random
import numpy as np
import pprint
from minigrid.core.world_object import Key
import logging
import copy
from collections import defaultdict
from minigrid.core.constants import IDX_TO_OBJECT
from minigrid.minigrid_env import MiniGridEnv  # הוספת import למעלה
from typing import cast
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.constants import IDX_TO_OBJECT

for obj, idx in OBJECT_TO_IDX.items():
    print(f"type={idx}: {obj}")
# יצירת קובץ לוג בשם run.log, עם אובררייט בכל הרצה
logging.basicConfig(
    filename='run.log',
    filemode='w',  # 'a' אם אתה רוצה לצרף ללוג קיים
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


# נבנה מילון הפוך: מ־int → name
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

# Mapping action indices to names
action_names = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done"
}

seen_types = set()

type_names = {
    0: "empty",
    1: "wall",
    2: "unseen",
    3: "floor",
    4: "ball",
    5: "door",
    6: "key",
    7: "ball?",
    8: "goal",
    9: "lava",
    10: "agent with key"
    # תוכל לעדכן לפי מה שתראה בלוג
}

def extract_info_from_obs(obs):
    import logging
    log = logging.getLogger(__name__)

    def print_obs_image(obs_image):
        log.info("\n🧭 Local 7x7 observation (agent always centered at [3,3]):")
        for i in range(7):
            row = ''
            for j in range(7):
                obj_type = obs_image[i][j][0]
                symbol = {
                    0: '⬛',  # unseen
                    1: '▫️',  # empty
                    2: '🧱',  # wall
                    3: '⬜',  # floor
                    4: '🚪',  # door
                    5: '🗝️',  # key
                    6: '⚽',  # ball
                    7: '📦',  # box
                    8: '🏁',  # goal
                    10: '🔺',  # agent (not always appears!)
                }.get(obj_type, '?')
                row += symbol
            log.info(row)

    print_obs_image(obs["image"])

    image = obs["image"]  # shape: (7, 7, 3)
    doors = []
    keys = []
    goals = []
    door_states = {}
    has_key = False
    pos = None  # 🔁 was: agent_pos

    log.info("\n🖼️ Observation summary:")
    for i in range(7):
        for j in range(7):
            obj_type = image[i][j][0]
            obj_color = image[i][j][1]
            obj_state = image[i][j][2]
            obj_name = IDX_TO_OBJECT.get(obj_type, "unknown")

            if obj_type != 0:
                log.info(f" - [{i},{j}] → type={obj_type} ({obj_name}), color={obj_color}, state={obj_state}")

            if obj_name == "door":
                doors.append((i, j))
                door_states[(i, j)] = (obj_state == 0)
            elif obj_name == "key":
                keys.append((i, j))
            elif obj_name == "goal":
                goals.append((i, j))
            elif obj_name == "agent":
                pos = (i, j)
                if obj_state == 2:
                    has_key = True

    if pos is not None:
        log.info(f"📍 Agent is at position {pos} in 7x7 observation grid.")
    else:
        log.warning("❓ Agent not found in observation!")

    return {
        "door_positions": doors,
        "door_open_map": door_states,
        "key_positions": keys,
        "goal_positions": goals,
        "has_key": has_key,
        "pos": pos  
    }


# def filter_belief_by_observation(prev_belief, obs):
#     filtered = {}
#     obs_info = extract_info_from_obs(obs)

#     for state_key, prob in prev_belief.items():
#         state = key_to_state(state_key)

#         match = (
#             state['door_open'] == obs_info['door_open'] and
#             state['has_key'] == obs_info['has_key'] and
#             (obs_info['key_pos'] is None or state['key'] == obs_info['key_pos']) and
#             (obs_info['goal_pos'] is None or state['goal'] == obs_info['goal_pos']) and
#             (obs_info['door_pos'] is None or state['door'] == obs_info['door_pos'])
#         )

#         if match:
#             filtered[state_key] = prob

#     # נרמול
#     total = sum(filtered.values())
#     if total > 0:
#         for k in filtered:
#             filtered[k] /= total
#     else:
#         print("⚠️ No matching states after semantic filtering.")
#         return prev_belief  # fallback

#     return filtered

def extract_semantic_facts_from_obs(obs):
    facts = {
        "has_key": False,
        "visible_key_pos": None,
        "visible_door_pos": None,
        "door_open": None,
        "visible_objects": set(),
    }

    image = obs["image"]  # shape (7, 7, 3)
    for i in range(7):
        for j in range(7):
            cell = image[i][j]
            obj_type = cell[0]
            obj_state = cell[2]

            if obj_type == 5:  # key
                facts["visible_key_pos"] = (i, j)
                facts["visible_objects"].add("key")
            elif obj_type == 4:  # door
                facts["visible_door_pos"] = (i, j)
                facts["visible_objects"].add("door")
                facts["door_open"] = (obj_state == 0)  # usually 0=open, 1=closed
            elif obj_type == 10:  # agent holding key (maybe)
                facts["has_key"] = True

    return facts


def state_to_key(state):
    def norm(pos):
        return tuple(int(x) for x in pos) if pos is not None else None

    return (
        norm(state['pos']),  # ⬅️ במקום 'agent'
        int(state.get('dir', 0)),  # ברירת מחדל ל-0 אם לא קיים
        norm(state['key']),
        norm(state['door']),
        norm(state['goal']),
        state['door_open'],
        state['has_key']
    )


def key_to_state(key, base_state=None):
    state = {
        'pos': key[0],
        'dir': key[1],
        'key': key[2],
        'door': key[3],
        'goal': key[4],
        'door_open': key[5],
        'has_key': key[6]
    }
    if base_state and 'grid' in base_state:
        state['grid'] = base_state['grid']
    return state

def generate_states_with_partial_observability(env, known_agent_pos):
    width, height = env.width, env.height
    grid = env.grid

    # שלב 1: מוצאים משבצות חוקיות (כל מה שלא קיר)
    legal_positions = []
    for x in range(width):
        for y in range(height):
            obj = grid.get(x, y)
            if obj is None or obj.type != 'wall':
                legal_positions.append((x, y))

    # הסרת מקום הסוכן מתוך רשימת האפשרויות למפתח/דלת/יעד
    legal_object_positions = [pos for pos in legal_positions if pos != known_agent_pos]

    all_states = []

    from itertools import permutations

    for (pos_key, pos_door, pos_goal) in permutations(legal_object_positions, 3):
        for has_key in [False, True]:
            for door_open in [False, True]:
                for dir in range(4):
                    # בונים גריד ריק (מלבד הקירות)
                    grid_array = np.zeros((height, width), dtype=np.uint8)
                    for x in range(width):
                        for y in range(height):
                            obj = grid.get(x, y)
                            if obj is not None and obj.type == 'wall':
                                grid_array[y, x] = OBJECT_TO_IDX['wall']
                            else:
                                grid_array[y, x] = OBJECT_TO_IDX['empty']

                    # משבצים את האובייקטים:
                    if not has_key:
                        xk, yk = pos_key
                        grid_array[yk, xk] = OBJECT_TO_IDX['key']
                    xd, yd = pos_door
                    grid_array[yd, xd] = OBJECT_TO_IDX['door']
                    xg, yg = pos_goal
                    grid_array[yg, xg] = OBJECT_TO_IDX['goal']
                    xa, ya = known_agent_pos
                    grid_array[ya, xa] = OBJECT_TO_IDX['agent']

                    state = {
                        'grid': grid_array,
                        'pos': known_agent_pos,
                        'dir': dir,
                        'key': None if has_key else pos_key,
                        'door': pos_door,
                        'goal': pos_goal,
                        'has_key': has_key,
                        'door_open': door_open
                    }
                    all_states.append(state)

    print(f"✅ Total generated states: {len(all_states)}")
    return all_states
   


def is_valid_position(env, pos):
    x, y = pos
    base_env = env.unwrapped
    return 0 <= x < base_env.width and 0 <= y < base_env.height

def set_env_state(env, state):
    from minigrid.core.world_object import Key, Door, Goal

    def is_valid(pos):
        if pos is None:
            return False
        x, y = pos
        return 0 <= x < env.width and 0 <= y < env.height

    # בדיקת מיקום חוקי של הסוכן
    if not is_valid(state['pos']):
        raise ValueError(f"❌ Invalid agent position: {state['pos']}")

    env.agent_pos = state['pos']
    env.agent_dir = state.get('dir', 0)
    env.step_count = 0

    # ניקוי האובייקטים הישנים מהגריד (key, door, goal)
    for i in range(env.width):
        for j in range(env.height):
            obj = env.grid.get(i, j)
            if obj and obj.type in ["key", "door", "goal"]:
                env.grid.set(i, j, None)

    # הוספת מפתח (אם הסוכן לא מחזיק בו)
    if not state.get('has_key', False):
        key_pos = state.get('key')
        if is_valid(key_pos):
            env.grid.set(*key_pos, Key("yellow"))

    # הוספת דלת
    door_pos = state.get('door')
    if is_valid(door_pos):
        door = Door("yellow", is_locked=not state.get('door_open', False))
        door.is_open = state.get('door_open', False)
        env.grid.set(*door_pos, door)

    # הוספת יעד
    goal_pos = state.get('goal')
    if is_valid(goal_pos):
        env.grid.set(*goal_pos, Goal())

    # האם הסוכן מחזיק מפתח
    env.agent_carrying = Key("yellow") if state.get('has_key', False) else None


def extract_observation_key(obs):
    image = obs["image"]
    if np.any(image[:, :, 0] == 8):   # goal color is 8
        return "see_goal"
    elif np.any(image[:, :, 0] == 5): # key color is 5
        return "see_key"
    elif np.any(image[:, :, 0] == 2): # wall
        return "see_wall"
    else:
        return "nothing_seen"

def env_to_state(env):
    env = env.unwrapped  # הסרת עטיפות של Gym
    from minigrid.core.world_object import Key, Door, Goal
    from minigrid.core.constants import OBJECT_TO_IDX
    import numpy as np

    agent_pos = tuple(env.agent_pos)
    agent_dir = env.agent_dir
    has_key = env.agent_carrying is not None and env.agent_carrying.type == "key"

    key_pos = None
    door_pos = None
    goal_pos = None
    door_open = False

    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            if obj.type == "key":
                key_pos = (x, y)
            elif obj.type == "door":
                door_pos = (x, y)
                door_open = obj.is_open
            elif obj.type == "goal":
                goal_pos = (x, y)

    # ✨ יצירת grid מספרי עם הצירים בסדר (y,x)
    grid_array = np.full((env.height, env.width), OBJECT_TO_IDX["empty"], dtype=np.uint8)

    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj is None:
                grid_array[y, x] = OBJECT_TO_IDX["empty"]
            elif obj.type == "wall":
                grid_array[y, x] = OBJECT_TO_IDX["wall"]
            else:
                grid_array[y, x] = OBJECT_TO_IDX["empty"]  # אובייקטים אחרים ייכנסו בהמשך לפי המיקום המדויק

    # הוספת האובייקטים לפי מיקומם
    if key_pos:
        x, y = key_pos
        grid_array[y, x] = OBJECT_TO_IDX["key"]

    if door_pos:
        x, y = door_pos
        grid_array[y, x] = OBJECT_TO_IDX["door"]

    if goal_pos:
        x, y = goal_pos
        grid_array[y, x] = OBJECT_TO_IDX["goal"]

    if agent_pos:
        x, y = agent_pos
        grid_array[y, x] = OBJECT_TO_IDX["agent"]

    return {
        'pos': agent_pos,
        'dir': agent_dir,
        'key': key_pos,
        'door': door_pos,
        'goal': goal_pos,
        'door_open': door_open,
        'has_key': has_key,
        'grid': grid_array  # ✅ הכי חשוב
    }


def find_object_position(env, obj_type):
    for i in range(env.width):
        for j in range(env.height):
            obj = env.grid.get(i, j)
            if obj and obj.type == obj_type:
                return (i, j)
    return None
class Belief:
    def __init__(self):
        self.belief = {}

    def normalize(self):
        total = sum(self.belief.values())
        if total > 0:
            for k in self.belief:
                self.belief[k] /= total

    def sample(self):
        """
        מחזירה מפתח (state_key) שנבחר אקראית לפי ההסתברויות באמונה.
        """
        keys = list(self.belief.keys())
        probs = [self.belief[k] for k in keys]
        return random.choices(keys, weights=probs, k=1)[0]



    import logging

    def update(self, prev_belief, action, obs, transition_model, observation_model,state_map):
        IDX_TO_SYMBOL = {
        0: "⬛",  # unseen
        1: "▫️",  # empty
        2: "🧱",  # wall
        3: "◻️",  # floor
        4: "🚪",  # door
        5: "🗝️",  # key
        6: "⚽",  # ball
        7: "📦",  # box
        8: "🎯",  # goal
        9: "🔥",  # lava
        10: "🤖",  # agent
    }

    
        def display_observation(obs_image):
            obs_types = obs_image[:, :, 0]
            return "\n".join("".join(IDX_TO_SYMBOL.get(int(t), "?") for t in row) for row in obs_types)

        def find_partial_obs_match(obs_image, state_grid):
            obs_types = obs_image[:, :, 0]  # תצפית בגודל 7x7
            H, W = state_grid.shape         # הלוח האמיתי, תמיד 5x5

            logging.info("\n🗺️ Full state grid:")
            for row in state_grid:
                logging.info("".join(IDX_TO_SYMBOL.get(int(cell), "?") for cell in row))

            for k in range(4):
                rotated = np.rot90(obs_types, k=k)

                for flip in [False, True]:
                    candidate = np.fliplr(rotated) if flip else rotated
                    visible_mask = candidate != 0

                    # מציאת גבולות תת-המלבן השונה מ־0
                    rows, cols = np.where(visible_mask)
                    if len(rows) == 0 or len(cols) == 0:
                        continue  # אין מה להשוות

                    min_r, max_r = rows.min(), rows.max()
                    min_c, max_c = cols.min(), cols.max()

                    sub_candidate = candidate[min_r:max_r + 1, min_c:max_c + 1]
                    sub_mask = visible_mask[min_r:max_r + 1, min_c:max_c + 1]
                    sub_H, sub_W = sub_candidate.shape

                    label = f"k={k * 90}°, flip={'Yes' if flip else 'No'}"
                    logging.info(f"\n🔍 Cropped rotated observation ({label}):")
                    for i in range(sub_H):
                        row_symbols = ""
                        for j in range(sub_W):
                            if sub_mask[i, j]:
                                row_symbols += IDX_TO_SYMBOL.get(int(sub_candidate[i, j]), "?")
                            else:
                                row_symbols += IDX_TO_SYMBOL[0]
                        logging.info(row_symbols)

                    # סריקה על הלוח 5x5
                    for top in range(H - sub_H + 1):
                        for left in range(W - sub_W + 1):
                            window = state_grid[top:top + sub_H, left:left + sub_W]
                            comparison = (window == sub_candidate)
                            if np.all(comparison[sub_mask]):
                                logging.info(f"✅ Match found at (top={top}, left={left}) with {label}")
                                return True

            logging.info("❌ No match found in any rotation/mirroring.")
            return False


        logging.info(f"\n🔁 Belief update (custom): Action = {action_names[action]}")
        logging.info(f"🔢 Previous belief size: {len(prev_belief)}")

        # 🖼️ הצגת התצפית
        logging.info("\n🖼️ Raw observation image:\n" + display_observation(obs['image']))
        logging.info(obs['image'])
        new_belief = {}
        num_considered = 0
        num_retained = 0

        for state_key, prob in prev_belief.items():
            num_considered += 1
            state = key_to_state(state_key,state_map[state_key])
            logging.debug("3")
            next_state = transition_model(state, action)
            next_key = state_to_key(next_state)
            # 🆕 עדכון state_map אם חסר
            if next_key not in state_map:
                state_map[next_key] = next_state
            agent_dir = next_state['dir']
            state_grid = next_state.get('grid')

            if state_grid is None:
                logging.warning(f"⚠️ Skipping state with no grid: {next_state}")
                continue

            match = find_partial_obs_match(obs['image'], state_grid)

            if match:
                new_belief[state_to_key(next_state)] = prob
                num_retained += 1
            else:
                logging.warning(f"❌ State rejected due to visual mismatch: {next_state}")

        # נרמול והחלפת אמונה
        if new_belief:
            total = sum(new_belief.values())
            for k in new_belief:
                new_belief[k] /= total
            logging.info(f"✅ Belief updated. {num_retained}/{num_considered} states retained.")
            self.belief = new_belief
        else:
            logging.warning("⚠️ No visual matches — belief unchanged.")
            self.belief = dict(prev_belief)

        logging.info(f"📉 Belief normalized. Total states: {len(self.belief)}")



class Node:
    def __init__(self, history=None):
        self.visits = 0
        self.total_reward = 0
        self.children = {}  # action -> Node
        self.particles = []  # list of state keys
        self.history = history  # היסטוריית פעולות: tuple של (a1, a2, ...)


    def value(self):
        return self.total_reward / self.visits if self.visits > 0 else 0

    def ucb_score(self, parent_visits, c=1.41):
        if self.visits == 0:
            return float("inf")
        return self.value() + c * np.sqrt(np.log(parent_visits) / self.visits)

class POMCP:
    def __init__(self, env_name, n_simulations=5000, rollout_depth=50):
        self.env_name = env_name
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.tree = {}


    def plan(self, belief,state_map):
        logging.info("\n🔍 Starting POMCP planning...")
        logging.info(f"📦 Belief has {len(belief.belief)} possible states")
        self.tree = {}
        root_history = ()

        if root_history not in self.tree:
            logging.info("🌱 Initializing root node in tree")
            self.tree[root_history] = Node()
        else:
            logging.info(f"📊 Root already exists with visits: {self.tree[root_history].visits}")

        for i in range(self.n_simulations):
            sim_env = gym.make(self.env_name)
            sim_env.reset()

            sampled_key = belief.sample()
            if sampled_key is None:
                logging.warning("❌ No states left in belief to sample from. Skipping simulation.")
                return [2, 2, 2]

            sampled_state = key_to_state(sampled_key,state_map[sampled_key])
            x, y = sampled_state['pos']  # ✅ שונה ל-'pos'

            env_core = cast(MiniGridEnv, sim_env.unwrapped)

            if not (0 <= x < env_core.width and 0 <= y < env_core.height):
                logging.warning(f"⚠️ Sampled agent position out of bounds: {(x, y)} – skipping")
                continue

            try:
                set_env_state(env_core, sampled_state)
            except ValueError as e:
                logging.warning(str(e))
                continue

            if i % 500 == 0:
                logging.debug(f"🌀 Simulation {i+1}/{self.n_simulations} with sampled state: {sampled_state}")

            self.simulate(sim_env, root_history, self.rollout_depth, sim_index=i, rollout_step=0)

        # ⬇️ Action selection phase
        actions = []
        node = self.tree[root_history]
        history = root_history

        logging.info("\n📊 Selecting 3 best actions along a single path:")
        for step in range(3):
            if history not in self.tree:
                logging.warning(f"⚠️ History {history} not found in tree.")
                break

            node = self.tree[history]
            if not node.children:
                logging.warning("🚨 No children in node — tree expansion might have failed.")
                break

            action_stats = []
            for a, child in node.children.items():
                score = child.ucb_score(node.visits)
                logging.info(f"  ➤ Action {action_names[a]} | UCB: {score:.2f} | Visits: {child.visits}")
                action_stats.append((a, score, child))

            explored_actions = [tup for tup in action_stats if tup[2].visits > 0]
            unexplored_actions = [tup for tup in action_stats if tup[2].visits == 0]

            if explored_actions:
                best_action, best_score, best_child = max(explored_actions, key=lambda x: x[1])
            elif unexplored_actions:
                best_action, best_score, best_child = max(unexplored_actions, key=lambda x: x[1])
            else:
                logging.warning("⚠️ No best action found — breaking early.")
                break

            logging.info(f"✅ Step {step+1}: Chose action {action_names[best_action]} (UCB={best_score:.2f})")
            actions.append(best_action)
            history = history + (best_action,)

        logging.info(f"\n🎯 Final selected actions: {[action_names[a] for a in actions]}")
        logging.info(f"📏 Plan length: {len(actions)}")

        return actions


   

    #  def simulate(self, env, history, depth, sim_index, rollout_step=0, prev_obs=None):
    #     if depth == 0:
    #         logging.debug("💟 Reached rollout depth 0 — returning 0")
    #         return 0
    #     logging.debug(f"env to state : {env_to_state(env)}")
    #     if history not in self.tree:
    #         logging.debug(f"🌱 Creating new LEAF node for history: {history}")
    #         self.tree[history] = Node(history=history)
    #         state_key = state_to_key(env_to_state(env))
    #         self.tree[history].particles.append(state_key)
            
    #         heuristic_value = self.heuristic(env_to_state(env))
    #         reward = 0
    #         total = reward + heuristic_value
    #         self.tree[history].visits += 1
    #         self.tree[history].total_reward += total
    #         logging.debug(f"📊 Leaf node initialized — Visits: 1, Total Reward: {total:.2f}")
    #         return total

    #     node = self.tree[history]
    #     state_key = state_to_key(env_to_state(env))
    #     node.particles.append(state_key)

    #     if len(node.children) == 0:
    #         k = 3 * ((sim_index // 1000) + 1)
    #         if rollout_step % k == k - 1:
    #             action = random.choice(range(env.action_space.n))
    #             logging.debug(f"🌿 Forced exploration at step {rollout_step}: using random action {action_names[action]}")
    #         else:
    #             best_score = float("-inf")
    #             best_action = None
    #             for a in range(env.action_space.n):
    #                 dummy_node = Node()
    #                 score = dummy_node.ucb_score(node.visits)
    #                 if score > best_score:
    #                     best_score = score
    #                     best_action = a
    #             action = best_action or random.choice(range(env.action_space.n))
    #             logging.debug(f"🌾 Leaf node UCB selected action: {action_names[action]}")

    #         state = env_to_state(env)
    #         logging.debug("2")
    #         next_state = transition_model(state, action)

    #         if not is_valid_position(env, next_state['pos']):
    #             logging.warning(f"❌ Invalid next position: {next_state['pos']} — skipping step.")
    #             return -1

    #         if next_state['pos'] == state['pos'] and action == 2:
    #             logging.debug("⛔ Action blocked — skipping simulation step.")
    #             reward = -0.1
    #             done = False
    #             obs = prev_obs
    #         else:
    #             try:
    #                 obs, reward, terminated, truncated, _ = env.step(action)
    #                 done = terminated or truncated
    #                 next_state = env_to_state(env)
    #             except AssertionError:
    #                 logging.warning(f"❌ env.step failed due to out-of-bounds despite check. Skipping.")
    #                 return -1

    #         new_history = history + (action,)
    #         if new_history not in self.tree:
    #             self.tree[new_history] = Node(history=new_history)
    #         node.children[action] = self.tree[new_history]

    #         heuristic_value = self.heuristic(env_to_state(env))
    #         total = reward + heuristic_value

    #         node.visits += 1
    #         node.total_reward += total
    #         logging.debug(f"📈 New leaf node evaluated | Visits: {node.visits}, Total Reward: {node.total_reward:.2f}")
    #         return total

    #     logging.debug(f"🌲 Internal node | Depth: {depth} | Selecting best UCB action")
    #     best_score = float("-inf")
    #     best_action = None
    #     for a in range(env.action_space.n):
    #         if a in node.children:
    #             child = node.children[a]
    #             score = child.ucb_score(node.visits)
    #             logging.debug(f"  ➡ Action: {action_names[a]} | UCB: {score:.2f} | Visits: {child.visits}")
    #         else:
    #             score = float("inf")
    #         if score > best_score:
    #             best_score = score
    #             best_action = a

    #     assert best_action is not None

    #     state = env_to_state(env)
    #     logging.debug("1")
    #     next_state = transition_model(state, best_action)

    #     if not is_valid_position(env, next_state['pos']):
    #         logging.warning(f"❌ Invalid internal next position: {next_state['pos']} — skipping.")
    #         return -1

    #     if next_state['pos'] == state['pos'] and best_action == 2:
    #         logging.debug("⛔ Action blocked — skipping simulation step (internal node).")
    #         reward = -0.1
    #         done = False
    #         obs = prev_obs
    #     else:
    #         try:
    #             obs, reward, terminated, truncated, _ = env.step(best_action)
    #             done = terminated or truncated
    #             next_state = env_to_state(env)
    #         except AssertionError:
    #             logging.warning(f"❌ env.step failed due to out-of-bounds despite check (internal). Skipping.")
    #             return -1

    #     new_history = history + (best_action,)
    #     if new_history not in self.tree:
    #         self.tree[new_history] = Node(history=new_history)
    #     node.children[best_action] = self.tree[new_history]

    #     total_reward = reward
    #     if not done:
    #         total_reward += self.simulate(env, new_history, depth - 1, sim_index, rollout_step + 1, prev_obs=obs)

    #     node.visits += 1
    #     node.total_reward += total_reward
    #     logging.debug(f"📊 Updated internal node after action {action_names[best_action]} | Visits: {node.visits}, Total: {node.total_reward:.2f}")

    #     return total_reward
    
    def simulate(self, env, history, depth, sim_index, rollout_step=0, prev_obs=None):
        if depth == 0:
            logging.debug("💟 Reached rollout depth 0 — returning 0")
            return 0

        prev_state = env_to_state(env)
        logging.debug(f"env to state : {prev_state}")

        if history not in self.tree:
            logging.debug(f"🌱 Creating new LEAF node for history: {history}")
            self.tree[history] = Node(history=history)
            state_key = state_to_key(prev_state)
            self.tree[history].particles.append(state_key)

            # No action yet — just compare state to itself
            heuristic_value = self.heuristic2(prev_state, None, prev_state)
            reward = 0
            total = reward + heuristic_value
            self.tree[history].visits += 1
            self.tree[history].total_reward += total
            logging.debug(f"📊 Leaf node initialized — Visits: 1, Total Reward: {total:.2f}")
            return total

        node = self.tree[history]
        state_key = state_to_key(prev_state)
        node.particles.append(state_key)

        # ---------- LEAF EXPANSION ----------
        if len(node.children) == 0:
            k = 3 * ((sim_index // 1000) + 1)
            if rollout_step % k == k - 1:
                action = random.choice(range(env.action_space.n))
                logging.debug(f"🌿 Forced exploration at step {rollout_step}: using random action {action_names[action]}")
            else:
                best_score = float("-inf")
                best_action = None
                for a in range(env.action_space.n):
                    dummy_node = Node()
                    score = dummy_node.ucb_score(node.visits)
                    if score > best_score:
                        best_score = score
                        best_action = a
                action = best_action or random.choice(range(env.action_space.n))
                logging.debug(f"🌾 Leaf node UCB selected action: {action_names[action]}")

            next_state = transition_model(prev_state, action)

            if not is_valid_position(env, next_state['pos']):
                logging.warning(f"❌ Invalid next position: {next_state['pos']} — skipping step.")
                return -1

            if next_state['pos'] == prev_state['pos'] and action == 2:
                logging.debug("⛔ Action blocked — skipping simulation step.")
                reward = -0.1
                done = False
                obs = prev_obs
            else:
                try:
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_state = env_to_state(env)
                except AssertionError:
                    logging.warning(f"❌ env.step failed due to out-of-bounds despite check. Skipping.")
                    return -1

            new_history = history + (action,)
            if new_history not in self.tree:
                self.tree[new_history] = Node(history=new_history)
            node.children[action] = self.tree[new_history]

            heuristic_value = self.heuristic2(prev_state, action, next_state)
            total = reward + heuristic_value

            node.visits += 1
            node.total_reward += total
            logging.debug(f"📈 New leaf node evaluated | Visits: {node.visits}, Total Reward: {node.total_reward:.2f}")
            return total

        # ---------- INTERNAL NODE ----------
        logging.debug(f"🌲 Internal node | Depth: {depth} | Selecting best UCB action")
        best_score = float("-inf")
        best_action = None
        for a in range(env.action_space.n):
            if a in node.children:
                child = node.children[a]
                score = child.ucb_score(node.visits)
                logging.debug(f"  ➡ Action: {action_names[a]} | UCB: {score:.2f} | Visits: {child.visits}")
            else:
                score = float("inf")
            if score > best_score:
                best_score = score
                best_action = a

        assert best_action is not None

        next_state = transition_model(prev_state, best_action)

        if not is_valid_position(env, next_state['pos']):
            logging.warning(f"❌ Invalid internal next position: {next_state['pos']} — skipping.")
            return -1

        if next_state['pos'] == prev_state['pos'] and best_action == 2:
            logging.debug("⛔ Action blocked — skipping simulation step (internal node).")
            reward = -0.1
            done = False
            obs = prev_obs
        else:
            try:
                obs, reward, terminated, truncated, _ = env.step(best_action)
                done = terminated or truncated
                next_state = env_to_state(env)
            except AssertionError:
                logging.warning(f"❌ env.step failed due to out-of-bounds despite check (internal). Skipping.")
                return -1

        new_history = history + (best_action,)
        if new_history not in self.tree:
            self.tree[new_history] = Node(history=new_history)
        node.children[best_action] = self.tree[new_history]

        total_reward = reward
        if not done:
            total_reward += self.simulate(env, new_history, depth - 1, sim_index, rollout_step + 1, prev_obs=obs)

        node.visits += 1
        node.total_reward += total_reward
        logging.debug(f"📊 Updated internal node after action {action_names[best_action]} | Visits: {node.visits}, Total: {node.total_reward:.2f}")

        return total_reward

        def rollout(self, env, depth):
            total_reward = 0
            for step in range(depth):
                if step % 3 == 2:
                    action = 2
                else:
                    weights = [0.1] * 7
                    weights[2] = 0.5
                    action = random.choices(range(7), weights=weights)[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    return total_reward

            # apply heuristic bonus
            bonus = self.heuristic( env_to_state(env))
            return total_reward + bonus

    
    def heuristic(self, state):
        agent = state['pos']
        key = state['key']
        door = state['door']
        goal = state['goal']
        door_open = state['door_open']
        has_key = state['has_key']

        score = 0.0

        # 🧭 חלק 1: קירבה למטרה (goal)
        proximity_score = 0
        if agent and goal:
            dist_to_goal = np.linalg.norm(np.array(agent) - np.array(goal))
            proximity_score = 1 / (1 + dist_to_goal)
            score += proximity_score
            logging.debug(f"🧭 Agent at {agent}, Goal at {goal} → Dist: {dist_to_goal:.2f} → Score: {proximity_score:.2f}")
        else:
            logging.debug("🧭 Missing agent or goal position → No proximity score")

        # 🧱 חלק 2: קירבה לדלת (אם סגורה)
        door_score = 0
        if not door_open and agent and door:
            dist_to_door = np.linalg.norm(np.array(agent) - np.array(door))
            door_score = 0.5 / (1 + dist_to_door)
            score += door_score
            logging.debug(f"🚪 Agent at {agent}, Door at {door} (closed) → Dist: {dist_to_door:.2f} → Score: {door_score:.2f}")
        elif door_open:
            logging.debug("🚪 Door already open → No distance penalty")
        else:
            logging.debug("🚪 Missing agent or door → No door score")

        # 🔑 חלק 3: בונוס אם יש מפתח
        key_bonus = 0.5 if has_key else 0
        score += key_bonus
        logging.debug(f"🔑 Has key: {has_key} → Bonus: {key_bonus}")

        # 🚪 חלק 4: בונוס אם הדלת פתוחה
        door_open_bonus = 1.0 if door_open else 0
        score += door_open_bonus
        logging.debug(f"🪟 Door open: {door_open} → Bonus: {door_open_bonus}")

        # 🏁 חלק 5: בונוס אם הסוכן בדיוק על היעד
        goal_bonus = 5.0 if agent == goal else 0
        score += goal_bonus
        if goal_bonus > 0:
            logging.debug("🏁 Agent reached the goal! → Bonus: 5.00")

        # 🎯 סיכום כללי
        logging.debug(f"🔮 Total heuristic: {score:.2f}")

        return score
    def heuristic2(self, prev_state, action, next_state):
        if prev_state is None or next_state is None:
            return 0.0

        agent = next_state.get('pos')
        key = next_state.get('key')
        door = next_state.get('door')
        goal = next_state.get('goal')
        door_open = next_state.get('door_open', False)
        has_key = next_state.get('has_key', False)

        prev_pos = prev_state.get('pos')
        prev_has_key = prev_state.get('has_key', False)
        prev_door_open = prev_state.get('door_open', False)

        score = 0.0

        # 🧱 עונש אם פעולה לא שינתה מיקום
        if action == 2 and agent == prev_pos:
            score -= 0.5
            logging.debug("⛔ Forward blocked → Penalty: -0.5")

        # ❌ עונשים על פעולות לא חוקיות
        if action == 3:  # pickup
            if has_key or prev_has_key:
                score -= 1.0
                logging.debug("❌ Tried to pickup while holding key → Penalty: -1.0")
            elif prev_pos is None or key is None or np.linalg.norm(np.array(prev_pos) - np.array(key)) > 1:
                score -= 1.0
                logging.debug("❌ Tried to pickup away from key → Penalty: -1.0")

        if action == 5:  # toggle
            if door_open or prev_door_open:
                score -= 0.5
                logging.debug("🔁 Toggled already open door → Penalty: -0.5")
            elif prev_pos is None or door is None or np.linalg.norm(np.array(prev_pos) - np.array(door)) > 1:
                score -= 1.0
                logging.debug("❌ Toggled far from door → Penalty: -1.0")

        # 🧭 קרבה מותנית
        if agent is not None:
            if not has_key and key is not None:
                dist = np.linalg.norm(np.array(agent) - np.array(key))
                key_score = 1.0 / (1.0 + dist)
                score += key_score
                logging.debug(f"🔑 Heading to key → Score: {key_score:.2f}")
            elif not door_open and door is not None:
                dist = np.linalg.norm(np.array(agent) - np.array(door))
                door_score = 1.5 / (1.0 + dist)
                score += door_score
                logging.debug(f"🚪 Heading to door → Score: {door_score:.2f}")
            elif goal is not None:
                dist = np.linalg.norm(np.array(agent) - np.array(goal))
                goal_score = 3.0 / (1.0 + dist)
                score += goal_score
                logging.debug(f"🎯 Heading to goal → Score: {goal_score:.2f}")
            if goal is not None and agent == goal:
                score += 10.0
                logging.debug("🏁 Reached goal → Bonus: +10.0")

        logging.debug(f"🔮 Final heuristic: {score:.2f}")
        return score




def is_adjacent(pos1, pos2):
    if pos1 is None or pos2 is None:
        print(f"⚠️ Tried to compare None positions: {pos1}, {pos2}")
        return False

    # Convert NumPy ints (np.int64) to native Python ints
    x1, y1 = int(pos1[0]), int(pos1[1])
    x2, y2 = int(pos2[0]), int(pos2[1])

    adjacent = abs(x1 - x2) + abs(y1 - y2) == 1
    return adjacent


def transition_model(state, action):
    try:
        logging.debug(f"📥 Incoming state:\n{pprint.pformat(state)}")
        new_state = copy.deepcopy(state)
        logging.debug(f"📥 new state:\n{pprint.pformat(new_state)}")
        ax, ay = state['pos']
        dir = state['dir']
        width, height = 7, 7


        def in_bounds(pos):
            x, y = pos
            return 0 <= x < width and 0 <= y < height
        
        logging.debug(f"📦 Grid preserved: {isinstance(new_state.get('grid'), np.ndarray)}")

        logging.debug(f"🎬 Transition: Action {action} from pos {state['pos']} dir={dir}")

        if action == 0:  # left
            new_state['dir'] = (dir - 1) % 4
            logging.debug(f"↪️ Turn left → dir={new_state['dir']}")

        elif action == 1:  # right
            new_state['dir'] = (dir + 1) % 4
            logging.debug(f"↩️ Turn right → dir={new_state['dir']}")

        elif action == 2:  # forward
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]
            next_pos = (ax + dx, ay + dy)
            if not in_bounds(next_pos):
                logging.debug(f"⛔ Blocked: {next_pos} out of bounds.")
            elif new_state['door'] == next_pos and not new_state.get('door_open', False):
                logging.debug(f"🚪 Blocked by closed door at {next_pos}.")
            else:
                logging.debug(f"✅ Move forward to {next_pos}")
                new_state['pos'] = next_pos

        elif action == 3:  # pickup
            if new_state['pos'] == new_state['key']:
                new_state['has_key'] = True
                new_state['key'] = None
                logging.debug("🔑 Picked up key.")
            else:
                logging.debug("❌ No key to pick up.")

        elif action == 4:  # drop
            if new_state.get('has_key', False):
                new_state['has_key'] = False
                new_state['key'] = new_state['pos']
                logging.debug(f"📤 Dropped key at {new_state['pos']}")
            else:
                logging.debug("❌ Tried to drop key but didn't have it.")

        elif action == 5:  # toggle
            if is_adjacent(new_state['pos'], new_state['door']) and new_state.get('has_key', False):
                new_state['door_open'] = True
                logging.debug("🚪 Door opened successfully.")
            else:
                logging.debug("❌ Toggle failed (not adjacent or no key).")

        elif action == 6:  # done
            logging.debug("🏁 Done action — no effect.")

        return new_state
    except Exception as e:
        logging.warning(f"⚠️ transition_model failed — returning original state. Error: {e}")
        return state

def observation_model(state, action, obs):
    try:
        facts = extract_semantic_facts_from_obs(obs)
        pos, dir, key_pos, door_pos, has_key, door_open = state

        # אם אני רואה שאין מפתח בתמונה – זרוק מצבים שיש בהם מפתח בטווח הראייה
        if "key" not in facts["visible_objects"] and key_pos is not None:
            return 0.0

        if "door" not in facts["visible_objects"] and door_pos is not None:
            return 0.0

        if facts["has_key"] != has_key:
            return 0.0

        if facts["door_open"] is not None and facts["door_open"] != door_open:
            return 0.0

        return 1.0  # מצב תואם את כל מה שאני יכול לדעת מהתמונה

    except Exception as e:
        print(f"⚠️ Semantic observation match failed: {e}")
        return 0.0

def main():
    logger = logging.getLogger(__name__)
    env_name = "MiniGrid-DoorKey-5x5-v0"
    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset()

    print(f"\n🚀 Environment {env_name} initialized.")
    assert isinstance(env.unwrapped, MiniGridEnv)
    agent_pos = env.unwrapped.agent_pos
    print(f"🧍 Initial agent position: {agent_pos}")

    belief = Belief()
    states = generate_states_with_partial_observability(env.unwrapped, agent_pos)
    print(f"📊 Initial possible states generated: {len(states)}")
# 🗺️ Print each state in a readable format
    def symbol_from_type(t):
        type_name = IDX_TO_OBJECT.get(t, "unknown")
        return {
            "empty": "⬜",
            "wall": "🧱",
            "door": "🚪",
            "key": "🗝️",
            "goal": "🎯",
            "agent": "🤖",
        }.get(type_name, "❓")

    for i, state in enumerate(states):
        try:
            pos = state['pos']
            direction = state['dir']
            has_key = state['has_key']
            door_open = state['door_open']
            grid = state.get('grid')

            logger.info(f"{i:03d}) pos={pos}, dir={direction}, has_key={has_key}, door_open={door_open}")

            if grid is not None:
                logger.info(f"🗺️ Grid for state {i:03d}:")
                for row in grid:
                    symbols = [symbol_from_type(cell) for cell in row]
                    logger.info("".join(symbols))
            else:
                logger.warning(f"⚠️ No grid in state {i:03d}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to print state {i}: {state} | Error: {e}")

    state_map = {}
    for state in states:
        key = state_to_key(state)
        logger.info(f"🗺️ key for state {key}:")
        state_map[key] = state
        belief.belief[key] = 1.0 / len(states)
    # for s in states:
    #     belief.belief[state_to_key(s)] = 1.0
    # belief.normalize()
    print(f"✅ Initial belief normalized. Number of states: {len(belief.belief)}")

    pomcp = POMCP(env_name)
    done = False
    step_num = 0

    while not done:
        print(f"\n🔁 Step {step_num} — Beginning planning phase...")
        actions = pomcp.plan(belief,state_map)
        print(f"\n🎬 Actions planned: {[action_names[a] for a in actions]}")

        for i, action in enumerate(actions):
            print(f"\n✅ Executing action {i+1}/{len(actions)}: {action_names[action]}")
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"📷 Observation shape: {obs['image'].shape}")
            print(f"🏅 Reward received: {reward}")
            print(f"🧠 Belief size before update: {len(belief.belief)}")

            belief.update(belief.belief, action, obs, transition_model, observation_model,state_map)

            print(f"🧠 Belief size after update: {len(belief.belief)}")

            done = terminated or truncated
            if done:
                print("\n🏁 Episode finished.")
                break

        step_num += 1

    env.close()
    print("✅ Environment closed.")

if __name__ == "__main__":
    main()

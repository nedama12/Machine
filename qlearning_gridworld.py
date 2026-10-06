import random
from collections import deque

import numpy as np
from sklearn.linear_model import SGDRegressor

GRID_LAYOUT = [
    "Aoo#ooDooo",
    "o#oooo#oDo",
    "oDo#o#ooDo",
    "ooo#ooDo#o",
    "#oDoo#Do#o",
    "oo#oooooo#",
    "ooo#D#oo#o",
    "#oDooDo#oo",
    "oooo#ooooo",
    "o#oooo#ooT",
]

GRID_ROWS = len(GRID_LAYOUT)
GRID_COLS = len(GRID_LAYOUT[0])

START_STATE = (0, 0)
GOAL_STATE = (GRID_ROWS - 1, GRID_COLS - 1)

ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_DELTAS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

REWARD_NORMAL = -1
REWARD_INVALID = -5
REWARD_WALL = -10
REWARD_DANGER = -15
REWARD_GOAL = 100

REWARD_TABLE = [
    ("Moving to a valid normal cell (o)", "Path", REWARD_NORMAL),
    ("Trying to leave the grid", "Invalid move", REWARD_INVALID),
    ("Hitting a wall (#)", "Wall", REWARD_WALL),
    ("Entering a Danger Zone (D)", "Danger Zone", REWARD_DANGER),
    ("Reaching the target (T)", "Target", REWARD_GOAL),
]

EPISODES = 1000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.994
LEARNING_RATE = 0.05
MAX_STEPS_TRAINING = 200
MAX_STEPS_EVALUATION = 100
BATCH_SIZE = 32
LEARN_EVERY = 4
REPLAY_SIZE = 5000
SEED = 42

N_FEATURES = GRID_ROWS * GRID_COLS * len(ACTIONS)


def cell_at(state):
    return GRID_LAYOUT[state[0]][state[1]]


def feature_index(state, action):
    return (state[0] * GRID_COLS + state[1]) * len(ACTIONS) + ACTIONS.index(action)


def encode(state, action):
    features = np.zeros(N_FEATURES)
    features[feature_index(state, action)] = 1.0
    return features


def step(state, action):
    """Applies one action and returns (next_state, cell_type, reward, done)."""
    row = state[0] + ACTION_DELTAS[action][0]
    col = state[1] + ACTION_DELTAS[action][1]

    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        return state, "Invalid move", REWARD_INVALID, False

    cell = GRID_LAYOUT[row][col]

    if cell == "#":
        return state, "Wall", REWARD_WALL, False

    if cell == "T":
        return (row, col), "Target", REWARD_GOAL, True

    if cell == "D":
        return (row, col), "Danger Zone", REWARD_DANGER, False

    return (row, col), "Path", REWARD_NORMAL, False


class QAgent:
    def __init__(self, gamma, epsilon, epsilon_min, epsilon_decay, rng):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng

        self.model = SGDRegressor(
            penalty=None,
            fit_intercept=False,
            learning_rate="constant",
            eta0=LEARNING_RATE,
        )
        # One initial call so predict() is available before any experience.
        self.model.partial_fit(
            [encode(START_STATE, a) for a in ACTIONS], [0.0] * len(ACTIONS)
        )

    def q_values(self, state):
        features = np.array([encode(state, a) for a in ACTIONS])
        return self.model.predict(features)

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)

        values = self.q_values(state)
        best = np.flatnonzero(values == values.max())
        return ACTIONS[int(self.rng.choice(list(best)))]

    def learn(self, batch):
        features = np.zeros((len(batch), N_FEATURES))
        targets = np.zeros(len(batch))

        # Row i of q_matrix holds the four Q-values of state i.
        q_matrix = self.model.coef_.reshape(-1, len(ACTIONS))

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = reward

            if not done:
                next_index = next_state[0] * GRID_COLS + next_state[1]
                target += self.gamma * q_matrix[next_index].max()

            features[i, feature_index(state, action)] = 1.0
            targets[i] = target

        self.model.partial_fit(features, targets)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes=EPISODES):
    rng = random.Random(SEED)
    agent = QAgent(GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, rng)
    memory = deque(maxlen=REPLAY_SIZE)

    successes = 0
    total_steps = 0
    rewards = []

    for _ in range(episodes):
        state = START_STATE
        total = 0

        for _ in range(MAX_STEPS_TRAINING):
            action = agent.choose_action(state, explore=True)
            next_state, _, reward, done = step(state, action)

            memory.append((state, action, reward, next_state, done))
            total_steps += 1
            total += reward
            state = next_state

            if len(memory) >= BATCH_SIZE and total_steps % LEARN_EVERY == 0:
                agent.learn(rng.sample(memory, BATCH_SIZE))

            if done:
                successes += 1
                break

        rewards.append(total)
        agent.decay()

    stats = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": 100 * successes / episodes,
        "average_reward": sum(rewards) / episodes,
        "final_epsilon": agent.epsilon,
    }

    return agent, stats


def evaluate(agent):
    """Runs the greedy policy (no exploration) and records every step."""
    state = START_STATE
    rows = []
    path = [state]
    total = 0
    reached = False

    for number in range(1, MAX_STEPS_EVALUATION + 1):
        action = agent.choose_action(state, explore=False)
        next_state, cell, reward, done = step(state, action)

        rows.append({
            "step": number,
            "state": state,
            "action": action,
            "next_state": next_state,
            "cell": cell,
            "reward": reward,
        })

        total += reward
        state = next_state
        path.append(state)

        if done:
            reached = True
            break

    return {
        "rows": rows,
        "path": path,
        "movements": len(rows),
        "total_reward": total,
        "danger_hits": sum(1 for r in rows if r["cell"] == "Danger Zone"),
        "reached": reached,
    }


def q_table(agent):
    table = []

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r][c] == "#":
                continue

            values = agent.q_values((r, c))
            table.append({
                "state": (r, c),
                "cell": GRID_LAYOUT[r][c],
                "values": [float(v) for v in values],
                "best": ACTIONS[int(np.argmax(values))],
            })

    return table


def run_session():
    agent, stats = train()
    return {
        "stats": stats,
        "evaluation": evaluate(agent),
        "q_table": q_table(agent),
    }


def build_grid(path=None):
    on_path = set(path or [])
    order = {s: i for i, s in enumerate(path or [])}

    return [
        [
            {
                "char": GRID_LAYOUT[r][c],
                "on_path": (r, c) in on_path,
                "order": order.get((r, c)),
            }
            for c in range(GRID_COLS)
        ]
        for r in range(GRID_ROWS)
    ]


def cell_counts():
    flat = "".join(GRID_LAYOUT)
    return {ch: flat.count(ch) for ch in "ATo#D"}

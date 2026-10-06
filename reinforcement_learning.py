import random

import numpy as np
from sklearn.linear_model import SGDRegressor

GRID_ROWS = 10
GRID_COLS = 10

GRID_LAYOUT = [
    "Ao##oooD##",
    "oD#o#oo#oo",
    "DooD#Doooo",
    "DDooooooDo",
    "oo#ooooo##",
    "ooo#o#oooo",
    "Dooo#oo#oo",
    "ooooooo#oo",
    "o#oDooo#oo",
    "o#oo#ooooT",
]

START_STATE = (0, 0)
GOAL_STATE = (9, 9)

REWARD_STEP = -1
REWARD_INVALID_MOVE = -5
REWARD_WALL = -10
REWARD_DANGER_ZONE = -20
REWARD_GOAL = 100

MAX_STEPS_PER_EPISODE = 200
MAX_STEPS_EVALUATION = 150

ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_DELTAS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

DEFAULT_EPISODES = 800
DEFAULT_GAMMA = 0.95
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_MIN = 0.05
DEFAULT_EPSILON_DECAY = 0.99
DEFAULT_LEARNING_RATE = 0.3

NUM_STATES = GRID_ROWS * GRID_COLS
NUM_STATE_ACTIONS = NUM_STATES * len(ACTIONS)


def cell_type(row, col):
    return GRID_LAYOUT[row][col]


def is_inside_grid(row, col):
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def state_action_index(state, action):
    row, col = state
    return (row * GRID_COLS + col) * len(ACTIONS) + ACTIONS.index(action)


def encode_state_action(state, action):
    features = [0.0] * NUM_STATE_ACTIONS
    features[state_action_index(state, action)] = 1.0
    return features


def step(state, action):
    row, col = state
    dr, dc = ACTION_DELTAS[action]
    next_row, next_col = row + dr, col + dc

    if not is_inside_grid(next_row, next_col):
        return state, "invalid_move", REWARD_INVALID_MOVE, False

    target_cell = cell_type(next_row, next_col)

    if target_cell == "#":
        return state, "wall", REWARD_WALL, False

    next_state = (next_row, next_col)

    if target_cell == "T":
        return next_state, "goal", REWARD_GOAL, True

    if target_cell == "D":
        return next_state, "danger_zone", REWARD_DANGER_ZONE, False

    return next_state, "path", REWARD_STEP, False


class QLearningAgent:
    """
    Estimates Q(s, a) with an SGDRegressor trained through incremental
    (partial_fit) updates, following the same approach used in the
    reference classroom exercise: predict() reads the current Q-value
    estimates and partial_fit() applies the Q-learning correction
    after every observed transition.
    """

    def __init__(
        self,
        gamma=DEFAULT_GAMMA,
        epsilon_start=DEFAULT_EPSILON_START,
        epsilon_min=DEFAULT_EPSILON_MIN,
        epsilon_decay=DEFAULT_EPSILON_DECAY,
        learning_rate=DEFAULT_LEARNING_RATE,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = SGDRegressor(
            learning_rate="constant",
            eta0=learning_rate,
            max_iter=1,
            warm_start=True,
            fit_intercept=False,
            penalty=None,
        )

        self._coef = np.zeros(NUM_STATE_ACTIONS)
        self._bootstrap_model()

    def _bootstrap_model(self):
        sample_features = [
            encode_state_action(START_STATE, action) for action in ACTIONS
        ]
        sample_targets = [0.0 for _ in ACTIONS]
        self.model.partial_fit(sample_features, sample_targets)
        self._coef = self.model.coef_

    def predict_q_values(self, state):
        """Reads Q(s, a) for every action directly from the trained weights."""
        return {
            action: float(self._coef[state_action_index(state, action)])
            for action in ACTIONS
        }

    def predict_q_values_via_model(self, state):
        """Same as predict_q_values but calling model.predict() explicitly."""
        features = [encode_state_action(state, action) for action in ACTIONS]
        predictions = self.model.predict(features)
        return dict(zip(ACTIONS, predictions))

    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(ACTIONS)

        q_values = self.predict_q_values(state)
        return max(q_values, key=q_values.get)

    def apply_updates(self, batch_features, batch_targets):
        self.model.partial_fit(batch_features, batch_targets)
        self._coef = self.model.coef_

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(episodes=DEFAULT_EPISODES, gamma=DEFAULT_GAMMA,
                 epsilon_start=DEFAULT_EPSILON_START,
                 epsilon_min=DEFAULT_EPSILON_MIN,
                 epsilon_decay=DEFAULT_EPSILON_DECAY):

    agent = QLearningAgent(
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    episode_rewards = []
    successful_episodes = 0

    for _ in range(episodes):
        state = START_STATE
        total_reward = 0.0

        batch_features = []
        batch_targets = []

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state, explore=True)
            next_state, cell, reward, done = step(state, action)

            if done:
                target = float(reward)
            else:
                next_q_values = agent.predict_q_values(next_state)
                target = reward + gamma * max(next_q_values.values())

            batch_features.append(encode_state_action(state, action))
            batch_targets.append(target)

            total_reward += reward
            state = next_state

            if done:
                successful_episodes += 1
                break

        agent.apply_updates(batch_features, batch_targets)
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    average_reward = sum(episode_rewards) / len(episode_rewards)

    return agent, {
        "episodes": episodes,
        "successful_episodes": successful_episodes,
        "success_rate": round(100 * successful_episodes / episodes, 2),
        "average_reward": round(average_reward, 2),
        "final_epsilon": round(agent.epsilon, 4),
    }


def evaluate_policy(agent, max_steps=MAX_STEPS_EVALUATION):
    state = START_STATE
    path_log = []
    path_cells = [state]
    total_reward = 0.0
    reached_goal = False

    for step_number in range(1, max_steps + 1):
        action = agent.select_action(state, explore=False)
        next_state, cell, reward, done = step(state, action)

        path_log.append({
            "step": step_number,
            "state": state,
            "action": action,
            "next_state": next_state,
            "cell_type": cell,
            "reward": reward,
        })

        total_reward += reward
        state = next_state
        path_cells.append(state)

        if done:
            reached_goal = True
            break

    return {
        "path_log": path_log,
        "path_cells": path_cells,
        "total_reward": round(total_reward, 2),
        "steps_taken": len(path_log),
        "reached_goal": reached_goal,
    }


def collect_q_values(agent):
    q_table = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if cell_type(row, col) == "#":
                continue

            q_values = agent.predict_q_values_via_model((row, col))
            q_table.append({
                "state": (row, col),
                "up": round(float(q_values["Up"]), 2),
                "down": round(float(q_values["Down"]), 2),
                "left": round(float(q_values["Left"]), 2),
                "right": round(float(q_values["Right"]), 2),
            })

    return q_table


def run_training_session(episodes=DEFAULT_EPISODES, gamma=DEFAULT_GAMMA,
                          epsilon_start=DEFAULT_EPSILON_START,
                          epsilon_min=DEFAULT_EPSILON_MIN,
                          epsilon_decay=DEFAULT_EPSILON_DECAY):

    agent, training_stats = train_agent(
        episodes=episodes,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    evaluation = evaluate_policy(agent)
    q_table = collect_q_values(agent)

    return {
        "training": training_stats,
        "evaluation": evaluation,
        "q_table": q_table,
        "grid": GRID_LAYOUT,
    }

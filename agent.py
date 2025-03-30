import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class State:
    points: int
    opponent_points: int
    wins: int
    opponent_wins: int
    current_round: int


class Trainer:
    def choose_action(self, state: State):
        action = -1
        while True:
            if state.opponent_points < state.points:
                action = round(
                    np.random.normal(loc=state.points + 2, scale=5, size=1)[0]
                )
                # action = np.random.randint(0, state.points)
            else:
                action = round(
                    np.random.normal(loc=state.points - 2, scale=5, size=1)[0]
                )
                # action = np.random.randint(0, state.points)
            if action >= 0 and action <= state.points:
                return action


class Player:
    def __init__(self, name, points, max_rounds, alpha=0.5, gamma=0.9):
        self.name = name
        self.init_points = points
        self.points = points
        self.round = 0
        self.wins = 0
        self.current_game_history: List[Tuple[State, int, float]] = []
        self.alpha = alpha
        self.gamma = gamma

        max_possible_reward = 10  # e.g., from your final game reward
        # my points, opponent points, my wins, current round, action/bid_amount
        self.q_table = np.random.uniform(
            low=0,  # Encourage exploration
            high=max_possible_reward,
            size=(
                points + 1,
                points + 1,
                max_rounds + 1,
                max_rounds + 1,
                max_rounds + 1,
                points + 1,
            ),
        )

    def update(self, final_reward: float):
        # Propagate final reward to all steps
        for t in reversed(range(len(self.current_game_history))):
            state, action, original_reward = self.current_game_history[t]

            # Blend final reward with original reward
            discount = self.gamma ** (len(self.current_game_history) - t - 1)
            blended_reward = original_reward + discount * final_reward

            # Get next state
            next_state = (
                self.current_game_history[t + 1][0]
                if t < len(self.current_game_history) - 1
                else None
            )

            # Update Q-value with blended reward
            self.reward(state, action, blended_reward, next_state)

    def reward(
        self, state: State, action: int, reward: float, next_state: State | None
    ):
        current_q = self.q_table[
            state.points,
            state.opponent_points,
            state.wins,
            state.opponent_wins,
            state.current_round,
            action,
        ]

        # Max Q-value for the next state (if next_state exists)
        if next_state is None:
            # Terminal state (end of game): no future rewards
            max_next_q = 0
        else:
            # Get all possible actions (bids) in next state
            valid_next_actions = range(min(next_state.points, 50) + 1)
            if valid_next_actions:
                # Extract Q-values for valid actions in next state
                next_q_slice = self.q_table[
                    next_state.points,
                    next_state.opponent_points,
                    next_state.wins,
                    next_state.opponent_wins,
                    next_state.current_round,
                    :,  # All actions
                ]
                # Mask invalid bids (those exceeding remaining points)
                valid_next_q = next_q_slice[valid_next_actions]
                max_next_q = np.max(valid_next_q)
            else:
                max_next_q = 0.0

        # Calculate the target Q-value using Bellman equation
        target = reward + self.gamma * max_next_q

        # Update Q-value for (state, action)
        self.q_table[
            state.points,
            state.opponent_points,
            state.wins,
            state.opponent_wins,
            state.current_round,
            action,
        ] += self.alpha * (target - current_q)

    def new_game(self):
        self.current_game_history = []

    def add_round(self, history):
        self.current_game_history.append(history)

    def choose_action(self, state: State, epsilon: float):
        if state.points > 0:
            valid_actions = list(
                range(1, min(state.points, 50) + 1)
            )  # 1–min(points,50)
        else:
            valid_actions = list(
                range(0, min(state.points, 50) + 1)
            )  # 0–min(points,50)

        if state.points == self.points and state.opponent_points == self.points:
            action = np.random.randint(2, 10)
        elif np.random.rand() < epsilon:
            # Explore: Random valid bid
            action = np.random.choice(valid_actions)
        else:
            # Exploit: Best valid bid according to Q-table
            q_slice = self.q_table[
                state.points,
                state.opponent_points,
                state.wins,
                state.opponent_wins,
                state.current_round,
                :,  # All actions
            ]
            # # Mask invalid actions (bids > remaining points)
            valid_q_values = q_slice[valid_actions]
            best_action_idx = np.argmax(valid_q_values)
            action = valid_actions[best_action_idx]

        return action

    def load(self):
        self.q_table = np.load(f"{self.name}.npy")

    def save(self):
        np.save(self.name, self.q_table)

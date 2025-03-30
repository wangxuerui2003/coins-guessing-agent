import numpy as np
from agent import Player, State, Trainer

POINTS = 50
ROUNDS = 7


class Model:
    def __init__(
        self,
        agent_name="agent",
        epsilon=0.99,
        epsilon_decay=0.999,
        lr=0.5,
        discount_factor=0.95,
        epochs=10000,
    ):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.discount_factor = discount_factor
        self.max_epochs = epochs
        self.epoch = 0
        self.agent = Player(agent_name, points=POINTS, max_rounds=ROUNDS)
        self.trainer = Trainer()
        self.agent_wins = 0
        self.trainer_wins = 0
        self.draws = 0

    def decay(self):
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

    def train(self):
        for e in range(self.max_epochs):
            self.epoch = e
            agent_wins = 0
            trainer_wins = 0
            agent_points = POINTS
            trainer_points = POINTS
            for r in range(ROUNDS):
                agent_state = State(
                    agent_points, trainer_points, agent_wins, trainer_wins, r
                )
                agent_action = self.agent.choose_action(agent_state, self.epsilon)

                trainer_state = State(
                    trainer_points, agent_points, trainer_wins, agent_wins, r
                )
                trainer_action = self.trainer.choose_action(trainer_state)

                agent_points -= agent_action
                trainer_points -= trainer_action

                if agent_action == trainer_action:
                    self.agent.add_round((agent_state, agent_action, 0))
                    continue

                winner = 1 if agent_action > trainer_action else 2
                if winner == 1:
                    agent_wins += 1
                else:
                    trainer_wins += 1
                self.agent.add_round((
                    agent_state,
                    agent_action,
                    +1 if winner == 1 else -1,
                ))

            if agent_wins == trainer_wins:
                self.decay()
                agent_final_reward = -2
                self.agent.update(agent_final_reward)
                self.agent.new_game()
                self.draws += 1
                continue

            # final reward for the game
            agent_final_reward = 10 if agent_wins > trainer_wins else -10
            self.agent.update(agent_final_reward)
            self.decay()
            self.agent.new_game()

            if agent_wins > trainer_wins:
                self.agent_wins += 1
            else:
                self.trainer_wins += 1

        print("Player 1 wins:", self.agent_wins)
        print("Player 2 wins:", self.trainer_wins)
        print("Draws:", self.draws)

    def save(self):
        self.agent.save()

import numpy as np
from agent import Player, State

ROUNDS = 7

p1_points = 50
p2_points = 50
p1_wins = 0
p2_wins = 0

agent = Player("agent", 50, ROUNDS)
agent.load()
print(agent.q_table)


def bid(player_num: int, points: int):
    while True:
        try:
            bid = int(input(f"Player {player_num} bid (0 - {points}): "))
            if bid > points:
                print("Not enough points")
                raise ValueError
            return bid
        except ValueError:
            pass
        except EOFError:
            exit(0)


for round in range(ROUNDS):
    print("Round", round + 1)

    p1_bid = bid(1, p1_points)
    # p2_bid = bid(2, p2_points)
    # p2_points -= p2_bid

    # use AI
    p2_state = State(p2_points, p1_points, p2_wins, p1_wins, round)
    p2_bid = agent.choose_action(p2_state, 0.01)
    print(p2_bid)
    p1_points -= p1_bid
    p2_points -= p2_bid
    if p1_bid > p2_bid:
        p1_wins += 1
        print("Player 1 won this round!")
    elif p2_bid > p1_bid:
        p2_wins += 1
        print("Player 2 won this round!")
    else:
        print("Draw!")

if p1_wins > p2_wins:
    print("Player 1 won!!!")
elif p2_wins > p1_wins:
    print("Player 2 won!!!")
else:
    print("Draw!!!!!!!!")

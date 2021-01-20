import random
import time
from collections import defaultdict

from alpha_0_algorithm import alpha_0_pipeline
from connect_n import Game
from agent import Agent

from hyperparameters import *


def play_game(game, seed, oponent, version=0):
    agent_1 = Agent(oponent,seed=seed, version=version, scnds_per_move=2, game=game, player=-1)
    agent_2 = Agent("user",seed=seed, player=1)
    game.start_game(agent_1,agent_2,1)


def tournament(game, seed, latest_alpha0_version):
    game.show_output = False
    agents = []
    for version in range(0,latest_alpha0_version+1):
        agents.append(Agent("alphaZero",seed=seed, version=version, scnds_per_move=0.1, game=game, player=1))
    # agents.append(Agent("mcts",seed=seed, player=1, scnds_per_move=0.1, game = game))
    # agents.append(Agent("random", seed=seed, player=-1,scnds_per_move=0.1, game = game))
    nr_games = 10
    statistics = defaultdict(lambda: 0)
    nr_games_played = 0
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            player_first_agent = -1
            agent_1 = agents[i]
            agent_2 = agents[j]
            agent_1.player = player_first_agent
            agent_2.player = -player_first_agent
            for h in range(nr_games):
                game.reset()
                game.start_game(agent_1, agent_2)
                nr_games_played += 1
                print("\r\t{} games played".format(nr_games_played),end="")
                statistics[i, j, game.winner*player_first_agent] = statistics[i, j, game.winner*player_first_agent]+1

    # print results
    print("\r")
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            print(f"{agents[i].name} vs {agents[j].name}:\t{statistics[i, j, 1]} wins {statistics[i, j, 0]} ties {statistics[i, j, -1]} losses")




if __name__ == '__main__':
    show_output = False
    name = "Connect_4"
    sleeping_time = 0


    seed = random.randint(0, 100000)
    print("seed =", seed, "device =", DEVICE)
    seed = 80977
    random.seed(seed)

    # game = Game(random, 3, 3, 3, gravity_on=False, show_output=show_output, sleeping_time=sleeping_time) # connect 3
    game = Game(random, 4, 6, 7, gravity_on=True, show_output=show_output, sleeping_time=sleeping_time)  # connect 4

    # play a tournament
    # tournament(game, seed, 3)


    # play a game (options for oponents include "mcts", "alphazero", "random", "user")
    # play_game(game, seed, oponent="mcts")

    # train an agent
    alpha_0_pipeline(3, game, name, seed)

import random
import time
from collections import defaultdict

from alpha_0_algorithm import alpha_0_pipeline
from connect_n import Game
from agent import Agent

from hyperparameters import *


def play_game(game, seed, opponent, version=0):
    agent_1 = Agent(opponent, seed=seed, version=version, scnds_per_move=2, game=game, player=-1)
    agent_2 = Agent("user",seed=seed, player=1)
    game.start_game(agent_1,agent_2,1)


def tournament(game, seed, latest_alpha0_version, every_version=1):
    game.show_output = False
    agents = []
    # for version in range(0,latest_alpha0_version+1):
    #     if version%every_version == 0:
    #         agents.append(Agent("alphaZero",seed=seed, version=version, scnds_per_move=0.1, game=game, player=1))
    agents.append(Agent("alphaZero", seed=seed, version=16, scnds_per_move=0.1, game=game, player=1))
    agents.append(Agent("alphaZero", seed=seed, version=16, scnds_per_move=0.1, game=game, player=1))
    # agents.append(Agent("mcts",seed=seed, player=1, scnds_per_move=0.1, game = game))
    # agents.append(Agent("random", seed=seed, player=-1,scnds_per_move=0.1, game = game))
    nr_games = 100
    statistics = defaultdict(lambda: 0)
    total_number_wins = defaultdict(lambda: 0)
    nr_games_played = 0
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            player_first_agent = -1
            agent_1 = agents[i]
            agent_2 = agents[j]
            agent_1.set_player(player_first_agent)
            agent_2.set_player(-player_first_agent)
            for h in range(nr_games):
                game.reset()
                game.start_game(agent_1, agent_2)
                nr_games_played += 1
                print("\r\t{} games played".format(nr_games_played),end="")
                statistics[i, j, game.winner*player_first_agent] = statistics[i, j, game.winner*player_first_agent]+1
                if player_first_agent==game.winner:
                    total_number_wins[i] = total_number_wins[i]+1
                if -player_first_agent==game.winner:
                    total_number_wins[j] = total_number_wins[j]+1

    # print results
    print("\r")
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            print(f"{agents[i].name} vs {agents[j].name}:\t{statistics[i, j, 1]} wins {statistics[i, j, 0]} ties {statistics[i, j, -1]} losses")
    for i in range(len(agents)):
        print(f"{agents[i].name} nr wins: {total_number_wins[i]}")




if __name__ == '__main__':
    show_output = True
    name = "Connect_4"
    sleeping_time = 0


    seed = random.randint(0, 100000)
    print("seed =", seed, "device =", DEVICE)
    random.seed(seed)

    # game = Game(random, 3, 3, 3, gravity_on=False, show_output=show_output, sleeping_time=sleeping_time) # connect 3
    game = Game(random, 4, 6, 7, gravity_on=True, show_output=show_output, sleeping_time=sleeping_time)  # connect 4

    # play a tournament
    # tournament(game, seed, 16, 1)


    # play a game (options for oponents include "mcts", "alphaZero", "random", "user")
    play_game(game, seed, opponent="mcts", version=1)

    # train an agent (-1 as first argument for starting from scratch)
    # alpha_0_pipeline(-1, game, name, seed)

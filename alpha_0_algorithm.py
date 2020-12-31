import time

from agent import Agent
from hyperparameters import *


def self_play(agent):
    pass


# play a game and return true if agent wins
def new_agent_wins(agent, agent_old):
    pass


def alpha_0_algorithm(start_version):
    actual_version = start_version
    agent = Agent(actual_version)
    while True:
        # self play for collecting games
        start_time = time.time()
        for i in range(1, NUMBER_GAMES_PER_SELF_PLAY + 1):
            self_play(agent)
            print('\r\t # games played {} ( out of {}) \t time used so far: {}'.format(i, NUMBER_GAMES_PER_SELF_PLAY,
                                                                                       (time.time() - start_time)),
                  end="")
        # learning from games to improve neural network
        start_time = time.time()
        for i in range(1, NUMBER_OF_BATCHES + 1):
            agent.train()
            print('\r\t # batches trained {} ( out of {}) \t time used so far: {}'.format(i, NUMBER_OF_BATCHES,
                                                                                          (time.time() - start_time)),
                  end="")
        # play vs older version
        if actual_version > 0:
            agent_old = Agent(actual_version)
            number_wins_new_agent = 0
            for i in range(1, NUMBER_GAMES_VS_OLD_VERSION + 1):
                if new_agent_wins(agent, agent_old):
                    number_wins_new_agent += 1
            if number_wins_new_agent >= WIN_PERCENTAGE * NUMBER_GAMES_VS_OLD_VERSION:
                # accept new version
                actual_version += 1
                agent.save(actual_version)
                print("version " + str(actual_version) + " accepted")
            else:
                agent = agent_old

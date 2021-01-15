import copy
import time
from itertools import repeat
from multiprocessing import Pool

from agent import Agent
from helper import compute_search_probabilities
from hyperparameters import *

from pathos.multiprocessing import ProcessingPool as Pool

# playing against itself to fill the replay buffer
from replay_buffer import Replay_buffer


def self_play_parallel(agent, game):
    start_time = time.time()
    pool = Pool(processes=NR_PROCESSES_ON_CPU)

    args = [(agent.clone(), game.clone(), True) for i in range(NR_PROCESSES_ON_CPU)]
    pool.map(self_play_process, args)
    pool.close()
    pool.join()

    print("\r\tSelf play completed\t time for Self play: {} seconds".format((time.time() - start_time)))


def self_play(agent, game):
    start_time = time.time()
    self_play_process((agent, game, False))
    print("\r\tSelf play completed\t time for Self play: {} seconds".format((time.time() - start_time)))


def self_play_process(args):
    start_time = time.time()
    agent_1, game, parallel = args
    agent_2 = agent_1.clone()
    agent_2.player = -agent_2.player
    agents = [agent_1, agent_2]
    nr_episodes = NUMBER_GAMES_PER_SELF_PLAY
    if parallel:
        nr_episodes = nr_episodes // NR_PROCESSES_ON_CPU
    for episode in range(1, nr_episodes + 1):
        game.reset()
        agent_1.mcts.reset()
        agent_2.mcts.reset()
        start_agent = agents[game.random.randint(0, 1)]
        crnt_agent = start_agent
        step = 0
        states_so_far_list = []
        crnt_player_list = []
        search_probabilities_list = []
        while game.winner is None:
            step += 1
            if len(game.feasible_actions) == 0:
                game.winner = 0
                break
            action = crnt_agent.compute_action(game, True)
            # if crnt_agent.player == 1:
            states_so_far_list.append(copy.deepcopy(game.all_board_states))
            crnt_player_list.append(crnt_agent.player)
            search_probabilities = compute_search_probabilities(crnt_agent, game)
            search_probabilities_list.append(search_probabilities)
            game.step_if_feasible(action, crnt_agent.player)
            crnt_agent = agent_1 if crnt_agent is agent_2 else agent_2
        # store information in shared rpb
        outcome = game.winner
        for states_so_far, crnt_player, search_probabilities in zip(states_so_far_list, crnt_player_list,
                                                                    search_probabilities_list):
            agent_1.replay_buffer.add_experience(states_so_far, search_probabilities, outcome, crnt_player,
                                                 game.state_shape)
        if not parallel:
            print("\r\tSelf play - {}% done\t time used so far: {} seconds".format(
                (episode + 1) / NUMBER_GAMES_PER_SELF_PLAY * 100, (time.time() - start_time)), end="")


# play a game and return true if agent wins
def play_game_return_winner(agent, agent_old, game):
    game.reset()
    agents = [agent, agent_old]
    crnt_agent = agents[game.random.randint(0, 1)]
    while game.winner is None:
        if len(game.feasible_actions) == 0:
            game.winner = 0
            break
        action = crnt_agent.compute_action(game, False)
        game.step_if_feasible(action, crnt_agent.player)
        crnt_agent = agent if crnt_agent is agent_old else agent_old
    return game.winner * agent.player


# evaluate version by playing vs previous version
def evaluate(agent, game):
    start_time = time.time()
    # play vs older version for evaluation
    agent_old = Agent(type="alphaZero", player=-1, seed=agent.seed, version=agent.version - 1,
                      scnds_per_move=SCNDS_PER_MOVE_TRAINING,
                      game=game, name_for_saving=agent.name_for_saving)
    number_wins_new_agent = 0
    number_ties = 0
    for i in range(1, NUMBER_GAMES_VS_OLD_VERSION + 1):
        outcome = play_game_return_winner(agent, agent_old, game)
        if outcome == 1:
            number_wins_new_agent += 1
        elif outcome == 0:
            number_ties += 1
    # ties count as wins times a weight, otherwise it is too hard to be accepted because of the high tie probability in some games
    if number_wins_new_agent + number_ties * WEIGHT_FOR_TIES_IN_EVALUATION >= WIN_PERCENTAGE * NUMBER_GAMES_VS_OLD_VERSION / 100:
        # accept new version
        agent.save(agent.version)
        print("version {} accepted with win probability: {}% and tie probability: {}%".format(agent.version,
                                                                                              number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION * 100,
                                                                                              number_ties / NUMBER_GAMES_VS_OLD_VERSION * 100))
    else:
        agent = Agent(type="alphaZero", player=1, seed=agent.seed, version=agent.version - 1,
                      scnds_per_move=SCNDS_PER_MOVE_TRAINING,
                      game=game, name_for_saving=agent.name_for_saving)
        print("version {} refused with win probability: {}% and tie probability: {}%".format(agent.version,
                                                                                             number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION * 100,
                                                                                             number_ties / NUMBER_GAMES_VS_OLD_VERSION * 100))
    print("\tEvaluation completed\t time for evaluation: {} seconds".format((time.time() - start_time)))
    print(
        "__________________________________________________________________________________________________________________________")
    return agent


def alpha_0_pipeline(start_version, game, name_for_saving, seed):
    agent = Agent(type="alphaZero", player=1, seed=seed, version=start_version, scnds_per_move=SCNDS_PER_MOVE_TRAINING,
                  game=game, name_for_saving=name_for_saving)
    if start_version == 0:
        agent.save(0)

    while True:
        # play games to fill replay buffer
        self_play(agent, game)
        # learning from games to improve neural network
        agent.train()
        agent.version = agent.version + 1
        # evaluate version
        agent = evaluate(agent, game)

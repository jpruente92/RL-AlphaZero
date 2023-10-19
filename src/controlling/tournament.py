from collections import defaultdict
from logging import Logger

from agents.alpha_zero_agent import AlphaZeroAgent
from agents.mcts_agent import MCTSAgent
from agents.randomized_agent import RandomizedAgent
from game_logic.two_player_game import TwoPlayerGame


class Tournament:

    def __init__(
            self,
            logger: Logger,
            game: TwoPlayerGame,
            seconds_per_move=1
    ):
        self.LOGGER = logger
        self.GAME = game
        self.SECONDS_PER_MOVE = seconds_per_move
        self.AGENTS = []

    # region Public Method
    def start_tournament(self, no_games: int):
        self.GAME.set_gui_off()

        self.LOGGER.info("Starting Tournament")
        number_by_index_first_index_second_outcome = defaultdict(lambda: 0)
        total_number_wins_by_player_index = defaultdict(lambda: 0)
        nr_games_played = 0

        assert len(self.AGENTS) >= 2, "Too less agents are registered to tournament"
        for agent in self.AGENTS:
            self.LOGGER.info(f"Player '{agent.NAME}' registered for tournament!")

        for index_first_player in range(len(self.AGENTS)):
            for index_second_player in range(index_first_player + 1, len(self.AGENTS)):
                agent_1 = self.AGENTS[index_first_player]
                agent_2 = self.AGENTS[index_second_player]
                agent_1.set_player(1)
                agent_2.set_player(-1)
                for index_game in range(no_games):
                    self.GAME.reset()
                    self.GAME.start_game(agent_1, agent_2)
                    nr_games_played += 1
                    print(f"\r\t{nr_games_played} out of {self._compute_total_no_games(no_games)} games played", end="")
                    self._update_tournament_statistics(
                        index_first_player,
                        index_second_player,
                        number_by_index_first_index_second_outcome,
                        total_number_wins_by_player_index)
        self._print_tournament_results(
            number_by_index_first_index_second_outcome=number_by_index_first_index_second_outcome,
            total_number_wins_by_player_index=total_number_wins_by_player_index)

    def add_alpha_zero_agent(self, alpha_zero_version: int):
        self.AGENTS.append(
            AlphaZeroAgent(
                logger=self.LOGGER,
                player_number=-1,
                game=self.GAME,
                version=alpha_zero_version,
                seconds_per_move=self.SECONDS_PER_MOVE
            )
        )

    def add_mcts_agent(self):
        self.AGENTS.append(
            MCTSAgent(
                logger=self.LOGGER,
                player_number=-1,
                game=self.GAME,
                seconds_per_move=self.SECONDS_PER_MOVE
            )
        )

    def add_randomized_agent(self):
        self.AGENTS.append(
            RandomizedAgent(
                logger=self.LOGGER,
                player_number=-1,
                game=self.GAME
            )
        )

    # endregion Public Method

    # region Private Methods
    def _print_tournament_results(
            self,
            number_by_index_first_index_second_outcome: dict,
            total_number_wins_by_player_index: dict):
        print("\r")
        self.LOGGER.info("RESULTS VS EACH OTHER")
        for i in range(len(self.AGENTS)):
            for j in range(i + 1, len(self.AGENTS)):
                self.LOGGER.info(f"\t\t{self.AGENTS[i].NAME} vs {self.AGENTS[j].NAME}:"
                                 f"\t{number_by_index_first_index_second_outcome[i, j, 1]} wins "
                                 f"{number_by_index_first_index_second_outcome[i, j, 0]} ties "
                                 f"{number_by_index_first_index_second_outcome[i, j, -1]} losses")
        self.LOGGER.info("TOTAL WINS")
        for i in range(len(self.AGENTS)):
            self.LOGGER.info(f"\t\t{self.AGENTS[i].NAME} nr wins: {total_number_wins_by_player_index[i]}")

    def _update_tournament_statistics(
            self,
            index_first_player,
            index_second_player,
            number_by_index_first_index_second_outcome,
            total_number_wins):
        number_by_index_first_index_second_outcome[index_first_player, index_second_player, self.GAME.winner] = \
            number_by_index_first_index_second_outcome[index_first_player, index_second_player, self.GAME.winner] + 1
        if self.GAME.winner == 1:
            total_number_wins[index_first_player] = total_number_wins[index_first_player] + 1
        if self.GAME.winner == -1:
            total_number_wins[index_second_player] = total_number_wins[index_second_player] + 1

    def _compute_total_no_games(self, no_games: int) -> int:
        return int(no_games * len(self.AGENTS) * (len(self.AGENTS) - 1) / 2)

    # endregion Private Methods

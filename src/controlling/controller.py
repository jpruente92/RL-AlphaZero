import random
from typing import Optional

from agents.alpha_zero_agent import AlphaZeroAgent
from agents.mcts_agent import MCTSAgent
from agents.randomized_agent import RandomizedAgent
from agents.user_agent import UserAgent
from alpha_zero.alpha_zero_algorithm import AlphaZero
from controlling.log_manager import LogManager
from controlling.profiling import profile
from controlling.tournament import Tournament
from enums.opponent_type import OpponentType
from game_logic.connect_n import ConnectN


class Controller:

    def __init__(self, seed: Optional[int] = None):
        self.LOGGER = self._prepare_logger()

        self.GAME = None

        self._set_seed(seed)

    # region Public Methods

    def play_game(self, opponent_type: OpponentType, alpha_zero_version=0, seconds_per_move=1):
        agent_1 = self._set_agent_1(
            opponent_type=opponent_type,
            alpha_zero_version=alpha_zero_version,
            seconds_per_move=seconds_per_move
        )
        agent_2 = UserAgent(
            logger=self.LOGGER,
            player_number=1,
            game=self.GAME
        )
        self.GAME.start_game(agent_1, agent_2, 1)

    @profile
    def simulate_tournament(
            self,
            no_games=1,
            seconds_per_move=1
    ) -> None:
        tournament = Tournament(
            logger=self.LOGGER,
            game=self.GAME,
            seconds_per_move=seconds_per_move
        )

        tournament.add_alpha_zero_agent(alpha_zero_version=0)
        tournament.add_alpha_zero_agent(alpha_zero_version=1)
        tournament.add_mcts_agent()

        tournament.start_tournament(no_games=no_games)

    @profile
    def train_alpha_zero(
            self,
            name_for_saving: str,
            start_version: int = 0,
    ):
        alpha_0 = AlphaZero(
            game=self.GAME,
            name_for_saving=name_for_saving
        )
        alpha_0.start_training_pipeline(
            start_version=start_version)

    def set_game_to_connect_tic_tac_toe(
            self
    ):
        self.GAME = \
            ConnectN(
                n=3,
                no_rows=3,
                no_columns=3,
                gravity_on=False
            )
        self.LOGGER.info("Game set to Tic Tac Toe")

    def set_game_to_connect_4(
            self
    ):
        self.GAME = \
            ConnectN(
                n=4,
                no_rows=6,
                no_columns=7,
                gravity_on=True
            )
        self.LOGGER.info("Game set to Connect 4")

    # endregion Public Methods

    # region Private Methods

    def _prepare_logger(self):
        log_manager = LogManager()
        return log_manager.get_logger()

    def _set_seed(self, seed: int):
        if seed is None:
            seed = random.randint(0, 100000)
        self.LOGGER.info(f"Set seed to {seed}")
        random.seed(seed)

    def _set_agent_1(self,
                     alpha_zero_version: int,
                     opponent_type: OpponentType,
                     seconds_per_move: int):
        self.GAME.set_gui_on()
        if opponent_type == OpponentType.USER:
            agent_1 = UserAgent(
                logger=self.LOGGER,
                player_number=-1,
                game=self.GAME)
        elif opponent_type == OpponentType.RANDOM:
            agent_1 = RandomizedAgent(
                logger=self.LOGGER,
                player_number=-1,
                game=self.GAME)
        elif opponent_type == OpponentType.MONTE_CARLO_TREE_SEARCH:
            agent_1 = MCTSAgent(
                logger=self.LOGGER,
                player_number=-1,
                game=self.GAME,
                seconds_per_move=seconds_per_move)
        elif opponent_type == OpponentType.ALPHA_ZERO:
            agent_1 = AlphaZeroAgent(
                logger=self.LOGGER,
                player_number=-1,
                game=self.GAME,
                version=alpha_zero_version,
                seconds_per_move=seconds_per_move)
        else:
            raise NotImplementedError
        return agent_1

    # endregion Private Methods

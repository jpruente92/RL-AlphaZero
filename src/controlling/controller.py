import random
from typing import Optional, Literal

from agents.alpha_zero_agent import AlphaZeroAgent
from agents.mcts_agent import MCTSAgent
from agents.randomized_agent import RandomizedAgent
from agents.user_agent import UserAgent
from alpha_zero.alpha_zero_algorithm import AlphaZero
from alpha_zero.replay_buffer import ReplayBuffer
from constants.hyper_parameters import DEVICE, SECONDS_PER_MOVE_TRAINING
from controlling.log_manager import LogManager
from controlling.profiling import profile
from controlling.tournament import Tournament
from enums.opponent_type import OpponentType
from game_logic.connect_n import ConnectN
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.mcts_with_nn import MCTSWithNeuralNetwork
from neural_network.neural_network_torch.network_manager_torch import NetworkManagerTorch


class Controller:

    def __init__(self, seed: Optional[int] = None):
        self.LOGGER = self._prepare_logger()
        self.GAME: Optional[TwoPlayerGame] = None

        self.LOGGER.info("Controller created")
        self.LOGGER.info(f"Device: {DEVICE}")
        self._set_seed(seed)

    # region Public Methods

    def play_game(self, opponent_type: OpponentType, alpha_zero_version=0, seconds_per_move=1):
        self.GAME.set_gui_on()
        agent_1 = self._set_agent_1(
            opponent_type=opponent_type,
            alpha_zero_version=alpha_zero_version,
            seconds_per_move=seconds_per_move
        )
        agent_2 = UserAgent(
            logger=self.LOGGER,
            player_number=1,
            game=self.GAME,
        )
        self.GAME.start_game(agent_1, agent_2, 1)

    def simulate_tournament(
            self,
            no_games=1,
            seconds_per_move=1,
            with_gui=False
    ) -> None:
        tournament = Tournament(
            logger=self.LOGGER,
            seconds_per_move=seconds_per_move,
            with_gui=with_gui
        )

        tournament.add_alpha_zero_agent(
            alpha_zero_agent=self._create_alpha_zero_agent(
                alpha_zero_version=0,
                seconds_per_move=seconds_per_move)
        )
        # tournament.add_alpha_zero_agent(
        #     alpha_zero_agent=self._create_alpha_zero_agent(
        #         alpha_zero_version=1,
        #         seconds_per_move=seconds_per_move)
        # )
        # tournament.add_randomized_agent(with_gui=with_gui)
        tournament.add_mcts_agent(game=self.GAME)

        tournament.start_tournament(
            no_games=no_games,
            game=self.GAME
        )

    def train_alpha_zero(
            self,
            start_version: int = 0,
            parallel: bool = False
    ):
        assert self.GAME is not None
        alpha_0 = AlphaZero(
            logger=self.LOGGER
        )
        alpha_0.start_training_pipeline(
            alpha_zero_agent=self._create_alpha_zero_agent(
                alpha_zero_version=start_version,
                seconds_per_move=SECONDS_PER_MOVE_TRAINING
            ),
            game=self.GAME,
            parallel=parallel
        )

    def set_game_to_connect_tic_tac_toe(
            self
    ):
        self.GAME = \
            ConnectN(
                logger=self.LOGGER,
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
                logger=self.LOGGER,
                n=4,
                no_rows=6,
                no_columns=7,
                gravity_on=True
            )
        self.LOGGER.info("Game set to Connect 4")

    def set_game_to_custom_connect_n(
            self,
            n: int,
            no_rows: int,
            no_columns: int,
            gravity_on: bool
    ):
        self.GAME = \
            ConnectN(
                logger=self.LOGGER,
                n=n,
                no_rows=no_rows,
                no_columns=no_columns,
                gravity_on=gravity_on
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
        player_number: Literal[-1, 1] = -1
        if opponent_type == OpponentType.USER:
            agent_1 = UserAgent(
                logger=self.LOGGER,
                player_number=player_number,
                game=self.GAME
            )
        elif opponent_type == OpponentType.RANDOM:
            agent_1 = RandomizedAgent(
                logger=self.LOGGER,
                player_number=player_number,
                sleep_time_before_move_s=0.5
            )
        elif opponent_type == OpponentType.MONTE_CARLO_TREE_SEARCH:
            agent_1 = MCTSAgent(
                logger=self.LOGGER,
                game=self.GAME,
                player_number=player_number,
                seconds_per_move=seconds_per_move)
        elif opponent_type == OpponentType.ALPHA_ZERO:
            agent_1 = self._create_alpha_zero_agent(alpha_zero_version, seconds_per_move)
        else:
            raise NotImplementedError
        return agent_1

    def _create_alpha_zero_agent(self, alpha_zero_version, seconds_per_move):
        player_number: Literal[-1, 1] = -1
        network_manager = NetworkManagerTorch(
            logger=self.LOGGER,
            name_for_saving=self.GAME.NAME_FOR_SAVING,
            version=alpha_zero_version,
            no_actions=self.GAME.NO_ACTIONS,
            state_shape=self.GAME.STATE_SHAPE
        )
        mcts = MCTSWithNeuralNetwork(
            logger=self.LOGGER,
            seconds_per_move=seconds_per_move,
            game=self.GAME,
            player_number=player_number,
            network_manager=network_manager
        )
        replay_buffer = ReplayBuffer(
            logger=self.LOGGER,
            name_for_saving=self.GAME.NAME_FOR_SAVING,
            version=alpha_zero_version
        )
        agent = AlphaZeroAgent(
            logger=self.LOGGER,
            network_manager=network_manager,
            mcts=mcts,
            replay_buffer=replay_buffer,
            player_number=-1,
            version=alpha_zero_version,
            seconds_per_move=seconds_per_move
        )
        return agent

    # endregion Private Methods

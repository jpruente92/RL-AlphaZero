import copy
import random
from logging import Logger
from typing import Optional, Literal

import numpy as np

from agents.base_agent import BaseAgent
from game_logic.game_state import GameState


class TwoPlayerGame:

    def __init__(
            self,
            logger: Logger,
            no_actions: int,
            state_shape: tuple,
            name_for_saving: str
    ):
        self.LOGGER = logger
        self.NO_ACTIONS = no_actions
        self.STATE_SHAPE = state_shape
        self.NAME_FOR_SAVING = name_for_saving

        self.next_user_action = None
        self.gui_on = None
        self.GUI = None

    # region Public Methods

    def create_start_game_state(
            self,
            player_number_to_move: Optional[int] = None
    ) -> GameState:
        return GameState(
            feasible_actions=[i for i in range(self.NO_ACTIONS)],
            winner=None,
            board=np.zeros(self.STATE_SHAPE),
            player_number_to_move=player_number_to_move,
            previous_game_state=None
        )

    def start_game(self,
                   agent_1: BaseAgent,
                   agent_2: BaseAgent,
                   index_starting_agent: Optional[int] = None
                   ) -> GameState:
        game_state = self.create_start_game_state()
        if index_starting_agent is None:
            index_starting_agent = random.randint(0, 1)

        game_state.player_number_to_move = agent_1.player_number if index_starting_agent == 0 else agent_2.player_number
        game_state = self._game_loop(
            agent_1=agent_1,
            agent_2=agent_2,
            game_state=game_state
        )
        if self.gui_on:
            while True:
                self.GUI.refresh_picture(game_state)
        return game_state

    def action_is_feasible(
            self,
            action: int,
            game_state: GameState
    ) -> bool:
        field = self._action_to_field(action=action, board=game_state.board)
        return self._field_is_free(
            field=field,
            board=game_state.board
        )

    def step_if_feasible(
            self,
            action: int,
            game_state: GameState
    ) -> Optional[GameState]:
        assert game_state.player_number_to_move is not None
        assert game_state.player_number_to_move != 0

        field = self._action_to_field(action=action, board=game_state.board)

        new_game_state = None
        if self._field_is_free(
                field=field,
                board=game_state.board
        ):
            new_board = copy.deepcopy(game_state.board)
            new_board[field] = game_state.player_number_to_move

            winner = self._compute_winner(
                field=field,
                board=new_board,
                player_number_to_move=game_state.player_number_to_move
            )
            feasible_actions = self._compute_feasible_actions(
                action=action,
                field=field,
                board=new_board
            )
            player_number_to_move: Literal[-1, 1] = -1
            if game_state.player_number_to_move == -1:
                player_number_to_move = 1

            new_game_state = GameState(
                feasible_actions=feasible_actions,
                winner=winner,
                board=new_board,
                player_number_to_move=player_number_to_move,
                previous_game_state=game_state
            )

        return new_game_state

    def set_gui_off(self):
        self.gui_on = False

    # endregion Public Methods

    # region Private Methods

    def _game_loop(
            self,
            agent_1: BaseAgent,
            agent_2: BaseAgent,
            game_state: GameState
    ) -> GameState:
        assert agent_1.player_number == 1 or agent_2.player_number == 1
        assert agent_1.player_number == -1 or agent_2.player_number == -1
        while game_state.winner is None:
            if len(game_state.feasible_actions) == 0:
                game_state.winner = 0
                break
            current_agent = agent_1 if game_state.player_number_to_move == agent_1.player_number else agent_2
            action = current_agent.compute_action(game_state)
            game_state = self.step_if_feasible(
                action=action,
                game_state=game_state
            )

            if self.gui_on:
                self.GUI.next_user_action = None
                self.GUI.refresh_picture(game_state)

        return game_state

    def _field_is_free(
            self,
            field: tuple,
            board: np.array
    ) -> bool:
        if not self._is_field_valid(field):
            self.LOGGER.debug(f"field {field} not valid")
            return False
        if board[field] == 0:
            return True
        self.LOGGER.debug(f"field {field} not free: {board[field]}")
        return False

    def _is_field_valid(
            self,
            field: tuple
    ) -> bool:
        if len(field) != len(self.STATE_SHAPE):
            return False
        for coord, size in zip(field, self.STATE_SHAPE):
            if coord < 0 or coord >= size:
                return False
        return True

    # endregion Private Methods

    # region Abstract Methods

    def set_gui_on(self):
        raise NotImplementedError

    def _action_to_field(
            self,
            action: int,
            board: np.array
    ) -> tuple:
        raise NotImplementedError

    def _compute_feasible_actions(
            self,
            action: int,
            field: tuple,
            board: np.array
    ) -> list[int]:
        raise NotImplementedError

    def _compute_winner(
            self,
            field: tuple,
            board: np.array,
            player_number_to_move: Literal[-1, 1]
    ) -> Literal[None, -1, 0, 1]:
        raise NotImplementedError

    # endregion Abstract Methods

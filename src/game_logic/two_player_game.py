import copy
import random
from typing import Optional, Literal

import numpy as np

from agents.base_agent import BaseAgent
from gui.gui import Gui


class TwoPlayerGame:

    def __init__(
            self,
            no_actions: int,
            state_shape: tuple
    ):
        self.NO_ACTIONS = no_actions
        self.STATE_SHAPE = state_shape
        self.BOARD = np.zeros(state_shape)
        self.FEASIBLE_ACTIONS = [i for i in range(self.NO_ACTIONS)]

        self.winner = None
        self.user_action = None
        self.all_board_states = []

        self.gui_on = None

    # region Public Methods

    def start_game(self,
                   agent_1: BaseAgent,
                   agent_2: BaseAgent,
                   index_starting_agent: Optional[int] = None
                   ) -> None:
        agents = [agent_1, agent_2]
        current_agent = self._initialize_current_agent(
            agents=agents,
            index_starting_agent=index_starting_agent
        )
        self._game_loop(agent_1, agent_2, current_agent)
        if self.gui_on:
            while True:
                self.gui.refresh_picture(self.BOARD)

    def step_if_feasible(
            self,
            action: int,
            player_number: Literal[-1, 1]
    ) -> None:
        field = self._action_to_field(action)

        if self._field_is_free(field=field):
            self.BOARD[field] = player_number
            self.all_board_states.append(copy.deepcopy(self.BOARD))
            self._update_winner(field, player_number)

        self._update_feasible_actions(action, field)

        if self.gui_on:
            self.gui.refresh_picture(self.BOARD)

    def set_gui_on(self):
        self.gui_on = True
        self.gui = Gui(self)
        self.gui.refresh_picture(self.BOARD)

    def set_gui_off(self):
        self.gui_on = False

    def reset(self):
        self.BOARD = np.zeros(self.STATE_SHAPE)
        self.winner = None
        self.FEASIBLE_ACTIONS = [i for i in range(self.NO_ACTIONS)]
        self.user_action = None
        self.all_board_states = []

    def board_equal(self, other_board: np.array):
        return np.array_equal(self.BOARD, other_board)

    # endregion Public Methods

    # region Private Methods

    def _initialize_current_agent(
            self,
            # agents: list[BaseAgent],  # todo: comment in if env is ready
            agents,
            index_starting_agent: Optional[int]
    ) -> BaseAgent:
        if index_starting_agent is None:
            index_starting_agent = random.randint(0, 1)
        return agents[index_starting_agent]

    def _game_loop(
            self,
            agent_1: BaseAgent,
            agent_2: BaseAgent,
            current_agent: BaseAgent
    ) -> None:
        while self.winner is None:
            if len(self.FEASIBLE_ACTIONS) == 0:
                self.winner = 0
                break
            action = current_agent.compute_action()
            self.step_if_feasible(action, current_agent.player_number)
            current_agent = agent_1 if current_agent is agent_2 else agent_2

    def _field_is_free(
            self,
            field: tuple
    ) -> bool:
        if not self._is_field_valid(field):
            return False
        if self.BOARD[field] == 0:
            return True
        return False

    def _is_field_valid(
            self,
            field: tuple
    ) -> bool:
        if len(field) != len(self.BOARD.shape):
            return False
        for coord, size in zip(field, self.BOARD.shape):
            if coord < 0 or coord >= size:
                return False
        return True

    # endregion Private Methods

    # region Abstract Methods

    def clone(self):
        raise NotImplementedError

    def _action_to_field(
            self,
            action: int
    ) -> tuple:
        raise NotImplementedError

    def _update_feasible_actions(
            self,
            action: int,
            field: tuple) -> None:
        raise NotImplementedError

    def _update_winner(
            self,
            field: tuple,
            player_number: Literal[-1, 1]
    ) -> None:
        raise NotImplementedError

    # endregion Abstract Methods

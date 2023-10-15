from logging import Logger
from typing import Literal


class BaseAgent:
    def __init__(
            self,
            logger: Logger,
            name: str,
            player_number: Literal[-1, 1],
            game
    ):
        self.LOGGER = logger
        self.NAME = name
        self.PLAYER_NUMBER = player_number
        self.GAME = game

    def set_player(self, player_number: Literal[-1, 1]):
        self.PLAYER_NUMBER = player_number

    # region Abstract Methods
    def compute_action(
            self
    ) -> int:
        raise NotImplementedError

    # endregion Abstract Methods

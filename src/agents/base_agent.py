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
        self.GAME = game

        self.player_number = player_number

    def set_player(self, player_number: Literal[-1, 1]):
        self.player_number = player_number

    # region Abstract Methods
    def compute_action(
            self
    ) -> int:
        raise NotImplementedError

    # endregion Abstract Methods

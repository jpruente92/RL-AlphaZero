import time
from typing import Literal

from agents.base_agent import BaseAgent
from game_logic.two_player_game import TwoPlayerGame


class UserAgent(BaseAgent):

    def __init__(
            self,
            player_number: Literal[-1, 1],
            game: TwoPlayerGame

    ):
        super().__init__(name="User", player_number=player_number, game=game)

    def compute_action(
            self,
            training: bool = False
    ) -> int:
        self.GAME.user_action = None
        while self.GAME.user_action is None:
            self.GAME.gui.refresh_picture(self.GAME.BOARD)
            time.sleep(0.01)
        action = self.GAME.user_action
        self.GAME.user_action = None
        return action if action is not None else -1

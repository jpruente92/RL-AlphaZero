import time
from logging import Logger
from typing import Literal

from agents.base_agent import BaseAgent
from game_logic.game_state import GameState
from game_logic.two_player_game import TwoPlayerGame


class UserAgent(BaseAgent):

    def __init__(
            self,
            logger: Logger,
            player_number: Literal[-1, 1],
            game: TwoPlayerGame,

    ):
        self.GUI = game.GUI
        self.GAME = game
        super().__init__(logger=logger, name="User", player_number=player_number)

    def compute_action(
            self,
            game_state: GameState
    ) -> int:
        while True:
            time.sleep(0.01)
            self.GUI.refresh_picture(game_state)
            action = self.GUI.next_user_action
            if action is not None and self.GAME.action_is_feasible(action=action, game_state=game_state):
                self.GUI.next_user_action = None
                break
        return action


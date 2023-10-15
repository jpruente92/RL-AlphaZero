from game_logic.two_player_game import TwoPlayerGame


class Node:
    def __init__(
            self,
            game: TwoPlayerGame,
            player_number: int,
            current_player_number_before_state: int
    ):
        self.GAME = game
        self.CURRENT_PLAYER_NUMBER_BEFORE_STATE = current_player_number_before_state
        self.PLAYER_NUMBER = player_number

        self.action_before_state = -1
        self.terminated = False
        self.children = []
        self.father = None

        self.depth = 0
        self.visit_count = 0
        self.sum_of_observed_values = 0

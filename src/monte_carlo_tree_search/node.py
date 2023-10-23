from game_logic.game_state import GameState


class Node:
    def __init__(
            self,
            game_state: GameState,
            action_before_state: int,
            father,
            depth: int
    ):
        self.GAME_STATE = game_state
        self.ACTION_BEFORE_STATE = action_before_state
        self.FATHER = father
        self.DEPTH = depth

        self.terminated = False
        self.children = []
        self.visit_count = 0
        self.sum_of_observed_values = 0

import time

from mcts import MCTS


class Player():
    def __init__(self, type, player, scnds_per_move=1, game=None):
        self.type = type
        self.player = player
        if type == "mcts":
            self.mcts = MCTS(scnds_per_move, game, player)

    def compute_action(self, game):
        game.user_action = None
        if self.type == "random":
            return game.random.choice(game.feasible_actions)
        elif self.type == "user":
            while game.user_action is None:
                game.gui.refresh_picture(game.board)
                time.sleep(0.01)
            action = game.user_action
            game.user_action = None
            return action
        elif self.type == "mcts":
            return self.mcts.step()

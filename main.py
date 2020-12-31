import random
import time

from connect_n import Game
from player import Player

seed = random.randint(0, 100000)
print("seed =", seed)
# seed = 1706
random.seed(seed)
# game = Game(random, 3, 3, 3, False, True) # connect 3
game = Game(random, 4, 6, 7, True, True)  # connect 4

player_1 = Player("mcts", scnds_per_move=3, game=game, player=1)
player_2 = Player("user", player=-1)

game.start_game(player_1,player_2)

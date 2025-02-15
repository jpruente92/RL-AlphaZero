import torch

#  region Training Parameters
WIN_PERCENTAGE = 55
NUMBER_GAMES_PER_SELF_PLAY = 3000
BATCH_SIZE = 64
NUMBER_OF_BATCHES_TRAINING = 50_000
PERCENTAGE_DATA_FOR_VALIDATION = 10
NUMBER_GAMES_VS_OLD_VERSION = 400
# in some games we have a lot of ties, and therefore we count them as wins but weighted with this weight,
# in standard alphazero this parameter would be 0
WEIGHT_FOR_TIES_IN_EVALUATION = 0.5
WEIGHT_POLICY_LOSS = 4.0
# small value to avoid log of 0
EPSILON = 0.0001

#  endregion Training Parameters

# region Optimizer Parameters
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
# endregion Optimizer Parameters

# region Network Architecture
NO_RESIDUAL_LAYERS = 19
NO_BOARD_STATES_SAVED = 1
# endregion Network Architecture

# region Replay Buffer Parameters
BUFFER_SIZE = 500_000
# endregion Replay Buffer Parameters

# region MCTS Parameters
# parameter for controlling exploration vs exploitation in MCTS - higher value leads to more exploration
C = 4
SECONDS_PER_MOVE_TRAINING = 0.2
MAX_NR_STEPS_TRAINING = 1_600
# endregion MCTS Parameters

# region Misc
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NR_PROCESSES_ON_CPU = 4
# endregion Misc

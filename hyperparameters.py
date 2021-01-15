import torch

# training parameters
WIN_PERCENTAGE = 55
NUMBER_GAMES_PER_SELF_PLAY = 2500  # 2500-25000
BATCH_SIZE = 128
NUMBER_OF_BATCHES_TRAINING = 1600          # 16000
NUMBER_OF_BATCHES_VALIDATION = 8
NUMBER_GAMES_VS_OLD_VERSION = 400        # 400
WEIGHT_FOR_TIES_IN_EVALUATION = 0.5 # in some games we have a lot of ties and therefore we count them as wins but weighted with this weight, in standard alphazero this parameter would be 0

# optimizer parameters
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

# network architecture
NR_RESIDUAL_LAYERS = 20

# replay buffer parameters
BUFFER_SIZE = 50000

# MDCS parameters
C = 4 # parameter for controlling exploration vs exploitation in MCTS - higher value leads to more exploration
SCNDS_PER_MOVE_TRAINING = 0.1
MAX_NR_STEPS_TRAINING = 1600

# Misc
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NR_PROCESSES_ON_CPU = 4



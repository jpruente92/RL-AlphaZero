import logging
import os

LOG_FOLDER_PATH = os.path.join("..", "logs")
LOG_FILE_NAME = "alpha_0"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(levelname)7s - [%(asctime)s] - %(message)s"
TIME_STAMP_FORMAT = "%d_%m_%Y_%H_%M_%S"

PROFILE_FILE_PATH = os.path.join("..", "profiles", "profile")
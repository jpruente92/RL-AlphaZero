import os.path
import sys
from datetime import datetime

from constants.logging_constants import *
from controlling.formatter import LoggerFormatter


class LogManager:

    def __init__(
            self,
            log_folder_path: str = LOG_FOLDER_PATH,
            log_file_name: str = LOG_FILE_NAME,
            log_level: int = LOG_LEVEL,
            log_format: str = LOG_FORMAT

    ):
        time_stamp = datetime.now().strftime(TIME_STAMP_FORMAT)
        self.log_file_path = os.path.join(log_folder_path, f"{log_file_name}_{time_stamp}.log")
        self.log_level = log_level
        self.log_format = log_format

        self._create_log_folder(log_folder_path)

    def get_logger(self) -> logging.Logger:
        logger = logging.getLogger("logger")
        logger.handlers.clear()
        logger.setLevel(self.log_level)
        handler_console = logging.StreamHandler(sys.stdout)
        formatter_console = LoggerFormatter(self.log_format, log_to_file=False)
        self._configure_and_add_handler(
            formatter=formatter_console,
            handler=handler_console,
            logger=logger
        )
        handler_file = logging.FileHandler(self.log_file_path)
        formatter_file = LoggerFormatter(self.log_format, log_to_file=True)
        self._configure_and_add_handler(
            formatter=formatter_file,
            handler=handler_file,
            logger=logger
        )
        return logger

    def _create_log_folder(self,
                           log_folder_path: str
                           ) -> None:
        os.makedirs(log_folder_path, exist_ok=True)

    def _configure_and_add_handler(
            self,
            formatter: logging.Formatter,
            handler: logging.Handler,
            logger: logging.Logger
    ) -> None:
        handler.setLevel(self.log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

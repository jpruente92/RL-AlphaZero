import logging


class LoggerFormatter(logging.Formatter):

    def __init__(
            self,
            log_format: str,
            log_to_file: bool):
        super().__init__(log_format)

        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        light_blue = "\x1b[94;20m"
        green = "\x1b[32;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        if log_to_file:
            self.FORMATS = {
                logging.DEBUG: log_format,
                logging.INFO: log_format,
                logging.WARNING: log_format,
                logging.ERROR: log_format,
                logging.CRITICAL: log_format
            }
        else:
            self.FORMATS = {
                logging.DEBUG: light_blue + log_format + reset,
                logging.INFO: green + log_format + reset,
                logging.WARNING: yellow + log_format + reset,
                logging.ERROR: red + log_format + reset,
                logging.CRITICAL: bold_red + log_format + reset
            }

    def format(self, record):
        record.levelname = f"{record.levelname:7}"

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

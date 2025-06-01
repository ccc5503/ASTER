import os
import logging
from datetime import datetime
import sys


class Logger:
    def __init__(self, name, log_dir=None):
        """

        Args:
            name
            log_dir
        """
        self.name = name

        if log_dir is None:
            log_dir = os.path.join("logs", name)

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.log_file = log_file

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

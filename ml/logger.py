import logging
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = "experiments/logs") -> logging.Logger:
    """
    Centralized logger for the project.
    Creates one log file per run inside experiments/logs/.
    """

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Log file name with timestamp
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # log everything, filter in handlers

    # Prevent duplicate handlers if logger is reused
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (writes all logs to file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler (prints only INFO+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

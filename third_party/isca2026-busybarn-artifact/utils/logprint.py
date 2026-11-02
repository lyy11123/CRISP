
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


import logging


logger = logging.getLogger('MyCustomLogger')
logger.setLevel(1)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def setup_custom_levels(levels):
    """
    Define and register custom logging levels.
    ``levels`` is a list of numeric log levels.
    """
    levels += [logging.DEBUG]
    for level in levels:
        logging.addLevelName(level, f"LEVEL{level}")

def log_message(level, message):
    """
    Emit a log message at the specified level.
    """
    if level in [lvl for lvl in logging._levelToName if isinstance(lvl, int)]:
        logger.log(level, message)
    else:
        raise ValueError("Specified logging level is not defined. Please define it first.")

def setup_logging_levels(levels):
    """
    Enable only the log levels listed in ``levels``.
    """
    levels += [logging.DEBUG]
    class LevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in levels

    stream_handler.filters.clear()
    stream_handler.addFilter(LevelFilter(levels))


if __name__ == "__main__":
    setup_custom_levels([5, 15, 25, 35, 45, 55])
    setup_logging_levels([5, 10])

    log_message(5, "This is a custom log message with level 5")
    logger.debug("This is a debug message")
    log_message(15, "This is a custom log message with level 15")
    log_message(25, "This is a custom log message with level 25")
    log_message(35, "This is a custom log message with level 35")
    log_message(45, "This is a custom log message with level 45")
    log_message(55, "This is a custom log message with level 55")
    log_message(10, "This is a custom log message with level 10")

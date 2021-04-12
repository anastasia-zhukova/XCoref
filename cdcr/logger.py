import logging


LEVEL = "INFO"
"""str: The log level to use."""

FORMAT = "%(asctime)s@%(thread)d %(levelname)s %(module)s(%(lineno)d):%(funcName)s|: %(message)s"
"""str: The log format to use."""

# DATEFMT = "%H:%M:%S.%f"

NAME = "newsalyze-backend"

LOGGER = logging.getLogger(NAME)


def setup():
    """Setup general logging (log level and format)."""
    logging.basicConfig(
        format=FORMAT,
        level=LEVEL)


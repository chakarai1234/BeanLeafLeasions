from argparse import Namespace
import logging
from logging import Logger
from typing import Union


def get_logger(logger_name: str, log_file: Union[str, None] = None) -> Logger:
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %I:%M:%S%p")

    logger.setLevel(logging.DEBUG)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Logger Initialized with file")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    if log_file is not None:
        logger.info("Logger Initialized with file")
    else:
        logger.info("Logger Initialized")

    return logger


def log(values: Union[Namespace, dict[str, Union[str, int, float]], list[dict[str, Union[str, int, float]]]],
        logger: Logger, justify: int):
    if isinstance(values, Namespace):
        for k, v in vars(values).items():
            logger.info("{}:\t{}".format(k.ljust(justify), v))
    elif isinstance(values, dict):
        logger.info("{}:\t{}".format(list(values.keys())[0].ljust(justify), list(values.values())[0]))
    elif isinstance(values, list):
        for dicts in values:
            if isinstance(dicts, dict):
                logger.info("{}:\t{}".format(list(dicts.keys())[0].ljust(justify), list(dicts.values())[0]))

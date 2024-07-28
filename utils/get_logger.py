from argparse import Namespace
import logging
from logging import Logger
from typing import Union


def get_logger(log_file: str) -> Logger:
    logger = logging.getLogger("Bean Leaf Lesions")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %I:%M:%S%p")

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def print_config(config: Namespace, logger: Logger, justify: int):
    for k, v in vars(config).items():
        logger.info("{}:\t{}".format(k.ljust(justify), v))


def log(dicts: dict[str, Union[str, int, float]], logger: Logger, justify: int):
    logger.info("{}:\t{}".format(list(dicts.keys())[0].ljust(justify), list(dicts.values())[0]))

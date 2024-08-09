from argparse import Namespace
import logging
from logging import Logger
from typing import Union, List, Dict, Mapping
from torchinfo import ModelStatistics


def get_logger(logger_name: str, log_file: Union[str, None] = None) -> Logger:
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s", "%Y-%m-%d %I:%M:%S%p")

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
        logger.info("Logger Initialized without file")

    return logger


def log(values: Union[Namespace, ModelStatistics, Exception, Mapping[str, Union[str, int, float, List[str], List[int], List[float]]], List[Mapping[str, Union[str, int, float, List[str], List[int], List[float]]]]],
        logger: Logger, justify: int):

    if isinstance(values, Namespace):
        for key, value in vars(values).items():
            logger.info("{}:\t{}".format(key.ljust(justify), value))
    elif isinstance(values, dict):
        for key, value in values.items():
            logger.info("{}:\t{}".format(key.ljust(justify), value))
    elif isinstance(values, list):
        for dicts in values:
            if isinstance(dicts, dict):
                for key, value in dicts.items():
                    logger.info("{}:\t{}".format(key.ljust(justify), value))
    elif isinstance(values, ModelStatistics):
        logger.info(f"\n{values}\n")
    elif isinstance(values, Exception):
        logger.info("{}:\t{}".format("Exception".ljust(justify), repr(values)))

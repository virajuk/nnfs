import logging

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="logs/logger.log",
                    level=logging.DEBUG,
                    format=LOG_FORMAT)


def get_logger():

    logger = logging.getLogger()
    return logger

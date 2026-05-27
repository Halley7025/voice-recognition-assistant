import logging, sys

def setup_logger(name='voice_assistant', level=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('[%(levelname).1s][%(name)s] %(message)s'))
        logger.addHandler(h)
        logger.setLevel(level or logging.INFO)
    return logger

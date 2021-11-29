import logging

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level = logging.INFO, filename = "test.log", format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Start logging.")

def log_level(level):
    if level == "debug":
        return logging.DEBUG
    if level == "info":
        return logging.INFO
    if level == "warning":
        return logging.WARNING
    if level == "error":
        return logging.ERROR
    if level == "critical":
        return logging.CRITICAL
    if level == "fatal":
        return logging.FATAL
    return logging.INFO
import logging

def init_logger(log_file = None, log_file_level = logging.NOTSET):

    #logging format --> need only message
    log_format = logging.Formatter("%(message)s")

    #set level to logging Info
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #streaming logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    #added longging handler
    logger.handlers = [console_handler]

    #if log file add it
    if log_file and log_file != '':
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
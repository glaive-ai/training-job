import logging

def setup_logging():
    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler and set its formatter
    file_handler = logging.FileHandler('output/log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a stream handler (stdout) and set its formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Change the level to suit your needs
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
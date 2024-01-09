import logging

def setup_logging(log_file_path, rank):
    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler and set its formatter
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if rank == 0:
        # Only write to stdout when you are the main process
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # Change the level to suit your needs
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    
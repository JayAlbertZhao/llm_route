import os
import logging
import datetime
import socket

def setup_logger(name, experiment_id=None, log_dir="logs"):
    """
    Sets up a logger that writes to a file in a specific experiment directory.
    """
    if experiment_id is None:
        # Default to timestamp if no ID provided
        experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment-specific log directory
    exp_dir = os.path.join(log_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Identify component (client/router/backend)
    hostname = socket.gethostname()
    filename = f"{name.lower()}_{hostname}.log"
    log_path = os.path.join(exp_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Writing to {log_path}")
    return logger, experiment_id


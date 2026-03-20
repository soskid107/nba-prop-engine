import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "nba_engine", log_level: str = "INFO", log_file: str = "logs/pipeline.log") -> logging.Logger:
    """
    Setup standardized logger with console and file output.
    
    Args:
        name: Logger name
        log_level: Console log level
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, assume it's configured
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG) # capture all at root level
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler (User friendly)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (Detailed)
    # Ensure directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Singleton default logger
default_logger = setup_logger()

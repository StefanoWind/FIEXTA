from typing import Optional
import logging
import os


class SingletonLogger:
    _instance = None

    def __new__(cls, verbose: bool = True, logger: Optional[logging.Logger] = None,filename=None):
        
        cls._instance = super(SingletonLogger, cls).__new__(cls)
        cls._instance.verbose = verbose

        if logger is None:
            
            # create default logger
            logger = logging.getLogger("default")
            if logger.hasHandlers():
                logger.handlers.clear()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            logger.setLevel(logging.INFO)
            
            if filename is not None:
                #create log file
                os.makedirs(os.path.dirname(filename),exist_ok=True)
                file_handler = logging.FileHandler(filename, mode="a")  # Append mode
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
        cls._instance.logger = logger
        return cls._instance

    def log(self, message: str, level: str = "info"):
        """Log a message at the specified level."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        if self.verbose and level.lower() != "debug":
            print(message)

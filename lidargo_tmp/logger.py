from typing import Optional
import logging
from typing import Optional


class SingletonLogger:
    _instance = None

    def __new__(cls, verbose: bool = True, logger: Optional[logging.Logger] = None):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance.verbose = verbose

            if logger is None:
                # Create default logger if none provided
                logger = logging.getLogger("default")
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)

            cls._instance.logger = logger
        return cls._instance

    def log(self, message: str, level: str = "info"):
        """Log a message at the specified level."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        if self.verbose and level.lower() != "debug":
            print(message)

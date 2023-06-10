import logging
import logging.handlers

def configure_logging(name: str, level=logging.DEBUG):
    file_handler = logging.handlers.RotatingFileHandler(
                f"{name}.log", maxBytes=10*1024*1024, backupCount=10)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    logging.basicConfig(
        handlers=[console_handler, file_handler],
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=level
    )

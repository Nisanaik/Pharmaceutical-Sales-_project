import logging
# Configure the logging settings
logging.basicConfig(filename='logfile.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d')

#  Create a logger object
logger = logging.getLogger()
logger = logging.getLogger("my_logger")

logger.setLevel(logging.DEBUG)

# Creating and configuring handlers
file_handler = logging.FileHandler("app.log")
console_handler = logging.StreamHandler()

# Setting the log level for each handler
file_handler.setLevel(logging.ERROR)
console_handler.setLevel(logging.DEBUG)

# Creating a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Setting the formatter for each handler
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Attaching the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Logging messages
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")


def my_function():
    logger.info('Starting my_function...')

    try:
        # Your code logic here
        result = 10 / 2
        logger.info('Result: %s', result)
    except Exception as e:
        logger.error('An error occurred: %s', str(e))
    finally:
        logger.info('Finished my_function.')

if __name__ == "__main__":
    # Your main code here
    logger.info('Program started.')
    my_function()
    logger.info('Program finished.')

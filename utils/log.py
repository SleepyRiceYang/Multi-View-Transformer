import mindspore.log as logger

class Logger:
    def __init__(self, filename, level=logger.INFO,
                 format='%(asctime)s %(levelname)s %(message)s',
                 datefmt='%a, %d %b %Y %H:%M:%S', filemode='w'):
        self.level = level
        self.format = format
        self.datefmt = datefmt
        self.filename = filename
        self.filemode = filemode
        logger.basicConfig(level=self.level,
                            format=self.format,
                            datefmt=self.datefmt,
                            filename=self.filename,
                            filemode=self.filemode)
        self._set_streaming_handler()

    def _set_streaming_handler(self, level=logger.INFO, formatter='%(asctime)s %(levelname)-8s %(message)s'):
        console = logger.StreamHandler()
        console.setLevel(level)
        curr_formatter = logger.Formatter(formatter)
        console.set_formatter(curr_formatter)
        logger.getLogger(self.filename).add_handler(console)

    def get_logger(self):
        return logger.getLogger(self.filename)


def get_logger(log_path):
    logger.basicConfig(level=logger.INFO,
                        filename=log_path,
                        format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    console = logger.StreamHandler()
    console.level = logger.NOTSET
    log = logger.getLogger()
    log.addHandler(console)
    return log

import unittest
import logging
from src.utils import setup_logging


class TestUtils(unittest.TestCase):

    def test_setup_logging(self):
        try:
            setup_logging()
            logging.info("Logging is set up correctly")
        except Exception as e:
            self.fail(f'setup_logging() raised Exception unexpectedly: {e}')


if __name__ == '__main__':
    unittest.main()

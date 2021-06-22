from unittest import TestCase

from time_vars import TimeUtils
import datetime as dt

class TestTimeUtils(TestCase):

    time = TimeUtils()

    def test_get_start(self,my_time):
        assert self.time.get_start() == dt.datetime(2016, 1, 1)

    def test_get_end(self,my_time):
        assert  self.time.get_end() == dt.datetime.now()



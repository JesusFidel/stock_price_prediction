import datetime as dt


class TimeUtils:
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

if __name__ == "__main__":
    time = TimeUtils()

    time.get_start()
    time.get_end()
import datetime as dt


def random_date(np_random, year=2019):
    random_day = dt.timedelta(days=np_random.randint(1, 365))
    return dt.datetime(year, 1, 1) + random_day

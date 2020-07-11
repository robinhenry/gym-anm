import datetime as dt


def random_date(np_random, year):
    """
    Generate a random date within the year `year`.

    Parameters
    ----------
    np_random : numpy.random.RandomState
        The random seed.
    year : int
        The year from which to generate a random date.

    Returns
    -------
    datetime.datetime
        A datetime of 00:00 on a random day within the year `year`.
    """

    random_day = dt.timedelta(days=np_random.randint(1, 365))
    return dt.datetime(year, 1, 1) + random_day


def dt_to_minutes(delta):
    """
    Convert a `datetime.timedelta` object into seconds.

    Parameters
    ----------
    delta : `datetime.timedelta`
        The interval of time to transform into seconds.

    Returns
    -------
    int
        The interval of time in seconds.
    """

    return delta.seconds // 60 % 60
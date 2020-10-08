def epsilon_by_epoch(epoch, start=1.0, end=0.1, duration=10):
    """

    :param epoch: (int)
    :param start: (float)
    :param end: (float)
    :param duration: (int) epochs between start to end
    :return: epsilon (float)
    """

    if(epoch <= duration):
        return start - (start - end) * epoch / duration
    else:
        return end

import logging
def is_chi(item):
    """
    :param item: array of tile 34 indices
    :return: boolean
    """
    if len(item) != 3:
        return False

    return item[0] == item[1] - 1 == item[2] - 2


def is_pon(item):
    """
    :param item: array of tile 34 indices
    :return: boolean
    """
    if len(item) != 3:
        return False

    return item[0] == item[1] == item[2]


def is_pair(item):
    """
    :param item: array of tile 34 indices
    :return: boolean
    """
    return len(item) == 2


def check_ready_to_win(tiles_18):
    """
    Check the hand whether it's ready to win
    :param tiles_18:
    :return:
    """
    total = 0
    for num in tiles_18:
        total += num
    if total % 3 != 2:
        logging.getLogger("mahjong").error("error,手牌数目不对:{}".format(total))
        return False
    return True


def check_ready_to_touch(tiles_18):
    """
    Check the hand whether it's ready to touch
    :param tiles_18:
    :return:
    """
    total = 0
    for num in tiles_18:
        total += num
    if total % 3 != 1:
        print("error,手牌数目不对:", total)
        return False
    return True

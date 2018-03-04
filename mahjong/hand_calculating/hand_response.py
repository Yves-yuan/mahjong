# -*- coding: utf-8 -*-


class HandResponseZigong(object):
    cost = None
    hand = None

    def __init__(self, hand=None, cost=None):
        self.cost = cost
        self.hand = hand

    def __str__(self):
        return 'hand :{} , cost:{} '.format(self.hand, self.cost)


class HandResponse(object):
    cost = None
    han = None
    fu = None
    fu_details = None
    yaku = None
    error = None

    def __init__(self, cost=None, han=None, fu=None, yaku=None, error=None, fu_details=None):
        """
        :param cost: dict
        :param han: int
        :param fu: int
        :param yaku: list
        :param error: str
        :param fu_details: dict
        """
        self.cost = cost
        self.han = han
        self.fu = fu
        self.error = error

        if fu_details:
            self.fu_details = sorted(fu_details, key=lambda x: x['fu'], reverse=True)

        if yaku:
            self.yaku = sorted(yaku, key=lambda x: x.yaku_id)

    def __str__(self):
        if self.error:
            return self.error
        else:
            return '{} han, {} fu'.format(self.han, self.fu)

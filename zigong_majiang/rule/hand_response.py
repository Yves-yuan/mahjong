class HandResponseZigong(object):
    cost = None
    hand = None

    def __init__(self, hand=None, cost=None, error=None):
        self.cost = cost
        self.hand = hand
        self.error = error

    def __str__(self):
        return 'hand :{} , cost:{} '.format(self.hand, self.cost)




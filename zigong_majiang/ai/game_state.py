import copy
from collections import namedtuple

from zigong_majiang.rule.hand_calculator import HandCalculator


class GameState:
    def __init__(self, hands, discards, melds_3, melds_4, turn=0):
        """
        表示牌局当前状态的类，记录了各个玩家的手牌，丢弃牌，碰牌，杠牌，到谁的轮
        :param hands: 长度为18的数组，每个值代表数组索引的牌的数量
        :param discards: 丢弃的牌，列表，值为丢弃的牌的值
        :param melds_3:
        :param melds_4:
        :param turn: 轮次，0-2
        """
        self.turn = turn
        self.hands = hands
        self.discards = discards
        self.melds_3 = melds_3
        self.melds_4 = melds_4

    def get_discards(self, tile):
        count = 0
        for discard in self.discards:
            count += discard.count(tile)
        return count

    def check(self):
        print(self.hands)
        for index in range(0, len(self.hands)):
            hand = self.hands[index]
            total = 0
            for num in hand:
                total += num
            if index == 0:
                if total % 3 != 2:
                    print("error,手牌数目不对:", total)
                    return False
            else:
                if total % 3 != 1:
                    print("error,手牌数目不对:", total)
                    return False
        return True

    def hands_index(self, index):
        return copy.deepcopy(self.hands[index])

    def touch_tile(self, tile):
        total = 0
        for num in self.hands[self.turn]:
            total += num
        if total > 13:
            print("wrong")
        self.hands[self.turn][tile] += 1
        total = 0
        for num in self.hands[self.turn]:
            total += num
        if total > 14:
            print("wrong")

    def score(self, index):
        return HandCalculator.estimate_hand_value_zigong(self.hands[index], self.hands[index][0])

    def discard(self, tile):
        self.discards[self.turn].append(tile)
        self.hands[self.turn][tile] -= 1
        self.turn += 1
        self.turn %= 3

    def clone(self):
        game_state = GameState(hands=copy.deepcopy(self.hands), discards=copy.deepcopy(self.discards),
                               melds_3=copy.deepcopy(self.melds_3), melds_4=copy.deepcopy(self.melds_4),
                               turn=self.turn)
        return game_state

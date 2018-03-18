import copy
from collections import namedtuple

from zigong_majiang.rule.hand_calculator import HandCalculator


class GameState:
    def __init__(self, hands, discards, melds_3, melds_4, turn=0):
        self.turn = turn
        self.hands = hands
        self.discards = discards
        self.melds_3 = melds_3
        self.melds_4 = melds_4

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
        self.hands[self.turn][tile] += 1

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

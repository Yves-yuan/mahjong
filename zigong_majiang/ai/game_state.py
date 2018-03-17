import copy
from collections import namedtuple

from zigong_majiang.rule.hand_calculator import HandCalculator


class GameState(namedtuple('GameState', 'hands discards melds_3 melds_4')):
    def check(self):
        print(self.hands)
        for index in range(0, len(self.hands)):
            hand = self.hands[index]
            total = 0
            for num in hand:
                total += num
            if index == 0 and total % 3 != 2:
                print("error,手牌数目不对:", total)
                return False
            else:
                if total % 3 != 1:
                    return False
        return True

    def hands_index(self, index):
        return copy.deepcopy(self.hands[index])

    def touch_tile(self, index, tile):
        self.hands[index].append(tile)

    def score(self, index):
        return HandCalculator.estimate_hand_value_zigong(self.hands[index], self.hands[index][0])

    def discard(self, tile, index):
        self.discards[index].append(tile)
        self.hands[index].remove(tile)

    def clone(self):
        game_state = GameState(hands=copy.deepcopy(self.hands), discards=copy.deepcopy(self.discards),
                               melds_3=copy.deepcopy(self.melds_3), melds_4=copy.deepcopy(self.melds_4))
        return game_state

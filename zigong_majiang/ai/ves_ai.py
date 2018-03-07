from zigong_majiang.rule.Agari import Agari
from zigong_majiang.rule.tile import Tile


class VesAI(object):

    def __init__(self, n):
        self.n = n
        self.agari = Agari()

    def calc_effective_cards(self, tiles_18):
        for card in range(0, 18):
            if tiles_18[card] < 4:
                tiles_18[card] += 1
                if self.agari.is_agari_zigong(tiles_18):
                    print("听牌:", Tile(card))
                tiles_18[card] -= 1

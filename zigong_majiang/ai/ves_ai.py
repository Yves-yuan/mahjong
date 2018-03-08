import copy

from zigong_majiang.rule.agari import Agari
from zigong_majiang.rule.judge_tile_chain import JudgeTileChain
from zigong_majiang.rule.tile import Tile
from zigong_majiang.rule.touch_play_pair import TouchPlayPair


class VesAI(object):

    def __init__(self, n):
        self.n = n
        self.agari = Agari()

    def calc_draw_hands(self, tiles_18):
        draw_hands = []
        for card in range(0, 18):
            if tiles_18[card] < 4:
                tiles_18[card] += 1
                if self.agari.is_agari_zigong(tiles_18):
                    draw_hands.append(card)
                tiles_18[card] -= 1
        return draw_hands

    def calc_effective_cards(self, tiles_18, n, tpps):
        # 听牌
        draw_hands = self.calc_draw_hands(tiles_18)
        for draw_hand in draw_hands:
            print("听牌:", Tile(draw_hand).__str__())

        # 一层胡牌链
        first_layer_jtc = []
        for card in range(0, 18):
            if tiles_18[card] < 4:
                tiles_18[card] += 1
                for card1 in range(0, 18):
                    if tiles_18[card1] > 0 and card1 != card:
                        tiles_18[card1] -= 1
                        touch_play_pair = TouchPlayPair(card, card1)
                        if n == 1:
                            draw_hands = self.calc_draw_hands(tiles_18)
                            if draw_hands:
                                local = copy.deepcopy(tpps)
                                local.append(touch_play_pair)
                                first_layer_jtc.append(JudgeTileChain(local, draw_hands))
                        tiles_18[card1] += 1
                tiles_18[card] -= 1
        for node in first_layer_jtc:
            print(node)

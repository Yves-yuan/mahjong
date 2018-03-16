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

    def calc_effective_cards(self, tiles_18, n):
        if len(tiles_18) != 13:
            print("hands must be a 13 length array")
        judge_tile_chains = self.calc_effective_cards_internal(tiles_18, n)
        for chain in judge_tile_chains:
            print(chain)
        # for chain in judge_tile_chains:
        #     print(chain)

    def calc_effective_cards_internal(self, tiles_18, n, judge_tile_chains=[], touch_play_pairs=[]):
        draw_hands_cur = self.calc_draw_hands(tiles_18)
        if draw_hands_cur:
            judge_tile_chains.append(JudgeTileChain(copy.deepcopy(touch_play_pairs), draw_hands_cur))
            print(JudgeTileChain(copy.deepcopy(touch_play_pairs), draw_hands_cur))
        for card in range(0, 18):
            if tiles_18[card] < 4:
                tiles_18[card] += 1
                for card1 in range(0, 18):
                    if tiles_18[card1] > 0 and card1 != card:
                        tiles_18[card1] -= 1
                        touch_play_pair = TouchPlayPair(card, card1)
                        local = copy.deepcopy(touch_play_pairs)
                        local.append(touch_play_pair)
                        if n == 1:
                            draw_hands = self.calc_draw_hands(tiles_18)
                            if draw_hands:
                                judge_tile_chains.append(JudgeTileChain(local, draw_hands))
                                print(JudgeTileChain(local, draw_hands))
                        else:
                            judge_tile_chains.append(
                                self.calc_effective_cards_internal(tiles_18, n - 1, judge_tile_chains, local))
                        tiles_18[card1] += 1
                tiles_18[card] -= 1

        return judge_tile_chains

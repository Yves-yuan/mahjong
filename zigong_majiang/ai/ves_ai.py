import copy

from zigong_majiang.rule.hand.agari import Agari
from zigong_majiang.rule.chain.judge_tile_chain import JudgeTileChain
from zigong_majiang.rule.hand.hand_calculator import HandCalculator
from zigong_majiang.rule.tile.tile_convert import TilesConverter
from zigong_majiang.rule.chain.touch_play_pair import TouchPlayPair
import logging
from zigong_majiang.log.logger import logger


class VesAI(object):

    def __init__(self, n):
        self.n = n
        self.agari = Agari()
        self.log = logging.getLogger("mahjong")

    def calc_effective_cards(self, tiles_18, n):
        count = TilesConverter.tiles18_count(tiles_18)
        if count % 3 != 1:
            self.log.error("hands must be ready hand example:13 length or 7 length")
            return None
        judge_tile_chains = self.calc_effective_cards_internal(tiles_18, n, [], [])
        logger().debug(TilesConverter.tiles_18_to_str(tiles_18))
        for chain in judge_tile_chains:
            logger().debug(chain)
        return judge_tile_chains

    def calc_effective_cards_internal(self, tiles_18, n, judge_tile_chains, touch_play_pairs):
        if len(judge_tile_chains) > 5:
            logger().debug("Length of judge chains before calculate:{}".format(len(judge_tile_chains)))
        draw_hands_cur = HandCalculator.calc_draw_hands(tiles_18)
        if draw_hands_cur:
            judge_tile_chains.append(
                JudgeTileChain(copy.deepcopy(touch_play_pairs), draw_hands_cur, copy.deepcopy(tiles_18)))
        if n == 0:
            return judge_tile_chains

            # self.log.info(JudgeTileChain(copy.deepcopy(touch_play_pairs), draw_hands_cur,copy.deepcopy(tiles_18)))
        for card in range(0, 18):
            if tiles_18[card] < 4:
                tiles_18[card] += 1
                for card1 in range(0, 18):
                    if tiles_18[card1] > 0 and card1 != card:
                        tiles_18[card1] -= 1
                        touch_play_pair = TouchPlayPair(card, card1)
                        if n == 1:
                            draw_hands = HandCalculator.calc_draw_hands(tiles_18)
                            if draw_hands:
                                local = copy.deepcopy(touch_play_pairs)
                                local.append(touch_play_pair)
                                judge_tile_chains.append(JudgeTileChain(local, draw_hands, copy.deepcopy(tiles_18)))
                                # self.log.info(JudgeTileChain(local, draw_hands,copy.deepcopy(tiles_18)))
                        else:
                            local = copy.deepcopy(touch_play_pairs)
                            local.append(touch_play_pair)
                            judge_tile_chains.append(
                                self.calc_effective_cards_internal(tiles_18, n - 1, judge_tile_chains, local))
                        tiles_18[card1] += 1
                tiles_18[card] -= 1

        return judge_tile_chains

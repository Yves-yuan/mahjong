import copy

from mahjong.rule.algo.winchecker import WinChecker
from mahjong.rule.model.judge_tile_chain import JudgeTileChain
from mahjong.rule.algo.hand_calculator import HandCalculator
from mahjong.rule.util.tile_convert import TilesConv
from mahjong.rule.model.touch_play_pair import TouchPlayPair
import logging
from mahjong.log.logger import logger


class JudgeChainMaker(object):

    def __init__(self, n):
        self.n = n
        self.wc = WinChecker()
        self.log = logging.getLogger("mahjong")

    def calc_effective_cards(self, tiles_18, n):
        count = TilesConv.tiles18_count(tiles_18)
        if count % 3 != 1:
            self.log.error("hands must be ready hand example:13 length or 7 length")
            return None
        judge_tile_chains = self.calc_effective_cards_internal(tiles_18, n, [], [])
        logger().debug(TilesConv.tiles_18_to_str(tiles_18))
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

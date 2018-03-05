from zigong_majiang.rule.Agari import Agari
from zigong_majiang.rule.Response import HandResponseZigong
from zigong_majiang.rule.HandDriver import HandDivider
from zigong_majiang.rule.scores import ScoresCalculator
from zigong_majiang.rule.tile import TilesConverter


class HandCalculator(object):
    def estimate_hand_value_zigong(self, tiles, win_tile):
        """
        it's a specific version of China SiChuan Zigong majhong.
        :param tiles:
        :param win_tile:
        :return:
        """

        scores_calculator = ScoresCalculator()
        tiles_18 = TilesConverter.to_18_array(tiles)
        agari = Agari()
        if not agari.is_agari_zigong(tiles_18):
            return HandResponseZigong(error='Hand is not winning')

        divider = HandDivider()
        hand_options = divider.divide_hand_zigong(tiles_18)
        calculated_hands = []
        for hand in hand_options:
            # win_groups = self._find_win_groups(win_tile, hand, [])
            cost = scores_calculator.calculate_scores_zigong(hand, win_tile, tiles_18)
            calculated_hand = HandResponseZigong(hand, cost)
            calculated_hands.append(calculated_hand)
        return calculated_hands

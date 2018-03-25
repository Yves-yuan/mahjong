from zigong_majiang.rule.hand.agari import Agari
from zigong_majiang.rule.hand.hand_response import HandResponseZigong
from zigong_majiang.rule.hand.hand_driver import HandDivider
from zigong_majiang.rule.hand.scores import ScoresCalculator


class HandCalculator(object):
    @staticmethod
    def calc_draw_hands(tiles_18):
        """
        计算听牌列表
        :param tiles_18:
        :return:
        """
        draw_hands = []
        for card in range(0, 18):
            if tiles_18[card] < 4:
                tiles_18[card] += 1
                if Agari.is_win_zigong(tiles_18):
                    draw_hands.append(card)
                tiles_18[card] -= 1
        return draw_hands

    @staticmethod
    def estimate_hand_value_zigong(tiles_18, win_tile):
        """
        it's a specific version of China SiChuan Zigong majhong.
        :param tiles_18:
        :param tiles_72:
        :param win_tile:
        :return:
        """

        scores_calculator = ScoresCalculator()
        if not Agari.is_win_zigong(tiles_18):
            print("error,no win")
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

    @staticmethod
    def estimate_max_score(tiles_18, win_tile):
        """
        计算最大胡牌得分
        :param tiles_18:必须是胡牌的手牌，例如14张
        :param win_tile:
        :return:
        """
        ch = HandCalculator.estimate_hand_value_zigong(tiles_18,win_tile)
        max_cost = 0
        for result in ch:
            if result.cost > max_cost:
                max_cost = result.cost
        return max_cost

from mahjong.rule.algo.winchecker import WinChecker
from mahjong.rule.model.hand_cost import HandCost
from mahjong.rule.algo.hand_divider import HandDivider
from mahjong.rule.util.utils import is_pair, is_pon, is_chi


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
                if WinChecker.is_win(tiles_18):
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

        if not WinChecker.is_win(tiles_18):
            print("error,no win")
            return HandCost(error='Hand is not winning')

        divider = HandDivider()
        hand_options = divider.divide_hand_zigong(tiles_18)
        calculated_hands = []
        for hand in hand_options:
            # win_groups = self._find_win_groups(win_tile, hand, [])
            cost = HandCalculator.calculate_scores_zigong(hand, win_tile, tiles_18)
            calculated_hand = HandCost(hand, cost)
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

    @staticmethod
    def calculate_scores_zigong(hand, wintile, tiles_18):
        # find gang
        gang = 0
        for tile in tiles_18:
            if tile == 4:
                gang += 1

        # find qingyise
        qingyise = True
        last = -1
        for group in hand:
            if last < 0:
                last = group[0]
            if last != group[0] // 9:
                qingyise = False
                break
            last = group[0] / 9

        # calculate longqidui
        duiziNum = 0
        for group in hand:
            if len(group) == 2:
                duiziNum += 1
            if len(group) == 4:
                duiziNum += 2
        if duiziNum == 7:
            base = 4
            final = base * 2 ** gang
            if qingyise:
                final *= 4
            return final

        pair_sets = [x for x in hand if is_pair(x)]
        pon_sets = [x for x in hand if is_pon(x)]
        chi_sets = [x for x in hand if is_chi(x)]

        if len(chi_sets) == 0:
            core = 4
            for group in pair_sets:
                if len(group) == 4:
                    core *= 2
                if qingyise:
                    core *= 4
            return core

        # find kaertiao
        ka_er_tiao = False
        if wintile == 10:
            for group in pon_sets:
                if wintile in group:
                    ka_er_tiao = True
                    break

        base = 1
        if ka_er_tiao:
            base *= 2
        if qingyise:
            base *= 4
        base *= 2 ** gang
        return base

    @staticmethod
    def calc_score_for_results(results):
        max_cost = -1
        for result in results:
            if result.cost > max_cost:
                max_cost = result.cost
        return max_cost

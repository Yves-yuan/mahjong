import numpy as np
import copy

from zigong_majiang.ai.game_state import GameState
from zigong_majiang.ai.ves_ai import VesAI
from zigong_majiang.rule.hand_calculator import HandCalculator
from zigong_majiang.ai.constant import constant

S0 = 10
S1 = 6
S2 = 6
K0 = 2
K1 = 1
K2 = 1

ves_ai = VesAI(1)


class Attack:
    def __init__(self, s0=S0, s1=S1, s2=S2, k0=K0, k1=K1, k2=K2):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2

    @staticmethod
    def tile_remain_num(tile, game_state):
        s = game_state
        return 4 - s.hands[s.turn][tile] - s.get_discards(tile)

    @staticmethod
    def get_first_level_ts(t):
        if t >= 0:
            p = t % 9
            if p * (p - 8):
                return [t + 1, t - 1]
            else:
                return [t + 1 - 1 * p // 4]
        return None

    @staticmethod
    def get_second_level_ts(t):
        if t >= 0:
            p = t % 9
            if p * (p - 8) * (p - 1) * (p - 7):
                return [t + 2, t - 2]
            else:
                if p < 2:
                    return [t + 2]
                else:
                    return [t - 2]
        return None

    # def think_peng(self):
    #     l_r_t = self.player.cnt[game_state.receive_tiles[-1][1]]
    #     sum_rp, sum_rpp = 0, 0
    #     for t in range(0, 34):
    #         sum_rp += self.rp_t(t)
    #     self.player.cnt[l_r_t] -= 2
    #     for t in range(0, 34):
    #         sum_rpp += self.rp_t(t)
    #     sum_rpp += 3 * self.s0
    #     self.player.cnt[l_r_t] += 2
    #
    #     return (sum_rpp >= sum_rp)

    @staticmethod
    def think(game_state: GameState):
        """
        计算打牌的概率，返回数组，数组中
        :param game_state:
        :return:出牌概率归一化后的数组
        """
        expects = []
        for t in range(0, 18):
            expect = Attack.think_expectation(t, game_state)
            if expect > 0:
                expects.append(expect)
        expect_prob = np.array(Attack.weights2_probability(expects))

        tile_weights = []
        for t in range(0, 18):
            p_t = Attack.rp_t(t, game_state)
            if p_t > 0:
                tile_weights.append(p_t)
        probability = np.array(Attack.weights2_probability(tile_weights))
        final = expect_prob * 0.7 + probability * 0.3
        return final

    @staticmethod
    def think_expectation(tile, game_state: GameState):
        hand = copy.deepcopy(game_state.hands[game_state.turn])
        if hand[tile] < 0:
            print("error:tile num < 0")
            return -1
        if hand[tile] == 0:
            return -1
        hand[tile] -= 1
        chains = ves_ai.calc_effective_cards(hand, 1)
        expect = 0
        for chain in chains:
            # 计算每种胡牌链的期望
            expect_per_chain = 0.0
            pairs = chain.touchPlayPairs
            prob_pair = 1.0
            for pair in pairs:
                need = pair.touch_tile()
                remain = game_state.get_remain(need, game_state.turn)
                prob_pair *= remain
                prob_pair /= constant.PLAYER_NUM
            for dh in chain.drawHands:
                # 计算一个链种每种胡牌的期望
                chain.hand[dh] += 1
                cost = HandCalculator.estimate_max_score(chain.hand, dh)
                chain.hand[dh] -= 1
                remain_dh = game_state.get_remain(dh,game_state.turn)
                expect_per_chain += prob_pair * remain_dh / constant.PLAYER_NUM * cost
            expect += expect_per_chain
        return expect

    @staticmethod
    def weights2_probability(tile_weights_in):
        """
        通过邻近牌权重数组计算出牌概率，邻近权重越大表示越应该保留,出牌概率就越小，邻近权重越小出牌的概率越大
        :param tile_weights_in:输入邻近权重
        :return: 出牌概率，归一化
        """
        tile_weights = copy.deepcopy(tile_weights_in)
        sum1 = sum(tile_weights)
        for index in range(0, len(tile_weights)):
            if tile_weights[index] > 0:
                tile_weights[index] = sum1 / tile_weights[index]
        sum2 = sum(tile_weights)
        for index in range(0, len(tile_weights)):
            if tile_weights[index] > 0:
                tile_weights[index] = tile_weights[index] / sum2
        return tile_weights

    @staticmethod
    def rp_t(t, game_state):
        """
        计算邻近的牌的个数和权重，权重越高邻近的牌越多
        :param t:牌值
        :param game_state:牌局
        :return: 返回邻近牌的权重,邻近牌越多权重越高,如果手牌中没有该牌返回0
        """
        s = game_state
        p_t = 0
        sn_0 = s.hands[s.turn][t]
        if sn_0:
            sn_1, sn_2, kn_0, kn_1, kn_2 = 0, 0, 0, 0, 0
            kn_0 = Attack.tile_remain_num(t, game_state)
            # 计算相邻的牌的个数
            for t1 in Attack.get_first_level_ts(t):
                sn_1 += s.hands[s.turn][t1]
                kn_1 += Attack.tile_remain_num(t1, game_state)
            # 计算相隔一张牌的个数
            for t2 in Attack.get_second_level_ts(t):
                sn_2 += s.hands[s.turn][t2]
                kn_2 += Attack.tile_remain_num(t2, game_state)
            p_t = sn_0 * S0 + (sn_1 * S1) + (sn_2 * S2) + kn_0 * K0 + (kn_1 * K1) + (
                    kn_2 * K2)
        return p_t

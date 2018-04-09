import numpy as np
import copy

from mahjong.ai.monte.game_state import GameState
from mahjong.rule.algo.judge_chain_maker import JudgeChainMaker
from mahjong.rule.algo.winchecker import WinChecker
from mahjong.rule.algo.hand_calculator import HandCalculator
from mahjong.ai.constant import constant
import logging
from mahjong.log.logger import logger
from mahjong.rule.util.tile_convert import TilesConv
from mahjong.rule.util.utils import check_ready_to_win, check_ready_to_touch

S0 = 10
S1 = 6
S2 = 6
K0 = 2
K1 = 1
K2 = 1

ves_ai = JudgeChainMaker(1)

WEIGHT_FITST = 1
WEIGHT_SECOND = 0.1
WEIGHT_ZIMO = 1
WEIGHT_PASS_ZIMO = 0.6
WEIGHT_FANGPAO = 1
WEIGHT_PASS_FANGPAO = 0.6

class Attack:
    def __init__(self, s0=S0, s1=S1, s2=S2, k0=K0, k1=K1, k2=K2):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2

    @staticmethod
    def tile_remain_num(tile, game_state, index):
        s = game_state
        return 4 - s.hands[index][tile] - s.get_discards(tile)

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

    @staticmethod
    def think_gang(node, game_state: GameState, index, tile):
        num = game_state.get_tile_num_of_hand(tile, index)
        if num < 3:
            return False
        tree = node

    @staticmethod
    def think_peng(node, game_state: GameState, index, tile):
        num = game_state.get_tile_num_of_hand(tile, index)
        if num < 2:
            return False
        tree = node
        if tree.children is None:
            tree.expand_peng(index, tile)

        # 计算邻近度概率
        rps = []
        for node in tree.children:
            sum = 0
            for t in range(0, 18):
                sum += Attack.rp_t1(t, index, node.game_state)
            rps.append(sum)
        peng_rps_probability = np.array(Attack.weights2_probability(rps))

        # 计算期望值概率
        expects = []
        for node in tree.children:
            expect = Attack.think_expectation(node.game_state, index)
            expects.append(expect)

        expect_prob = np.array(Attack.expects2_probability(expects))

        # 设置节点综合打牌概率
        final = expect_prob * 0.6 + peng_rps_probability * 0.4
        for i in range(0, len(tree.children)):
            tree.children[i].set_peng_probability(final[i])

        return True

    @staticmethod
    def think_zimo(node):
        expects = []
        for n in node.children:
            if n.reason == "zimo":
                result = n.game_result
                score = result * WEIGHT_ZIMO
                expects.append(score)
            else:
                expect = Attack.think_expectation(n.game_state, node.get_turn()) * WEIGHT_PASS_ZIMO
                if expect < 0:
                    expect = 0
                expects.append(expect)

        expect_prob = np.array(Attack.expects2_probability(expects))

        rps = []
        for n in node.children:
            sum = 0
            for t in range(0, 18):
                sum += Attack.rp_t1(t, node.get_turn(), n.game_state)
            rps.append(sum)
        peng_rps_probability = np.array(Attack.weights2_probability(rps))
        final = expect_prob * 0.7 + peng_rps_probability * 0.3
        for i in range(0, len(node.children)):
            node.children[i].set_zimo_probability(final[i])

    @staticmethod
    def think_fangpao(node, game_state: GameState, index, discard_tile):
        hand = game_state.hands[index]
        hand[discard_tile] += 1
        result = WinChecker.is_win(hand)
        hand[discard_tile] -= 1
        if not result:
            return result
        if node.children is None:
            node.expend_fangpao(index, discard_tile)

        expects = []
        for n in node.children:
            if n.reason == "fangpao":
                result = n.game_result
                score = result * WEIGHT_FANGPAO
                expects.append(score)
            else:
                expect = Attack.think_expectation(n.game_state, index) * WEIGHT_PASS_FANGPAO
                if expect < 0:
                    expect = 0
                expects.append(expect)

        expect_prob = np.array(Attack.expects2_probability(expects))

        rps = []
        for n in node.children:
            sum = 0
            for t in range(0, 18):
                sum += Attack.rp_t1(t, index, n.game_state)
            rps.append(sum)
        peng_rps_probability = np.array(Attack.weights2_probability(rps))
        final = expect_prob * 0.7 + peng_rps_probability * 0.3
        for i in range(0, len(node.children)):
            node.children[i].set_fangpao_probability(final[i])

        return result

    @staticmethod
    def think(game_state: GameState):
        """
        计算打牌的概率，返回数组，数组中
        :param game_state:
        :return:出牌概率归一化后的数组
        """
        log = logging.getLogger("mahjong")
        expects = []
        for t in range(0, 18):
            expect = Attack.think_expectation_forplay(t, game_state)
            if expect >= 0:
                expects.append(expect)
        expect_prob = np.array(Attack.expects2_probability(expects))
        log.debug("The expectations{}".format(expect_prob))
        tile_weights = []
        for t in range(0, 18):
            p_t = Attack.rp_t1(t, game_state.turn, game_state)
            if p_t > 0:
                tile_weights.append(p_t)
        probability = np.array(Attack.weights2_probability(tile_weights))
        log.debug("The concentrations{}".format(probability))
        final = expect_prob * 0.6 + probability * 0.4
        log.debug("Final{}".format(final))
        return final

    @staticmethod
    def think_expectation(game_state: GameState, index):
        """
        计算玩家index的手牌胡牌期望,要求手牌数目是%3 == 1
        :param game_state:
        :param index:
        :return:
        """
        check = check_ready_to_touch(game_state.hands[index])
        if check == False:
            logger().error("hand length is not ready to touch")
            return -1
        hand = copy.deepcopy(game_state.hands[index])
        chains = ves_ai.calc_effective_cards(hand, 0)
        expect = 1
        logger().debug("Length of chains:{}".format(len(chains)))
        for chain in chains:
            # 计算每种胡牌链的期望
            expect_per_chain = 0.0
            pairs = chain.touchPlayPairs
            prob_pair = 1.0
            weight = WEIGHT_SECOND
            if len(pairs) == 0:
                weight = WEIGHT_FITST

            for pair in pairs:
                need = pair.touch_tile()
                remain = game_state.get_remain(need, index)
                prob_pair *= remain
                prob_pair /= constant.PLAYER_NUM
                if prob_pair > 1:
                    prob_pair = 1

            for dh in chain.drawHands:
                # 计算一个链种每种胡牌的期望
                chain.hand[dh] += 1
                cost = HandCalculator.estimate_max_score(chain.hand, dh)
                chain.hand[dh] -= 1
                remain_dh = game_state.get_remain(dh, index)
                prob_dh = remain_dh / constant.PLAYER_NUM
                if prob_dh > 1:
                    prob_dh = 1
                expect_per_chain += prob_pair * prob_dh * cost * weight
            expect += (expect_per_chain / len(chain.drawHands))
        return expect

    @staticmethod
    def think_expectation_forplay(tile, game_state: GameState):
        """
        在需要打牌的时候计算打牌期望，手牌是14张
        :param tile:
        :param game_state:
        :return:
        """
        check = check_ready_to_win(game_state.hands[game_state.turn])
        if check == False:
            logger().error("Error:hand number is not right.")
            return -1
        if game_state.hands[game_state.turn][tile] < 0:
            print("error:tile num < 0")
            return -1
        if game_state.hands[game_state.turn][tile] == 0:
            return -1
        game_state.hands[game_state.turn][tile] -= 1
        expect = Attack.think_expectation(game_state, game_state.turn)
        game_state.hands[game_state.turn][tile] += 1
        return expect

    @staticmethod
    def expects2_probability(expects):
        """
        Calculate probability of card to be played according to expectation array.
        The array represents the expectation of score for hands playing the index's card.
        :param expects:
        :return:
        """
        tile_expects = copy.deepcopy(expects)
        sum1 = sum(tile_expects)
        for index in range(0, len(tile_expects)):
            if tile_expects[index] > 0:
                tile_expects[index] = tile_expects[index] / sum1
        return tile_expects

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
    def rp_t1(t, index, game_state):
        """
        计算邻近的牌的个数和权重，权重越高邻近的牌越多
        :param index: 计算玩家{index}的t牌邻近权重
        :param t:牌值
        :param game_state:牌局
        :return: 返回邻近牌的权重,邻近牌越多权重越高,如果手牌中没有该牌返回0
        """
        s = game_state
        p_t = 0
        sn_0 = s.hands[index][t]
        if sn_0:
            sn_1, sn_2, kn_0, kn_1, kn_2 = 0, 0, 0, 0, 0
            kn_0 = Attack.tile_remain_num(t, game_state, index)
            # 计算相邻的牌的个数
            for t1 in Attack.get_first_level_ts(t):
                sn_1 += s.hands[index][t1]
                kn_1 += Attack.tile_remain_num(t1, game_state, index)
            # 计算相隔一张牌的个数
            for t2 in Attack.get_second_level_ts(t):
                sn_2 += s.hands[index][t2]
                kn_2 += Attack.tile_remain_num(t2, game_state, index)
            p_t = sn_0 * S0 + (sn_1 * S1) + (sn_2 * S2) + kn_0 * K0 + (kn_1 * K1) + (
                    kn_2 * K2)
        p_t /= TilesConv.tiles18_count(s.hands[index])
        return p_t

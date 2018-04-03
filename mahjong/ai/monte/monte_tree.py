import logging
import random

import numpy as np
import math
from enum import Enum
from mahjong.ai.constant.constant import const
from mahjong.ai.monte.game_state import GameState
from mahjong.ai.attack.attack import Attack
from mahjong.rule.algo.winchecker import WinChecker
from mahjong.rule.algo.hand_calculator import HandCalculator
from mahjong.rule.model.tile import Tile
from mahjong.simulator.game_server import GameServer
from mahjong.log.logger import logger

N = 7
W = N + 2
empty = "\n".join([(N + 1) * ' '] + N * [' ' + N * '.'] + [(N + 2) * ' '])
colstr = 'ABCDEFGHJKLMNOPQRST'

N_SIMS = 5
PUCT_C = 0.1
PROPORTIONAL_STAGE = 3
TEMPERATURE = 2
P_ALLOW_RESIGN = 0.8
RAVE_EQUIV = 100
EXPAND_VISITS = 1
PRIOR_EVEN = 4  # should be even number; 0.5 prior
PRIOR_NET = 40
REPORT_PERIOD = 200
RESIGN_THRES = 0.025

PLAYER_NUM = 3
UNIFORM_PROBABILITY = 1 / 18
UNIFORM_DISTRIBUTION = [UNIFORM_PROBABILITY] * 18


class TreeNode:
    """ Monte-Carlo tree node;
       v is #visits, w is #wins for to-play (expected reward is w/v)
       pv, pw are prior values (node value = w/v + pw/pv)
       av, aw are amaf values ("all moves as first", used for the RAVE tree policy)
       children is None for leaf nodes """

    def __init__(self, net, game_state: GameState):
        self.net = net
        self.game_state = game_state
        self.game_result = None
        self.attack_drop_p = 0
        self.peng_probability = 0
        self.fangpao_probability = 0
        self.v = 0
        self.w = 0
        self.pv = 0
        self.pw = 0
        self.av = 0
        self.aw = 0
        self.children = None
        self.touch_tile = -1
        self.discard_tile = -1
        self.peng_tile = -1
        self.peng_index = -1
        self.fangpao_from_index = -1
        self.fangpao_to_index = -1
        self.reason = "dogfall"
        self.pass_p = -1
        self.pass_fangpao_index = -1

    def get_hands_str_index(self, index):
        return self.game_state.get_hands_str_index(index)

    def get_last_discard(self):
        return self.game_state.last_discard

    def get_discard_tile(self):
        return self.discard_tile

    def get_turn(self):
        return self.game_state.turn

    def zimo(self, result):
        self.game_result = result
        self.reason = "zimo"

    def pass_peng(self, index):
        self.pass_p = index

    def peng(self, index, peng_tile, discard_tile):
        # 碰牌的手牌扣除两张碰的牌
        self.game_state.hands[index][peng_tile] -= 2
        self.game_state.hands[index][discard_tile] -= 1
        self.discard_tile = discard_tile
        self.game_state.last_discard = discard_tile
        self.game_state.discards[index].append(discard_tile)
        # 捡起上次玩家丢弃的牌，加入碰的牌
        self.game_state.discards[self.game_state.get_next_turn(-1)].pop()
        # 记录节点碰牌
        self.peng_tile = peng_tile
        self.peng_index = index
        # 记录牌局状态玩家{index}碰牌
        self.game_state.melds_3[index].append(peng_tile)
        # 轮次变换，轮到碰牌的人的下家摸牌
        self.game_state.turn = (index + 1) % 3

    def fangpao(self, lose_index, win_index, result):
        self.fangpao_from_index = lose_index
        self.fangpao_to_index = win_index
        self.game_result = result
        self.reason = "fangpao"

    def pass_fangpao(self, index):
        self.pass_fangpao_index = index

    def set_attack_drop_p(self, p):
        self.attack_drop_p = p

    def set_fangpao_probability(self, p):
        self.fangpao_probability = p

    def set_peng_probability(self, p):
        self.peng_probability = p

    def clone(self):
        node = TreeNode(None, self.game_state.clone())
        return node

    def touch(self, tile):
        self.game_state.touch_tile(tile)
        self.touch_tile = tile

    def discard(self, tile):
        self.game_state.discard(tile)
        self.discard_tile = tile

    def expend_fangpao(self, index, fangpao_tile):
        """扩展放炮子节点"""
        self.children = []
        hand_fangpao = self.game_state.hands_index(index)
        hand_fangpao[fangpao_tile] += 1
        result_fangpao = HandCalculator.estimate_hand_value_zigong(hand_fangpao, fangpao_tile)
        result_node = self.clone()
        value = UNIFORM_DISTRIBUTION[0]
        result_node.pv = PRIOR_NET
        result_node.pw = PRIOR_NET * value
        result_node.fangpao(self.game_state.get_next_turn(-1), index,
                            result_fangpao)
        self.children.append(result_node)
        pass_node = self.clone()
        value = UNIFORM_DISTRIBUTION[0]
        pass_node.pv = PRIOR_NET
        pass_node.pw = PRIOR_NET * value
        pass_node.pass_fangpao(index)
        self.children.append(pass_node)

    def expand_peng(self, index, peng_tile):
        """扩展碰牌子节点"""
        hand = self.game_state.hands[index]
        # 碰牌子节点
        self.children = []
        for t in range(0, len(hand)):
            if t != peng_tile:
                if hand[t] <= 0:
                    continue
            else:
                if hand[t] <= 2:
                    continue
            node = self.clone()
            node.peng(index, peng_tile, t)
            value = UNIFORM_DISTRIBUTION[t]
            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * value
            node.game_state.turn = (index + 1) % 3
            self.children.append(node)
        node = self.clone()
        node.pass_peng(index)
        node.pv = PRIOR_NET
        node.pw = PRIOR_NET * value
        self.children.append(node)

    def expand(self):
        """ add and initialize children to a leaf node """
        """ 扩展子节点，包括自己打出的牌和对手摸牌打牌 """
        # distribution = self.net.predict_distribution(self.game_state)
        distribution = UNIFORM_DISTRIBUTION
        self.children = []
        turn = self.game_state.turn
        for tile in range(0, len(self.game_state.hands[turn])):
            if self.game_state.hands[turn][tile] <= 0:
                continue
            node = self.clone()
            node.discard(tile)
            value = distribution[tile]
            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * value
            self.children.append(node)

    def puct_urgency(self, n0):
        # XXX: This is substituted by global_puct_urgency()
        expectation = float(self.w + PRIOR_EVEN / 2) / (self.v + PRIOR_EVEN)
        try:
            prior = float(self.pw) / self.pv
        except:
            prior = 0.1  # XXX
        return expectation + PUCT_C * prior * math.sqrt(n0) / (1 + self.v)

    def rave_urgency(self):
        v = self.v + self.pv
        expectation = float(self.w + self.pw) / v
        if self.av == 0:
            return expectation
        rave_expectation = float(self.aw) / self.av
        beta = self.av / (self.av + v + float(v) * self.av / RAVE_EQUIV)
        return beta * rave_expectation + (1 - beta) * expectation

    def winrate(self):
        return float(self.w) / self.v if self.v > 0 else float('nan')

    def prior(self):
        return float(self.pw) / self.pv if self.pv > 0 else float('nan')

    def best_move(self, proportional=False):
        """ best move is the most simulated one """
        if self.children is None:
            return None
        if proportional:
            probs = [(float(node.v) / self.v) ** TEMPERATURE for node in self.children]
            probs_tot = sum(probs)
            probs = [p / probs_tot for p in probs]
            # print([(str_coord(n.pos.last), p, p * probs_tot) for n, p in zip(self.children, probs)])
            i = np.random.choice(len(self.children), p=probs)
            return self.children[i]
        else:
            return max(self.children, key=lambda node: node.v)

    def distribution(self):
        distribution = np.zeros(N * N + 1)
        for child in self.children:
            p = float(child.v) / self.v
            c = child.pos.last
            if c is not None:
                x, y = c % W - 1, c // W - 1
                distribution[y * N + x] = p
            else:
                distribution[-1] = p
        return distribution


def puct_urgency_input(nodes):
    w = np.array([float(n.w) for n in nodes])
    v = np.array([float(n.v) for n in nodes])
    pw = np.array([float(n.pw) if n.pv > 0 else 1. for n in nodes])
    pv = np.array([float(n.pv) if n.pv > 0 else 10. for n in nodes])
    return w, v, pw, pv


def global_puct_urgency(n0, w, v, pw, pv):
    # Like Node.puct_urgency(), but for all children, more quickly.
    # Expects numpy arrays (except n0 which is scalar).
    expectation = (w + PRIOR_EVEN / 2) / (v + PRIOR_EVEN)
    prior = pw / pv
    return expectation + PUCT_C * prior * math.sqrt(n0) / (1 + v)


log = logging.getLogger("mahjong")


class MahjongState(Enum):
    Discarding = 1
    Penging = 2
    Ganging = 3
    Zimo = 4
    Fangpaoing = 5
    Touching = 6
    Over = 7


switcher = {
    MahjongState.Discarding: lambda: discarding,
    MahjongState.Penging: lambda: penging,
    MahjongState.Ganging: lambda: ganging,
    MahjongState.Zimo: lambda: zimoing,
    MahjongState.Fangpaoing: lambda: fangpaoing,
    MahjongState.Touching: lambda: touching,
    MahjongState.Over: lambda: touching,
}


def discarding(root, tree, nodes, server):
    tree.v += 1
    if tree.children is None:
        tree.expand()

    children = list(nodes[-1].children)
    attack_probabilitys = Attack.think(nodes[-1].game_state)
    for index in range(0, len(children)):
        children[index].set_attack_drop_p(attack_probabilitys[index])
    # Pick the most urgent child
    random.shuffle(children)  # randomize the max in case of equal urgency

    urgencies = global_puct_urgency(nodes[-1].v, *puct_urgency_input(children))
    attack_probabilitys_np = np.array([n.attack_drop_p for n in children])
    if root:
        dirichlet = np.random.dirichlet((0.03, 1), len(children))
        urgencies = urgencies * 0.5 + dirichlet[:, 0] * 0.25 + attack_probabilitys_np * 0.25
    urgencies = urgencies * 0.7 + attack_probabilitys_np * 0.3
    log.debug("urgencies:{}".format(urgencies))
    node = max(zip(children, urgencies), key=lambda t: t[1])[0]
    nodes.append(node)
    log.info(" player:{} discard tile:{}".format(tree.get_turn(), Tile(node.discard_tile)))
    node.v += 1
    return MahjongState.Fangpaoing


def fangpaoing(root, tree, nodes, server):
    fangpao_nodes, state = fangpao_check(tree, root)
    nodes.extend(fangpao_nodes)
    return state


def zimoing(root, tree, nodes, server):
    is_win = WinChecker.is_win(tree.game_state.hands_index(tree.get_turn()))
    if is_win:
        game_result = HandCalculator.estimate_hand_value_zigong(tree.game_state.hands_index(tree.get_turn()),
                                                                tree.touch_tile)
        nodes[-1].zimo(game_result)
        logger().info("Player{} zimo{} hands{}".format(tree.get_turn(), tree.touch_tile,
                                                       tree.get_hands_str_index(tree.get_turn())))
        return MahjongState.Over
    return MahjongState.Discarding


def ganging(root, tree, nodes, server):
    gang_node, state = gang_check(tree)
    nodes.extend(gang_node)
    return state


def penging(root, tree, nodes, server):
    peng_node, state = peng_check(tree, root)
    nodes.extend(peng_node)
    return state


def touching(root, tree, nodes, server):
    node = nodes[-1]
    # 如果牌墙还有牌，那么就摸牌，扩展子树
    if len(server.tiles) > 0:
        touch_tile(server, node, nodes)
        return MahjongState.Zimo

    return MahjongState.Over


def tree_descend(tree: TreeNode, server, state: MahjongState):
    root = True
    nodes = [tree]
    while state is not MahjongState.Over:
        func = switcher.get(state)()
        state = func(root, tree, nodes, server)
        tree = nodes[-1]
        if root:
            root = False
    return nodes


def touch_tile(server, node, nodes):
    # 如果牌墙还有牌，那么就摸牌，扩展子树
    if len(server.tiles) > 0:
        tile = server.tiles.pop(0)
        child = node.clone()
        child.touch(tile)
        nodes.append(child)
        logger().info("Player{} touch tile{}".format(child.get_turn(), Tile(tile)))


def fangpao_check(node, root):
    nodes = []
    for index_fangpao in range(0, PLAYER_NUM - 1):
        think_fangpao_index = node.game_state.get_next_turn(index_fangpao)
        discard = node.get_last_discard()
        if Attack.think_fangpao(node, node.game_state, think_fangpao_index, discard):
            random.shuffle(node.children)  # randomize the max in case of equal urgency
            urgencies = global_puct_urgency(node.v, *puct_urgency_input(node.children))
            fangpao_probabilitys_np = np.array([n.fangpao_probability for n in node.children])
            if root:
                dirichlet = np.random.dirichlet((0.03, 1), len(node.children))
                urgencies = urgencies * 0.5 + dirichlet[:, 0] * 0.25 + fangpao_probabilitys_np * 0.25
            else:
                urgencies = urgencies * 0.7 + fangpao_probabilitys_np * 0.3
            log.debug("urgencies:{}".format(urgencies))
            child = max(zip(node.children, urgencies), key=lambda t: t[1])[0]
            nodes.append(child)
            hand_fangpao = child.get_hands_str_index(think_fangpao_index)
            if child.fangpao_to_index >= 0:
                logger().info(
                    "fangpao===>player{} to player{} with hands{}".format(child.fangpao_from_index,
                                                                          child.fangpao_to_index,
                                                                          hand_fangpao))
                return nodes, MahjongState.Over
            else:
                logger().info("pass_fangpao====>player{} pass_fangpao player{} with hands{}".format(
                    think_fangpao_index, child.game_state.get_next_turn(-1)
                    , hand_fangpao))
    return nodes, MahjongState.Ganging


def gang_check(node):
    """
    杠牌判断，判断是否杠牌，node为丢弃牌的节点
    :param node:
    :return:
    """
    # if node.get_discard_tile() < 0:
    #     logger().error("The node is not a discard node when checking peng.")
    #     return None
    # for index in range(0, PLAYER_NUM - 1):
    #     think_gang_index = node.game_state.get_next_turn(index)
    #     if Attack.think_peng(node.game_state, think_gang_index, node.get_discard_tile()):
    #         peng_node = node.clone()
    #         peng_node.peng(think_gang_index, node.get_discard_tile())
    #         logger().info("Player:{} peng tile:{}".format(think_gang_index, node.get_discard_tile()))
    #         return peng_node
    return [], MahjongState.Penging


def peng_check(node, root):
    """
    判断是否碰牌，node为丢弃牌的节点
    :param node:
    :return:
    """
    nodes = []

    for index in range(0, PLAYER_NUM - 1):
        think_peng_index = node.game_state.get_next_turn(index)
        if not Attack.think_peng(node, node.game_state, think_peng_index, node.get_last_discard()):
            continue
        random.shuffle(node.children)  # randomize the max in case of equal urgency
        urgencies = global_puct_urgency(node.v, *puct_urgency_input(node.children))
        attack_probabilitys_np = np.array([n.attack_drop_p for n in node.children])
        if root:
            dirichlet = np.random.dirichlet((0.03, 1), len(node.children))
            urgencies = urgencies * 0.5 + dirichlet[:, 0] * 0.25 + attack_probabilitys_np * 0.25
        urgencies = urgencies * 0.7 + attack_probabilitys_np * 0.3
        log.debug("urgencies:{}".format(urgencies))
        child = max(zip(node.children, urgencies), key=lambda t: t[1])[0]
        if child.peng_tile > 0:
            log.info("Player:{} peng tile:{} drop{} hands{}".format(child.peng_index, Tile(child.peng_tile),
                                                                    Tile(child.discard_tile),
                                                                    child.get_hands_str_index(child.peng_index)))
            child.v += 1
            nodes.append(child)
            return nodes, MahjongState.Fangpaoing
        else:
            log.info("Player:{} pass peng".format(child.pass_p))
        child.v += 1
        nodes.append(child)
    return nodes, MahjongState.Touching


def tree_update(nodes):
    scores = [0] * 3
    """ Store simulation result in the tree (@nodes is the tree path) """
    for node in reversed(nodes):
        if node.game_result is not None:
            if node.reason == "zimo":
                turn = node.game_state.turn
                score = HandCalculator.calc_score_for_results(node.game_result)
                scores[turn] += score
                for i in range(1, const.PLAYER_NUM):
                    scores[(turn + i) % 3] -= score
            elif node.reason == "fangpao":
                lose_turn = node.fangpao_from_index
                win_turn = node.fangpao_to_index
                score = HandCalculator.calc_score_for_results(node.game_result)
                scores[win_turn] += score
                scores[lose_turn] -= score
        else:
            node.w += scores[node.game_state.turn]  # score is for to-play, node statistics for just-played


def tree_search(tree, n, game_server: GameServer, state):
    """ Perform MCTS search from a given state for a given #iterations """
    """ 模特卡罗搜索最佳出牌 """

    i = 0
    # 模拟对局n次,每次需要对剩余牌面重新洗牌
    while i < n:
        # 代表自己的Client,代表游戏的Server,搜索过程会改变对象信息，所以需要克隆
        server = game_server.clone()

        # 对剩余牌墙重新洗牌
        random.shuffle(server.tiles)

        # 模拟对局
        nodes = tree_descend(tree, server, state)
        logger().info("Start scan nodes...")
        print_nodes(nodes)
        i += 1
        logger().info("simulation {} over,total:{} \n".format(i, n))
        last_node = nodes[-1]
        if last_node.game_result is not None:
            tree_update(nodes)
        else:
            continue

    return tree.best_move(True)


def print_nodes(nodes):
    first = True
    for node in nodes:
        if first:
            first = False
            continue
        if node.touch_tile >= 0:
            logger().info("player{} touch tile:{}".format(node.game_state.turn, Tile(node.touch_tile)))
        if node.discard_tile >= 0:
            logger().info("player{} drop tile:{}".format((node.game_state.turn + 2) % 3, Tile(node.discard_tile)))
        if node.peng_tile >= 0:
            logger().info("player{} peng tile:{}".format(node.peng_index, Tile(node.peng_tile)))
        if node.pass_p > 0:
            logger().info("player{} pass peng".format(node.pass_p))

        if node.game_result != 0:
            if node.reason == "zimo":
                logger().info("player{} zimo. ".format(node.game_state.turn))
            elif node.reason == "fangpao":
                logger().info("player{} fangpao to player{}".format(node.fangpao_from_index, node.fangpao_to_index))

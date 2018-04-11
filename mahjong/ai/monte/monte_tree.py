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
        self.after_touch_probability = 0
        self.beat_probability = 0
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
        self.gang_index = -1
        self.gang_tile = -1
        self.touch_gang_index = -1
        self.melds3_gang_index = -1  # 摸牌后手牌有一张碰牌有三张
        self.touch_gang_tile = -1
        self.melds3_gang_tile = -1
        self.zimo_tile = -1
        self.fangpao_from_index = -1
        self.fangpao_to_index = -1
        self.reason = "dogfall"
        self.pass_beat_index = -1
        self.decision_index = -1  # 做决策的是哪个玩家

    def get_hands_str_index(self, index):
        return self.game_state.get_hands_str_index(index)

    def get_last_discard(self):
        return self.game_state.last_discard

    def get_discard_tile(self):
        return self.discard_tile

    def get_turn(self):
        return self.game_state.turn

    def zimo(self, result, t):
        self.game_result = result
        self.reason = "zimo"
        self.zimo_tile = t

    def melds3_gang(self, tile):
        self.game_state.hands[self.get_turn()][tile] -= 1
        self.game_state.melds_4[self.get_turn()].append(tile)
        self.melds3_gang_tile = tile
        self.melds3_gang_index = self.get_turn()
        self.game_state.melds_3[self.get_turn()].remove(tile)

    def touch_gang(self, tile):
        self.game_state.hands[self.get_turn()][tile] -= 4
        self.game_state.melds_4[self.get_turn()].append(tile)
        self.touch_gang_tile = tile
        self.touch_gang_index = self.get_turn()

    def gang(self, index, gang_tile):
        self.game_state.hands[index][gang_tile] -= 3
        self.game_state.melds_4[index].append(gang_tile)
        self.game_state.turn = index
        self.gang_index = index
        self.gang_tile = gang_tile

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

    def pass_beat(self, index):
        self.pass_beat_index = index

    def set_beat_probability(self, p):
        self.beat_probability = p

    def set_after_touch_probability(self, p):
        self.after_touch_probability = p

    def clone(self):
        node = TreeNode(None, self.game_state.clone())
        return node

    def touch(self, tile):
        self.game_state.touch_tile(tile)
        self.touch_tile = tile

    def discard(self, tile):
        self.game_state.discard(tile)
        self.discard_tile = tile

    def expand_beating(self, hand, discard, index):
        self.children = []
        hand[discard] += 1
        is_win = WinChecker.is_win(hand)
        hand[discard] -= 1
        if is_win:
            meld3 = self.game_state.melds_3[index]
            meld4 = self.game_state.melds_4[index]
            hand[discard] += 1
            result_fangpao = HandCalculator.estimate_max_score(hand, discard, meld3, meld4)
            hand[discard] -= 1
            result_node = self.clone()
            result_node.decision_index = index
            value = UNIFORM_DISTRIBUTION[0]
            result_node.pv = PRIOR_NET
            result_node.pw = PRIOR_NET * value
            result_node.fangpao(self.game_state.get_next_turn(-1), index,
                                result_fangpao)
            self.children.append(result_node)
        if hand[discard] >= 3:
            logger().info("can gang")
            node = self.clone()
            node.decision_index = index
            node.gang(index, discard)
            value = UNIFORM_DISTRIBUTION[0]
            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * value
            self.children.append(node)
        if hand[discard] >= 2:
            for t in range(0, len(hand)):
                if t != discard:
                    if hand[t] <= 0:
                        continue
                else:
                    if hand[t] <= 2:
                        continue
                node = self.clone()
                node.decision_index = index
                node.peng(index, discard, t)
                value = UNIFORM_DISTRIBUTION[t]
                node.pv = PRIOR_NET
                node.pw = PRIOR_NET * value
                self.children.append(node)

        node = self.clone()
        node.decision_index = index
        node.pass_beat(index)
        value = UNIFORM_DISTRIBUTION[0]
        node.pv = PRIOR_NET
        node.pw = PRIOR_NET * value
        self.children.append(node)

    def expend_after_touch(self):
        self.children = []

        is_win = WinChecker.is_win(self.game_state.hands_index(self.get_turn()))
        if is_win:
            meld3 = self.game_state.melds_3[self.get_turn()]
            meld4 = self.game_state.melds_4[self.get_turn()]
            score = HandCalculator.estimate_max_score(self.game_state.hands_index(self.get_turn()), self.touch_tile,
                                                      meld3, meld4)
            node1 = self.clone()
            node1.decision_index = self.get_turn()
            node1.zimo(score, self.touch_tile)
            value = UNIFORM_DISTRIBUTION[0]
            node1.pv = PRIOR_NET
            node1.pw = PRIOR_NET * value
            self.children.append(node1)

        hand = self.game_state.hands_index(self.get_turn())

        # discarding
        for t in range(0, len(hand)):
            if hand[t] <= 0:
                continue
            node2 = self.clone()
            node2.decision_index = self.get_turn()
            node2.discard(t)
            value = UNIFORM_DISTRIBUTION[0]
            node2.pv = PRIOR_NET
            node2.pw = PRIOR_NET * value
            self.children.append(node2)

        # touch gang
        for t in range(0, len(hand)):
            if hand[t] != 4:
                continue
            node3 = self.clone()
            node3.decision_index = self.get_turn()
            node3.touch_gang(t)
            value = UNIFORM_DISTRIBUTION[0]
            node3.pv = PRIOR_NET
            node3.pw = PRIOR_NET * value
            self.children.append(node3)
        # meld3 gang
        for t in range(0, len(hand)):
            if hand[t] != 1:
                continue
            meld3 = self.game_state.melds_3[self.get_turn()]
            if t not in meld3:
                continue
            node4 = self.clone()
            node4.decision_index = self.get_turn()
            node4.melds3_gang(t)
            value = UNIFORM_DISTRIBUTION[0]
            node4.pv = PRIOR_NET
            node4.pw = PRIOR_NET * value
            self.children.append(node4)

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
    AfterTouching = 4
    Fangpaoing = 5
    Touching = 6
    Beating = 7
    # TouchGanging = 7
    Over = 10


switcher = {
    MahjongState.AfterTouching: lambda: after_touching,
    MahjongState.Touching: lambda: touching,
    MahjongState.Beating: lambda: beating,
}


def need_check(hand, tile):
    hand[tile] += 1
    if WinChecker.is_win(hand):
        hand[tile] -= 1
        return True
    hand[tile] -= 1
    if hand[tile] >= 2:
        return True

    return False


def beating_check(node: TreeNode, root):
    nodes = []
    for index in range(0, PLAYER_NUM - 1):
        think_beat_index = node.game_state.get_next_turn(index)
        hand = node.game_state.hands_index(think_beat_index)
        discard = node.get_last_discard()
        if not need_check(hand, discard):
            continue
        if node.children is None:
            node.expand_beating(hand, discard, think_beat_index)
        Attack.think_beat(node, think_beat_index)
        random.shuffle(node.children)  # randomize the max in case of equal urgency
        net_pro = global_puct_urgency(node.v, *puct_urgency_input(node.children))
        beat_probabilitys_np = np.array([n.beat_probability for n in node.children])
        if root:
            dirichlet = np.random.dirichlet((0.03, 1), len(node.children))
            net_pro = net_pro * 0.5 + dirichlet[:, 0] * 0.25 + beat_probabilitys_np * 0.25
        else:
            net_pro = net_pro * 0.7 + beat_probabilitys_np * 0.3
        log.debug("net_pro:{}".format(net_pro))
        child = max(zip(node.children, net_pro), key=lambda t: t[1])[0]
        child.v += 1
        nodes.append(child)
        if child.fangpao_from_index >= 0:
            # fangpao
            logger().info(
                "Player{} fangpao to Player:{} hands{}".format(child.fangpao_from_index, child.fangpao_to_index,
                                                               child.get_hands_str_index(child.fangpao_to_index)))
            return nodes, MahjongState.Over
        elif child.gang_index >= 0:
            # gang
            logger().info("Player{} gang:{} hands{}".format(child.gang_index, Tile(child.gang_tile),
                                                            child.get_hands_str_index(child.gang_index)
                                                            ))
            return nodes, MahjongState.Touching
        elif child.peng_index >= 0:
            # peng
            logger().info("Player:{} peng tile:{} drop tile:{} hands{}".format(child.peng_index, Tile(child.peng_tile),
                                                                               Tile(child.discard_tile),
                                                                               child.get_hands_str_index(
                                                                                   child.peng_index)))
            return nodes, MahjongState.Beating
        else:
            logger().info("Player:{} pass beat with hand: {}".format(child.pass_beat_index,
                                                                     child.get_hands_str_index(child.pass_beat_index)))
    return nodes, MahjongState.Touching


def after_touch_check(node: TreeNode, root):
    nodes = []
    if node.children is None:
        node.expend_after_touch()
    Attack.think_after_touch(node)
    random.shuffle(node.children)  # randomize the max in case of equal urgency
    urgencies = global_puct_urgency(node.v, *puct_urgency_input(node.children))
    after_touch_probabilitys_np = np.array([n.after_touch_probability for n in node.children])
    if root:
        dirichlet = np.random.dirichlet((0.03, 1), len(node.children))
        urgencies = urgencies * 0.5 + dirichlet[:, 0] * 0.25 + after_touch_probabilitys_np * 0.25
    else:
        urgencies = urgencies * 0.7 + after_touch_probabilitys_np * 0.3
    log.debug("urgencies:{}".format(urgencies))
    child = max(zip(node.children, urgencies), key=lambda t: t[1])[0]
    child.v += 1
    nodes.append(child)
    if child.zimo_tile >= 0:
        # 自摸
        logger().info("Player{} zimo:{} hands{}".format(child.get_turn(), child.zimo_tile,
                                                        child.get_hands_str_index(child.get_turn())))
        return nodes, MahjongState.Over
    elif child.touch_gang_index >= 0:
        # 杠了牌
        logger().info("Player{} touch gang:{} hands{}".format(child.get_turn(), Tile(child.touch_gang_tile),
                                                              child.get_hands_str_index(child.get_turn())
                                                              ))
        return nodes, MahjongState.Touching
    elif child.melds3_gang_index >= 0:
        # melds3 杠牌
        logger().info("Player{} melds3 gang:{} hands{}".format(child.get_turn(), Tile(child.melds3_gang_tile),
                                                               child.get_hands_str_index(child.get_turn())))
        return nodes, MahjongState.Touching
    else:
        logger().info("Player{} drop: {} with hand: {}".format(node.get_turn(), Tile(child.discard_tile),
                                                               child.get_hands_str_index(node.get_turn())))
        return nodes, MahjongState.Beating


def beating(root, tree, nodes, server):
    ns, state = beating_check(tree, root)
    nodes.extend(ns)
    return state


def after_touching(root, tree, nodes, server):
    ns, state = after_touch_check(tree, root)
    nodes.extend(ns)
    return state


def touching(root, tree, nodes, server):
    node = nodes[-1]
    # 如果牌墙还有牌，那么就摸牌，扩展子树
    if len(server.tiles) > 0:
        touch_tile(server, node, nodes)
        return MahjongState.AfterTouching

    return MahjongState.Over


def tree_descend(tree: TreeNode, server, state: MahjongState):
    root = True
    nodes = [tree]
    tree.v += 1
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
        child.decision_index = child.get_turn()
        child.touch(tile)
        child.v += 1
        nodes.append(child)
        logger().info("Player{} touch tile{} hand:{}".format(child.get_turn(), Tile(tile),
                                                             child.get_hands_str_index(child.get_turn())))


def tree_update(nodes):
    scores = [0] * 3
    """ Store simulation result in the tree (@nodes is the tree path) """
    for node in reversed(nodes):
        if node.game_result is not None:
            if node.reason == "zimo":
                turn = node.game_state.turn
                score = node.game_result
                scores[turn] += score * 2
                for i in range(1, const.PLAYER_NUM):
                    scores[(turn + i) % 3] -= score
            elif node.reason == "fangpao":
                lose_turn = node.fangpao_from_index
                win_turn = node.fangpao_to_index
                score = node.game_result
                scores[win_turn] += score
                scores[lose_turn] -= score
        else:
            node.w += scores[node.decision_index]  # score is for to-play, node statistics for just-played


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

        if node.game_result != 0:
            if node.reason == "zimo":
                logger().info("player{} zimo. ".format(node.game_state.turn))
            elif node.reason == "fangpao":
                logger().info("player{} fangpao to player{}".format(node.fangpao_from_index, node.fangpao_to_index))

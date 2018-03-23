import random

import numpy as np
import math

import sys

from zigong_majiang.ai.game_state import GameState
from zigong_majiang.ai.pure_attack.attack import Attack
from zigong_majiang.rule.agari import Agari
from zigong_majiang.rule.hand_calculator import HandCalculator
from zigong_majiang.simulator.client import Client
from zigong_majiang.simulator.game_server import GameServer

N = 7
W = N + 2
empty = "\n".join([(N + 1) * ' '] + N * [' ' + N * '.'] + [(N + 2) * ' '])
colstr = 'ABCDEFGHJKLMNOPQRST'

N_SIMS = 500
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
        self.v = 0
        self.w = 0
        self.pv = 0
        self.pw = 0
        self.av = 0
        self.aw = 0
        self.children = None
        self.touch_tile = 0
        self.discard_tile = 0

    def set_attack_drop_p(self, p):
        self.attack_drop_p = p

    def clone(self):
        node = TreeNode(None, self.game_state.clone())
        return node

    def touch(self, tile):
        self.game_state.touch_tile(tile)
        self.touch_tile = tile

    def discard(self, tile):
        self.game_state.discard(tile)
        self.discard_tile = tile

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

    @staticmethod
    def player_num():
        return PLAYER_NUM

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


def tree_descend(tree: TreeNode, server, disp=False):
    """ Descend through the tree to a leaf """
    """ 蒙特卡洛模拟打麻将，每一个节点代表打牌后的牌面,结束条件是有人和牌或者牌被摸完 """
    """ 目前采取随机打牌的策略 """
    tree.v += 1
    nodes = [tree]
    index = 0

    # Initialize root node
    # 每个节点代表自己或对手打牌后的牌面
    if tree.children is None:
        tree.expand()

    while nodes[-1].children is not None:
        # 如果已经和牌，直接退出
        tree = nodes[-1]
        is_win = Agari.is_win_zigong(tree.game_state.hands_index(index))
        if is_win:
            print("开始计算",tree.game_state.hands_index(index))
            nodes[-1].game_result = HandCalculator.estimate_hand_value_zigong(tree.game_state.hands_index(index),
                                                                              tree.game_state.hands_index(index)[0])
            print("结束计算")
            return nodes

        children = list(nodes[-1].children)
        attack_probabilitys = Attack.think(nodes[-1].game_state)
        for index in range(0, len(children)):
            children[index].set_attack_drop_p(attack_probabilitys[index])
        # Pick the most urgent child
        random.shuffle(children)  # randomize the max in case of equal urgency

        urgencies = global_puct_urgency(nodes[-1].v, *puct_urgency_input(children))
        attack_probabilitys_np = np.array([n.attack_drop_p for n in children])
        dirichlet = np.random.dirichlet((0.03, 1), len(children))
        urgencies = urgencies * 0.5 + dirichlet[:, 0] * 0.25 + attack_probabilitys_np * 0.25

        node = max(zip(children, urgencies), key=lambda t: t[1])[0]
        nodes.append(node)

        # updating visits on the way *down* represents "virtual loss", relevant for parallelization
        node.v += 1

        # 如果牌墙还有牌，那么就摸牌，扩展子树
        index += 1
        index %= PLAYER_NUM

        if len(server.tiles) > 0:
            tile = server.tiles.pop(0)
            child = node.clone()
            child.touch(tile)
            nodes.append(child)
            # 扩展子树
            child.expand()

    return nodes


def tree_update(nodes, score, disp=False):
    """ Store simulation result in the tree (@nodes is the tree path) """
    for node in reversed(nodes):
        if node.game_state.turn == nodes[0].game_state.turn:
            node.w += score  # score is for to-play, node statistics for just-played
        else:
            node.w -= score

def tree_search(tree, n, game_server: GameServer, disp=False, debug_disp=False):
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
        nodes = tree_descend(tree, server, disp=debug_disp)
        print_nodes(nodes)
        i += 1
        print(i," ",n)
        last_node = nodes[-1]
        if last_node.game_result is not None:
            # 计算最大得分
            max_cost = -1
            for result in last_node.game_result:
                if result.cost > max_cost:
                    max_cost = result.cost

        else:
            continue
        tree_update(nodes, max_cost, disp=debug_disp)


    return tree.best_move(True)


def print_nodes(nodes):
    first = True
    for node in nodes:
        if first:
            first = False
            continue
        print("玩家{}打牌:{},玩家{}摸牌:{}".format((node.game_state.turn + 2) % 3, node.discard_tile, node.game_state.turn,
                                           node.touch_tile))
    if nodes[-1].game_result is None:
        print("平局")
    else:
        print(nodes[-1].game_result.__str__())

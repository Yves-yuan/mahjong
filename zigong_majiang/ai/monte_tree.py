import random

import numpy as np
import math

import sys

from zigong_majiang.ai.game_state import GameState
from zigong_majiang.rule.hand_calculator import HandCalculator
from zigong_majiang.simulator.client import Client
from zigong_majiang.simulator.game_server import GameServer

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
        self.v = 0
        self.w = 0
        self.pv = 0
        self.pw = 0
        self.av = 0
        self.aw = 0
        self.children = None

    def expand(self):
        """ add and initialize children to a leaf node """
        """ 扩展子节点，包括自己打出的牌和对手摸牌打牌 """
        distribution = self.net.predict_distribution(self.game_state)
        self.children = []
        for c in self.game_state.hands():
            pos2 = 0
            if pos2 is None:
                continue
            node = TreeNode(self.net, pos2)
            self.children.append(node)
            x, y = c % W - 1, c // W - 1
            value = distribution[y * N + x]

            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * value

        # Add also a pass move - but only if this doesn't trigger a losing
        # scoring (or we have no other option)
        if not self.children:
            can_pass = True
        else:
            can_pass = self.game_state.score() >= 0

        if can_pass:
            node = TreeNode(self.net, self.game_state.pass_move())
            self.children.append(node)
            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * distribution[-1]

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


def tree_descend(tree: TreeNode, server, disp=False):
    """ Descend through the tree to a leaf """
    """ 蒙特卡洛模拟打麻将，每一个节点代表打牌后的牌面,结束条件是有人和牌或者牌被摸完 """
    """ 目前采取随机打牌的策略 """
    tree.v += 1
    nodes = [tree]
    passes = 0
    root = True
    index = 0

    # Initialize root node
    # 每个节点代表自己或对手打牌后的牌面
    if tree.children is None:
        tree.expand()

    while nodes[-1].children is not None:
        # 如果已经和牌，直接退出
        result = HandCalculator.estimate_hand_value_zigong(tree.game_state.hands_index(index),
                                                           tree.game_state.hands_index(index)[0])
        if result.is_win:
            nodes[-1].game_result = result
            return nodes

        children = list(nodes[-1].children)

        # 代表是自己的轮
        # 每一次循环处理打牌的决定和摸牌，摸牌后扩展子树
        # Pick the most urgent child
        random.shuffle(children)  # randomize the max in case of equal urgency
        urgencies = global_puct_urgency(nodes[-1].v, *puct_urgency_input(children))
        if root:
            dirichlet = np.random.dirichlet((0.03, 1), len(children))
            urgencies = urgencies * 0.75 + dirichlet[:, 0] * 0.25
            root = False
        node = max(zip(children, urgencies), key=lambda t: t[1])[0]
        nodes.append(node)

        if node.pos.last is None:
            passes += 1
        else:
            passes = 0

        # updating visits on the way *down* represents "virtual loss", relevant for parallelization
        node.v += 1

        # 如果牌墙还有牌，那么就摸牌，扩展子树
        index += 1
        index %= PLAYER_NUM

        if len(server.tiles) > 0:
            tile = server.tiles.pop[0]
            node.game_state.touch_tile(index, tile)
            # 扩展子树
            node.expand()

    return nodes


def tree_update(nodes, amaf_map, score, disp=False):
    """ Store simulation result in the tree (@nodes is the tree path) """
    for node in reversed(nodes):
        node.w += score < 0  # score is for to-play, node statistics for just-played
        # Update the node children AMAF stats with moves we made
        # with their color
        amaf_map_value = 1 if node.pos.n % 2 == 0 else -1
        if node.children is not None:
            for child in node.children:
                if child.pos.last is None:
                    continue
                if amaf_map[child.pos.last] == amaf_map_value:
                    child.aw += score > 0  # reversed perspective
                    child.av += 1
        score = -score


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

        i += 1

        last_node = nodes[-1]
        if last_node.pos.last is None and last_node.pos.last2 is None:
            score = 1 if last_node.pos.score() > 0 else -1
        else:
            score = tree.net.predict_winrate(last_node.pos)

        tree_update(nodes, score, disp=debug_disp)

    return tree.best_move(tree.pos.n <= PROPORTIONAL_STAGE)

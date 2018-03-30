from mahjong.ai.monte.game_state import GameState
from mahjong.ai.monte.monte_tree import TreeNode, tree_search, N_SIMS, MahjongState
from mahjong.log.logger import Logger
from mahjong.rule.model.tile import Tile
from mahjong.rule.util.tile_convert import TilesConv
from mahjong.simulator.client import Client
from mahjong.simulator.game_server import GameServer
import logging
import cProfile


def test_monte():
    Logger.init()
    log = logging.getLogger("mahjong")
    server = GameServer()
    client1 = Client(1)
    client2 = Client(2)
    client3 = Client(3)
    server.bind(client1)
    server.bind(client2)
    server.bind(client3)
    server.init()
    server.deal()
    tile = server.tiles.pop(0)
    client1.touch_tile(tile)

    hands = [client1.hands(),
             client2.hands(),
             client3.hands()]
    discards = [[], [], []]
    melds_3 = [[], [], []]
    melds_4 = [[], [], []]
    for hand in hands:
        log.info(TilesConv.tiles_18_to_str(hand))
    game_state = GameState(hands=hands, discards=discards, melds_3=melds_3, melds_4=melds_4)
    game_state.check()

    tree_node = TreeNode(net=None, game_state=game_state)

    node = tree_search(tree_node, N_SIMS, server, MahjongState.Discarding)
    log.info("The result of monte tree search is :touch tile {} for given hand {}".
             format(Tile(node.discard_tile), TilesConv.tiles_18_to_str(hands[0])))


cProfile.run("test_monte()")

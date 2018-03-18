import copy
import random

from zigong_majiang.rule.agari import Agari
from zigong_majiang.rule.hand_calculator import HandCalculator
from zigong_majiang.rule.tile_convert import TilesConverter
from zigong_majiang.simulator.game_result import GameResult


def tiles_to_string(tiles):
    sort_tiles = sorted(tiles)
    tongzi = ""
    tiaozi = ""
    for tile in sort_tiles:
        if tile < 9:
            tongzi += ('0' + tile)
        else:
            tiaozi += ('0' + tile - 9)
    return tongzi, tiaozi


class Client:
    def __init__(self, client_id: int):
        self.tiles_18 = [0] * 18
        self.last_tile = 0
        self.calculator = HandCalculator()
        self.agari = Agari()
        self.id = client_id
        self.opponents = []

    def set_opponent(self, opponent):
        self.opponents.append(opponent)

    def play_hand(self):
        while True:
            r = random.randint(0, 17)
            if self.tiles_18[r] > 0:
                self.tiles_18[r] -= 1
                break
        return r

    def touch_tile(self, tile: int):
        self.tiles_18[tile] += 1
        self.last_tile = tile

    def estimate_hand_value(self, tile):
        is_win = Agari.is_win_zigong(self.tiles_18)
        if is_win:
            results = self.calculator.estimate_hand_value_zigong(self.tiles_18, tile)
            return GameResult(self.id, True, results)
        else:
            return GameResult(self.id)

    def hand_str(self):
        tongzi, tiaozi = TilesConverter.tiles_18_to_str(self.tiles_18)
        r = "tongzi:{} tiaozi:{}".format(tongzi, tiaozi)
        return r

    def hands(self):
        return self.tiles_18

    def clone(self):
        client = Client(self.id)
        client.last_tile = self.last_tile
        client.tiles_18 = copy.deepcopy(self.tiles_18)
        return client

    def __str__(self):
        return 'client id :{} , hand:{} '.format(self.id, self.tiles_18)

from zigong_majiang.rule.Agari import Agari
from zigong_majiang.rule.hand import HandCalculator
from zigong_majiang.rule.tile import TilesConverter


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


class Client(object):
    def __init__(self, client_id: int):
        self.tiles = [int]
        self.calculator = HandCalculator()
        self.agari = Agari()
        self.id = client_id

    def play_hand(self, index):
        self.tiles.pop(index)

    def touch_tile(self, tile: int):
        self.tiles.append(tile)

    def estimate_hand_value(self, tile):
        tongzi, tiaozi = tiles_to_string(self.tiles)
        tiles_72 = TilesConverter.string_to_72_array(tongzi=tongzi, tiaozi=tiaozi)
        tiles_18 = TilesConverter.to_18_array(tiles_72)
        is_win = self.agari.is_agari_zigong(tiles_18)
        if is_win:
            results = self.calculator.estimate_hand_value_zigong(tiles_72, tile)
            return results
        else:
            return []

    def __str__(self):
        return 'client id :{} , hand:{} '.format(self.id, self.tiles)

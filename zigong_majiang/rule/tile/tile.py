from zigong_majiang.rule.tile.tile_convert import TilesConverter


class Tile(object):
    def __init__(self, tile):
        self.tile = tile

    def __str__(self):
        return TilesConverter.tile_to_string(self.tile)


class Hands(object):
    def __init__(self, tiles_18):
        self.hands = []
        for index in range(0, 18):
            if tiles_18[index] > 0:
                for i in range(0, tiles_18[index]):
                    self.hands.append(Tile(index))

    def __str__(self):
        hands = "["
        for tile in self.hands:
            hands += tile.__str__()
            hands += ","
        hands = hands[:-1]
        hands += "]"
        return hands

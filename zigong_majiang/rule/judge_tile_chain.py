from zigong_majiang.rule.tile import Tile


class JudgeTileChain(object):
    def __init__(self, touch_play_pairs, draw_hands):
        self.touchPlayPairs = touch_play_pairs
        self.drawHands = draw_hands

    def __str__(self):
        ret = "胡牌链:"
        for tpp in self.touchPlayPairs:
            ret += tpp.__str__()

        ret += "胡牌:"
        for tile in self.drawHands:
            ret += Tile(tile).__str__()

        return ret


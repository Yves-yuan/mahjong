from zigong_majiang.rule.tile.tile import Tile


class JudgeTileChain(object):

    def __init__(self, touch_play_pairs, draw_hands,hand):
        """
        胡牌链
        :param touch_play_pairs:摸牌打牌列表
        :param draw_hands: 听牌值
        :param hand: 听牌时的手牌
        """
        self.touchPlayPairs = touch_play_pairs
        self.drawHands = draw_hands
        self.hand = hand

    def __str__(self):
        ret = "胡牌链:"
        for tpp in self.touchPlayPairs:
            ret += tpp.__str__()

        ret += "胡牌:"
        for tile in self.drawHands:
            ret += Tile(tile).__str__()

        return ret


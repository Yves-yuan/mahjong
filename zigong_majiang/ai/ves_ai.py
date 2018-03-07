from zigong_majiang.rule.Agari import Agari


class VesAI(object):

    def __init__(self, n):
        self.n = n
        self.agari = Agari()
    def calc_effective_cards(self,tiles_18):
        for card in range(0,18):
            
            print(card)



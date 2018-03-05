import random

from zigong_majiang.simulator.Client import Client

DefaultGamePlayerNum = 3
Tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


class GameServer(object):
    def __init__(self, size=DefaultGamePlayerNum):
        self.size = size
        self.tiles = [0] * 72
        self.clients = [] * size

    def start_game(self):
        for index in range(len(Tiles)):
            self.tiles[4 * index] = Tiles[index]
            self.tiles[4 * index + 1] = Tiles[index]
            self.tiles[4 * index + 2] = Tiles[index]
            self.tiles[4 * index + 3] = Tiles[index]

        print("shuffle")
        random.shuffle(self.tiles)
        print(self)
        # deal,fa pai
        print("deal")
        self.deal()

        #show
        print(self)
        print(self.clients[0])
        print(self.clients[1])
        print(self.clients[2])

    def bind(self, client: Client):
        self.clients.append(client)

    def deal(self):
        # deal
        for index in range(0, 13):
            for ci in self.clients:
                tile = self.tiles.pop(0)
                ci.touch_tile(tile)

    def __str__(self):
        return 'server tiles:{} '.format(self.tiles)

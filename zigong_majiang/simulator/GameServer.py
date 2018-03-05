import random

DefaultGamePlayerNum = 3
Tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


class GameServer(object):
    def __init__(self, size=DefaultGamePlayerNum):
        self.size = size
        self.tiles = [0]*72

    def start(self):
        for index in range(len(Tiles)):
            print(index)
            self.tiles[4 * index] = Tiles[index]
            self.tiles[4 * index + 1] = Tiles[index]
            self.tiles[4 * index + 2] = Tiles[index]
            self.tiles[4 * index + 3] = Tiles[index]
        random.shuffle(self.tiles)
        print(self.tiles)

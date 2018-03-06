import random

from zigong_majiang.simulator.Client import Client

DefaultGamePlayerNum = 3
Tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
First = 0


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

        # Show cards
        print(self)
        print(self.clients[0])
        print(self.clients[1])
        print(self.clients[2])
        print("")
        print("Playing start")
        # Choose first one to play-card
        index = First
        while True:
            if len(self.tiles) == 0:
                print("Drawn game")
                break
            tile = self.tiles.pop(0)
            self.clients[index].touch_tile(tile)
            result = self.clients[index].estimate_hand_value(tile)
            if not result.is_win:
                card = self.clients[index].play_hand()
                print("Player:{} play card:{} hands:{}".format(self.clients[index].id, card,self.clients[index].hands()))
                print("Remain cards on desktop:",self.tiles)
                # Inform others
            else:
                print(result)
                break

            index += 1
            index %= 3

        print("Game over!")

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

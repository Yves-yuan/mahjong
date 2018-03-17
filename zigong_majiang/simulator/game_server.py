import copy
import random

from zigong_majiang.simulator.client import Client

DefaultGamePlayerNum = 3
Tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
First = 0


class GameServer(object):
    def __init__(self, size=DefaultGamePlayerNum):
        self.size = size
        self.tiles = [0] * 72
        self.clients = [] * size
        self.index = First

    def init(self):
        for index in range(0, len(self.clients)):
            self.clients[index].set_opponent(self.clients[(index + 1) % 3])

        for index in range(len(Tiles)):
            self.tiles[4 * index] = Tiles[index]
            self.tiles[4 * index + 1] = Tiles[index]
            self.tiles[4 * index + 2] = Tiles[index]
            self.tiles[4 * index + 3] = Tiles[index]

    def start_game(self):
        # 初始化
        self.init()

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
        while True:
            if len(self.tiles) == 0:
                print("Drawn game")
                break
            tile = self.tiles.pop(0)
            self.clients[self.index].touch_tile(tile)
            result = self.clients[self.index].estimate_hand_value(tile)
            if not result.is_win:
                card = self.clients[self.index].play_hand()
                print(
                    "Player:{} play card:{} hands:{}".format(self.clients[self.index].id, card,
                                                             self.clients[self.index].hand_str()))
                print("Remain cards on desktop:", self.tiles)
                # Inform others
            else:
                print(result)
                break

            self.index += 1
            self.index %= 3

        print("Game over!")

    def bind(self, client: Client):
        self.clients.append(client)

    def deal(self):
        # deal
        random.shuffle(self.tiles)
        for index in range(0, 13):
            for ci in self.clients:
                tile = self.tiles.pop(0)
                ci.touch_tile(tile)

    def clone(self):
        game_server = GameServer
        game_server.index = self.index
        game_server.clients = [client.clone() for client in self.clients]
        game_server.tiles = copy.deepcopy(self.tiles)
        return game_server

    def __str__(self):
        return 'server tiles:{} '.format(self.tiles)

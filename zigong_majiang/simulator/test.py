from zigong_majiang.simulator.Client import Client
from zigong_majiang.simulator.GameServer import GameServer

server = GameServer()
client1 = Client(1)
client2 = Client(2)
client3 = Client(3)
server.bind(client1)
server.bind(client2)
server.bind(client3)
server.start_game()

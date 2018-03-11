from itertools import permutations

from zigong_majiang.ai.perm_comb import PermCombGenerator
from zigong_majiang.ai.ves_ai import VesAI
from zigong_majiang.rule.tile import Hands
from zigong_majiang.rule.tile_convert import TilesConverter

Tiles = [0, 0, 0, 0,
         1, 1, 1, 1,
         2, 2, 2, 2,
         3, 3, 3, 3,
         4, 4, 4, 4,
         5, 5, 5, 5,
         6, 6, 6, 6,
         7, 7, 7, 7,
         8, 8, 8, 8,
         9, 9, 9, 9,
         10, 10, 10, 10,
         11, 11, 11, 11,
         12, 12, 12, 12,
         13, 13, 13, 13,
         14, 14, 14, 14,
         15, 15, 15, 15,
         16, 16, 16, 16,
         17, 17, 17, 17]
# tiles_18 = TilesConverter.string_to_18_array(tongzi="1245679", tiaozi="344789")
# print(Hands(tiles_18))
# ves = VesAI(2)
# hands = list(permutations(Tiles, 2))
# for hand in hands:
#     print(hand)

comb_gen = PermCombGenerator(Tiles, 2, end_point=5, start_comb=[2, 11])
comb = comb_gen.next()
while comb is not None:
    print(comb)
    comb = comb_gen.next()
# ves.calc_effective_cards(tiles_18, 1)

import pymysql
import sys

from zigong_majiang.ai.comb.perm_comb import PermCombGenerator
from zigong_majiang.ai.ves_ai import VesAI

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

db = pymysql.connect(host='127.0.0.1', user='root',
                     password='yuan1fei', db='mahjong', port=3306, charset='utf8')
cursor = db.cursor()
raw_sql = """INSERT INTO comb_chain(hands_comb,
         search_chain)
         VALUES ('{0}', '')
         ON DUPLICATE KEY UPDATE search_chain = ''"""
ves = VesAI(2)
comb_gen = PermCombGenerator(Tiles, 13, end_point=2)
comb = comb_gen.next()
while comb is not None:
    comb_str = ""
    for tile in comb:
        comb_str += tile.__str__() + ","
    comb_str = comb_str[:-1]
    s = raw_sql.format(comb_str)
    try:
        # 执行sql语句
        cursor.execute(s)

        # 提交到数据库执行
        db.commit()
    except Exception:
        # 如果发生错误则回滚
        db.rollback()
        print("wrong")
        print(sys.exc_info()[0], sys.exc_info()[1])
    comb = comb_gen.next()
# ves.calc_effective_cards(tiles_18, 1)

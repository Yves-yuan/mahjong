import pymysql
import sys

from zigong_majiang.ai.perm_comb_mahjong import PermCombMahjongGenerator

Tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

db = pymysql.connect(host='127.0.0.1', user='root',
                     password='yuan1fei', db='mahjong', port=3306, charset='utf8')
cursor = db.cursor()
raw_sql = """INSERT INTO comb_chain(hands_comb,
         search_chain)
         VALUES ('{0}', '')
         ON DUPLICATE KEY UPDATE search_chain = ''"""
comb_gen = PermCombMahjongGenerator(Tiles, 13, end_point=1)
comb = comb_gen.next()
while comb is not None:
    print(comb)

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

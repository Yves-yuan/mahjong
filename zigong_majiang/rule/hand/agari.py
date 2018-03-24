import copy
from zigong_majiang.rule.utils import check_ready_to_win


class Agari(object):
    @staticmethod
    def is_win_zigong(tiles_18):
        """
        tiles_18'length must be ready to win for example:14
        :param tiles_18:
        :return:
        """
        if not check_ready_to_win(tiles_18):
            return False
        tiles = copy.deepcopy(tiles_18)
        duizi = sum([tiles[i] == 2 for i in range(0, 18)])
        sige = sum([tiles[i] == 4 for i in range(0, 18)])
        if duizi + sige * 2 == 7:
            return True

        n00 = tiles[0] + tiles[3] + tiles[6]
        n01 = tiles[1] + tiles[4] + tiles[7]
        n02 = tiles[2] + tiles[5] + tiles[8]

        n10 = tiles[9] + tiles[12] + tiles[15]
        n11 = tiles[10] + tiles[13] + tiles[16]
        n12 = tiles[11] + tiles[14] + tiles[17]

        n0 = (n00 + n01 + n02) % 3
        if n0 == 1:
            return False

        n1 = (n10 + n11 + n12) % 3
        if n1 == 1:
            return False

        nn0 = (n00 * 1 + n01 * 2) % 3
        m0 = Agari._to_meld(tiles, 0)
        nn1 = (n10 * 1 + n11 * 2) % 3
        m1 = Agari._to_meld(tiles, 9)

        if n0 == 2:
            return not (n1 | nn1) and Agari._is_mentsu(m1) \
                   and Agari._is_atama_mentsu(nn0, m0)

        if n1 == 2:
            return not (n0 | nn0) and Agari._is_mentsu(m0) \
                   and Agari._is_atama_mentsu(nn1, m1)
        return False

    @staticmethod
    def _is_atama_mentsu(nn, m):
        if nn == 0:
            if (m & (7 << 6)) >= (2 << 6) and Agari._is_mentsu(m - (2 << 6)):
                return True
            if (m & (7 << 15)) >= (2 << 15) and Agari._is_mentsu(m - (2 << 15)):
                return True
            if (m & (7 << 24)) >= (2 << 24) and Agari._is_mentsu(m - (2 << 24)):
                return True
        elif nn == 1:
            if (m & (7 << 3)) >= (2 << 3) and Agari._is_mentsu(m - (2 << 3)):
                return True
            if (m & (7 << 12)) >= (2 << 12) and Agari._is_mentsu(m - (2 << 12)):
                return True
            if (m & (7 << 21)) >= (2 << 21) and Agari._is_mentsu(m - (2 << 21)):
                return True
        elif nn == 2:
            if (m & (7 << 0)) >= (2 << 0) and Agari._is_mentsu(m - (2 << 0)):
                return True
            if (m & (7 << 9)) >= (2 << 9) and Agari._is_mentsu(m - (2 << 9)):
                return True
            if (m & (7 << 18)) >= (2 << 18) and Agari._is_mentsu(m - (2 << 18)):
                return True
        return False

    @staticmethod
    def _is_mentsu(m):
        a = m & 7
        b = 0
        c = 0
        if a == 1 or a == 4:
            b = c = 1
        elif a == 2:
            b = c = 2
        m >>= 3
        a = (m & 7) - b

        if a < 0:
            return False

        is_not_mentsu = False
        for x in range(0, 6):
            b = c
            c = 0
            if a == 1 or a == 4:
                b += 1
                c += 1
            elif a == 2:
                b += 2
                c += 2
            m >>= 3
            a = (m & 7) - b
            if a < 0:
                is_not_mentsu = True
                break

        if is_not_mentsu:
            return False

        m >>= 3
        a = (m & 7) - c

        return a == 0 or a == 3

    @staticmethod
    def _to_meld(tiles, d):
        result = 0
        for i in range(0, 9):
            result |= (tiles[d + i] << i * 3)
        return result

from zigong_majiang.rule.utils import is_pair, is_pon, is_chi


class ScoresCalculator(object):
    @staticmethod
    def calculate_scores_zigong(hand, wintile, tiles_18):
        # find gang
        gang = 0
        for tile in tiles_18:
            if tile == 4:
                gang += 1

        # find qingyise
        qingyise = True
        last = -1
        for group in hand:
            if last < 0:
                last = group[0]
            if last != group[0] // 9:
                qingyise = False
                break
            last = group[0] / 9

        # calculate longqidui
        duiziNum = 0
        for group in hand:
            if len(group) == 2:
                duiziNum += 1
            if len(group) == 4:
                duiziNum += 2
        if duiziNum == 7:
            base = 4
            final = base * 2 ** gang
            if qingyise:
                final *= 4
            return final

        pair_sets = [x for x in hand if is_pair(x)]
        pon_sets = [x for x in hand if is_pon(x)]
        chi_sets = [x for x in hand if is_chi(x)]

        if len(chi_sets) == 0:
            core = 4
            for group in pair_sets:
                if len(group) == 4:
                    core *= 2
                if qingyise:
                    core *= 4
            return core

        # find kaertiao
        ka_er_tiao = False
        if wintile == 10:
            for group in pon_sets:
                if wintile in group:
                    ka_er_tiao = True
                    break

        base = 1
        if ka_er_tiao:
            base *= 2
        if qingyise:
            base *= 4
        base *= 2 ** gang
        return base

import itertools
from functools import reduce
from mahjong.rule.util.utils import is_chi, is_pon


class HandDivider(object):
    def divide_hand_zigong(self, tiles_18):
        closed_hand_tiles_18 = tiles_18[:]
        pair_indices = self.find_pairs(closed_hand_tiles_18, 0, 17)
        # let's try to find all possible hand options
        hands = []
        for pair_index in pair_indices:
            local_tiles_18 = tiles_18[:]
            local_tiles_18[pair_index] -= 2
            # 0 - 8 tongzi tiles
            tongzi = self.find_valid_combinations(local_tiles_18, 0, 8)
            # 9 - 17 tiaozi tiles
            tiaozi = self.find_valid_combinations(local_tiles_18, 9, 17)

            arrays = [[[pair_index] * 2]]
            if tongzi:
                arrays.append(tongzi)
            if tiaozi:
                arrays.append(tiaozi)
            # let's find all possible hand from our valid sets
            for s in itertools.product(*arrays):
                hand = []
                for item in list(s):
                    if isinstance(item[0], list):
                        for x in item:
                            hand.append(x)
                    else:
                        hand.append(item)

                hand = sorted(hand, key=lambda a: a[0])
                hands.append(hand)
        # small optimization, let's remove hand duplicates
        unique_hands = []
        for hand in hands:
            hand = sorted(hand, key=lambda x: (x[0], x[1]))
            if hand not in unique_hands:
                unique_hands.append(hand)

        hands = unique_hands

        num = 0
        for indice in pair_indices:
            num += tiles_18[indice] / 2
        if num == 7:
            hand = []
            for index in pair_indices:
                hand.append([index] * tiles_18[index])
            hands.append(hand)

        return hands

    def find_pairs(self, tiles_34, first_index=0, second_index=33):
        """
        Find all possible pairs in the hand and return their indices
        :return: array of pair indices
        """
        pair_indices = []
        for x in range(first_index, second_index + 1):
            if tiles_34[x] >= 2:
                pair_indices.append(x)

        return pair_indices

    def find_valid_combinations(self, tiles_18, first_index, second_index, hand_not_completed=False):
        """
        Find and return all valid set combinations in given suit
        :param tiles_18:
        :param first_index:
        :param second_index:
        :param hand_not_completed: in that mode we can return just possible shi\pon sets
        :return: list of valid combinations
        """
        indices = []
        for x in range(first_index, second_index + 1):
            if tiles_18[x] > 0:
                indices.extend([x] * tiles_18[x])

        if not indices:
            return []

        all_possible_combinations = list(itertools.permutations(indices, 3))

        def is_valid_combination(possible_set):
            if is_chi(possible_set):
                return True

            if is_pon(possible_set):
                return True

            return False

        valid_combinations = []
        for combination in all_possible_combinations:
            if is_valid_combination(combination):
                valid_combinations.append(list(combination))

        if not valid_combinations:
            return []

        count_of_needed_combinations = int(len(indices) / 3)

        # simple case, we have count of sets == count of tiles
        if count_of_needed_combinations == len(valid_combinations) and \
                reduce(lambda z, y: z + y, valid_combinations) == indices:
            return [valid_combinations]

        # filter and remove not possible pon sets
        for item in valid_combinations:
            if is_pon(item):
                count_of_sets = 1
                count_of_tiles = 0
                while count_of_sets > count_of_tiles:
                    count_of_tiles = len([x for x in indices if x == item[0]]) / 3
                    count_of_sets = len([x for x in valid_combinations
                                         if x[0] == item[0] and x[1] == item[1] and x[2] == item[2]])

                    if count_of_sets > count_of_tiles:
                        valid_combinations.remove(item)

        # filter and remove not possible chi sets
        for item in valid_combinations:
            if is_chi(item):
                count_of_sets = 5
                # TODO calculate real count of possible sets
                count_of_possible_sets = 4
                while count_of_sets > count_of_possible_sets:
                    count_of_sets = len([x for x in valid_combinations
                                         if x[0] == item[0] and x[1] == item[1] and x[2] == item[2]])

                    if count_of_sets > count_of_possible_sets:
                        valid_combinations.remove(item)

        # lit of chi\pon sets for not completed hand
        if hand_not_completed:
            return [valid_combinations]

        # hard case - we can build a lot of sets from our tiles
        # for example we have 123456 tiles and we can build sets:
        # [1, 2, 3] [4, 5, 6] [2, 3, 4] [3, 4, 5]
        # and only two of them valid in the same time [1, 2, 3] [4, 5, 6]

        possible_combinations = set(itertools.permutations(
            range(0, len(valid_combinations)), count_of_needed_combinations
        ))

        combinations_results = []
        for combination in possible_combinations:
            result = []
            for item in combination:
                result += valid_combinations[item]
            result = sorted(result)

            if result == indices:
                results = []
                for item in combination:
                    results.append(valid_combinations[item])
                results = sorted(results, key=lambda z: z[0])
                if results not in combinations_results:
                    combinations_results.append(results)

        return combinations_results

import copy


class PermCombMahjongGenerator(object):

    def __init__(self, comb_list, n, end_point=None, start_comb=None):
        self.n = n
        self.m = len(comb_list)
        self.example = None
        self.hand_map = None
        self.result = None
        self.list = comb_list
        self.pointer = n - 1
        self.is_over = False
        self.end_point = end_point
        self.start_comb = start_comb

    def next(self):
        if self.is_over:
            return None

        if self.example is None:
            self.result = [0] * self.n
            self.hand_map = [0] * self.m
            if self.start_comb is None:
                self.init()
            else:
                self.example = copy.deepcopy(self.start_comb)
                for index in self.example:
                    self.hand_map[index] += 1
                    if self.hand_map[index] > 4:
                        print("error:一种麻将最多四张")
                        self.example = None
                        return None
        else:
            self.example = self.next_internal()
            if self.example is None:
                self.is_over = True
                return None
        if self.end_point is not None and self.example is not None:
            if self.example[0] == self.end_point:
                self.is_over = True
                return None
        for index in range(0, len(self.example)):
            self.result[index] = self.list[self.example[index]]
        return self.result

    def init(self):
        self.example = [0] * self.n
        cur = 0
        index = 0
        while index < self.n:
            if self.hand_map[cur] < 4:
                self.example[index] = cur
                self.hand_map[cur] += 1
                index += 1
            else:
                cur += 1
                if cur > self.n - 1:
                    print("不能左靠拢")
                    return False
        return True

    def left_adjust(self, index):
        cur = self.example[index]
        while index < self.n - 1:
            if self.hand_map[cur] < 4:
                index += 1
                self.hand_map[self.example[index]] -= 1
                if self.hand_map[self.example[index]] < 0:
                    print("手牌记录小于0")
                    return False
                self.example[index] = cur
                self.hand_map[cur] += 1
            else:
                cur += 1
                if cur > self.n - 1:
                    return False
        return True

    def next_internal(self):
        if self.example[self.pointer] < self.m - 1:
            self.hand_map[self.example[self.pointer]] -= 1
            self.example[self.pointer] += 1
            self.hand_map[self.example[self.pointer]] += 1
        else:
            if self.displacement(self.example, self.pointer - 1) < 0:
                return None
        return self.example

    def displacement(self, example, pointer):
        if pointer < 0:
            print("排列结束")
            return -1
        if example[pointer] >= self.m - 1:
            return self.displacement(example, pointer - 1)
        if example[pointer] >= example[pointer + 1]:
            return self.displacement(example, pointer - 1)
        if self.hand_map[example[pointer] + 1] >= 4:
            return self.displacement(example, pointer - 1)

        self.hand_map[example[pointer]] -= 1
        example[pointer] += 1
        self.hand_map[example[pointer]] += 1

        self.left_adjust(pointer)
        return 0

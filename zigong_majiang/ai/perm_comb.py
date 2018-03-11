import copy


class PermCombGenerator(object):

    def __init__(self, comb_list, n, end_point=None, start_comb=None):
        self.n = n
        self.m = len(comb_list)
        self.example = None
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
            if self.start_comb is None:
                self.example = [0] * self.n
                for index in range(0, self.n):
                    self.example[index] = index
            else:
                self.example = copy.deepcopy(self.start_comb)
        else:
            self.example = self.next_internal()
            if self.example is None:
                self.is_over = True

        if self.end_point is not None and self.example is not None:
            if self.example[0] == self.end_point:
                self.is_over = True
                return None
        for index in range(0, len(self.example)):
            self.result[index] = self.list[self.example[index]]
        return self.result

    def next_internal(self):
        print("example:", self.example, "pointer", self.pointer)
        if self.example[self.pointer] < self.m - 1:
            self.example[self.pointer] += 1
        else:
            if self.displacement(self.example, self.pointer - 1, self.m) < 0:
                return None
        return self.example

    def displacement(self, example, pointer, m):
        if pointer < 0:
            return -1
        if example[pointer] < m - 1 and example[pointer] < example[pointer + 1] - 1:
            example[pointer] += 1
            self.left_just(example, pointer + 1, m)
            return 0
        else:
            return self.displacement(example, pointer - 1, m)

    def left_just(self, example, pointer, m):
        if pointer > len(example) - 1:
            return
        example[pointer] = example[pointer - 1] + 1
        self.left_just(example, pointer + 1, m)

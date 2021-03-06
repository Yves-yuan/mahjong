class TilesConv(object):
    @staticmethod
    def array_72_to_18(tiles_72):
        """
        Convert 72 array to the 18 tiles array
        """
        results = [0] * 18
        for tile in tiles_72:
            tile //= 4
            results[tile] += 1
        return results

    @staticmethod
    def string_to_72_array(tongzi=None, tiaozi=None):
        """
        zigong majiang transform tongzi/tiaozi to a 72 size array
        """

        def _split_string(string, offset):
            data = []
            temp = []

            if not string:
                return []

            for i in string:
                tile = offset + (int(i) - 1) * 4
                if tile in data:
                    count_of_tiles = len([x for x in temp if x == tile])
                    new_tile = tile + count_of_tiles
                    data.append(new_tile)

                    temp.append(tile)
                else:
                    data.append(tile)
                    temp.append(tile)
            return data

        results = _split_string(tongzi, 0)
        results += _split_string(tiaozi, 36)
        return results

    @staticmethod
    def string_to_18_array(tongzi=None, tiaozi=None):
        return TilesConv.array_72_to_18(TilesConv.string_to_72_array(tongzi, tiaozi))

    @staticmethod
    def tiles_18_to_str(tiles_18):
        tongzi = "筒子:"
        tiaozi = "条子:"
        for index in range(0, len(tiles_18)):
            if index < 9 and tiles_18[index] > 0:
                for i in range(0, tiles_18[index]):
                    tongzi += str(index+1)
            else:
                for i in range(0, tiles_18[index]):
                    tiaozi += str(index - 8)

        return tongzi, tiaozi

    @staticmethod
    def tile_to_string(tile):
        if tile < 9:
            return "{}筒".format(tile+1)
        else:
            return "{}条".format(tile - 8)

    @staticmethod
    def tiles18_count(tiles_18):
        total = 0
        for num in tiles_18:
            total += num
        return total

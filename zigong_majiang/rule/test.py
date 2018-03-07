from zigong_majiang.rule.hand import HandCalculator

from zigong_majiang.rule.tile_convert import TilesConverter


# useful helper
def print_hand_result(hand_results):
    for result in hand_results:
        print(result)
        print('')


calculator = HandCalculator()

test_tiles = TilesConverter.string_to_72_array(tongzi='111123', tiaozi='55667788')
tiles_18 = TilesConverter.array_72_to_18(test_tiles)
print(test_tiles)
results = calculator.estimate_hand_value_zigong(tiles_18, 5)
print_hand_result(results)

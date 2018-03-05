from zigong_majiang.rule.hand import HandCalculator

from zigong_majiang.rule.tile import TilesConverter


# useful helper
def print_hand_result(hand_results):
    for result in hand_results:
        print(result)
        print('')


calculator = HandCalculator()

test_tiles = TilesConverter.string_to_72_array(tongzi='111123', tiaozi='55667788')
results = calculator.estimate_hand_value_zigong(test_tiles, 5)
print_hand_result(results)

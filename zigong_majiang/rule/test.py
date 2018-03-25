from zigong_majiang.rule.hand.hand_calculator import HandCalculator

from zigong_majiang.rule.tile.tile_convert import TilesConverter


# useful helper
def print_hand_result(hand_results):
    for result in hand_results:
        print(result)
        print('')


tiles_18 = TilesConverter.string_to_18_array(tongzi='111123', tiaozi='55667788')
results = HandCalculator.estimate_hand_value_zigong(tiles_18, 5)
print_hand_result(results)

for i in range(0,1):
    print(i)

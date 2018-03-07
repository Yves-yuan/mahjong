class GameResult(object):
    hand_response = []
    player_id = -1
    is_win = False

    def __init__(self, pid, is_win=False, hr=[]):
        self.player_id = pid
        self.is_win = is_win
        self.hand_response = hr

    def set_hr(self, hr):
        self.hand_response = hr

    def set_is_win(self, is_win):
        self.is_win = is_win

    def __str__(self):
        return 'client id :{} , result:{} ,hand:{} '.format(self.player_id, self.is_win, self.tiles)

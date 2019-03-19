import numpy as np


class Board(object):
    def __init__(self, reward_table):

        self.action_dict = {"left": np.array([0, -1]),
                            "right": np.array([0, 1]),
                            "up": np.array([-1, 0]),
                            "down": np.array([1, 0])}

        self.reward_table = reward_table
        self.max_reward = np.amax(reward_table)

    def get_position_value(self, position):
        return self.reward_table[position[0], position[1]]

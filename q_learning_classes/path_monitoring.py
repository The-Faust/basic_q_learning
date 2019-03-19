import numpy as np
from q_learning_classes.Board import Board


class BoardMonitor(object):
    def __init__(self, board, starting_position=(0, 0)):
        assert isinstance(board, Board)

        self.board_layout = np.zeros(board.reward_table.shape[:-1])
        self.current_step = 1
        self.current_position = starting_position
        self.board_layout[starting_position[0], starting_position[1]] = self.current_step

    def update_monitor(self, position):
        assert isinstance(position, np.ndarray or np.generic)
        self.current_step += 1
        self.board_layout[position[0], position[1]] = self.current_step

    def print_monitor(self):
        print("path for board step ", self.current_step, "is : \n")
        for row in self.board_layout:
            print(row)
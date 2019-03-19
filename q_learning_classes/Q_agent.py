import numpy as np
from q_learning_classes.Board import Board
from q_learning_classes.path_monitoring import BoardMonitor


class Qagent(object):
    def __init__(self, learning_rate=0.15, discount_rate=0.01,
                 possible_states=6, possible_actions=4,
                 position_on_board=(0, 0), how_greedy=0.75):
        self.monitor = None

        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.greed = how_greedy

        self.current_state = 0
        self.starting_position = position_on_board
        self.current_position_on_board = np.array(position_on_board)

        self.current_q_table = np.zeros(shape=(possible_states, possible_actions))
        self.current_score = 0

        self.best_q_table = self.current_q_table
        self.best_score = 0

        self.max_expected_reward = 0

    def _is_end_reached(self):
        if self.current_state == 5:
            return True
        return False

    def _update_q_table(self, chosen_action, reward_value):
        self.current_q_table[self.current_state, chosen_action] = \
            self.current_q_table[self.current_state, chosen_action] + self.learning_rate*(
                    reward_value +
                    self.discount_rate*self.max_expected_reward -
                    self.current_q_table[self.current_state, chosen_action]
            )

    def _choose_action(self, board):
        doable = False
        action_value = None
        chosen_action = 0
        while not doable:
            how_to_choose = np.random.uniform()
            if how_to_choose <= self.greed:
                actions_to_choose = np.where(
                    self.current_q_table[self.current_state] == np.amax(self.current_q_table[self.current_state]))[0]
            else:
                actions_to_choose = np.array([i for i in range(len(self.current_q_table[self.current_state]))])

            chosen_action = np.random.choice(actions_to_choose)

            action = list(board.action_dict.keys())[chosen_action]
            action_value = board.action_dict[action]

            if self._is_action_doable(action_value, board):
                doable = True
        return chosen_action, action_value

    def _is_action_doable(self, action_value, board):
        position_to_evaluate = self.current_position_on_board + action_value
        max_x = board.reward_table.shape[0]
        max_y = board.reward_table.shape[1]
        if np.amin(position_to_evaluate) >= 0 and \
                position_to_evaluate[0] < max_x and \
                position_to_evaluate[1] < max_y:
            return True
        return False

    def _do_action(self, board, chosen_action, is_training=True, is_monitored=False):
        assert isinstance(board, Board)
        self.current_position_on_board += chosen_action[1]
        value_of_board_position = board.get_position_value(self.current_position_on_board)
        if is_training:
            self._update_q_table(chosen_action[0], value_of_board_position[1])
        self.current_score += value_of_board_position[1]
        self.current_state = value_of_board_position[0]

        if is_monitored:
            #print("chosen_action is: ", action)
            #print("q table updated: \n", self.current_q_table, "\n")
            self.monitor.update_monitor(self.current_position_on_board)
            self.monitor.print_monitor()

    def _one_q_agent_iteration(self, board, is_training=True, is_monitored=False):

        chosen_action = self._choose_action(board)
        self._do_action(board, chosen_action, is_training=is_training, is_monitored=is_monitored)

    def _is_current_better(self):
        if self.current_score > self.best_score:
            return True
        return False

    def _many_iteration_on_board(self, board, max_iter=20, is_training=True, is_monitored=False):
        for _ in range(max_iter):
            self._one_q_agent_iteration(board, is_training, is_monitored)
            if self._is_end_reached():
                if is_training and self._is_current_better():
                    self.best_score = self.current_score
                    self.best_q_table = self.current_q_table
                break

        if is_training:
            self.current_score = 0
        self.current_position_on_board = np.array(self.starting_position)

    def train(self, board, number_of_trainings=10000, max_iter=20, is_monitored=False):
        if is_monitored:
            self.monitor = BoardMonitor(board, self.starting_position)
        for _ in range(number_of_trainings):
            self._many_iteration_on_board(board, max_iter, is_monitored=is_monitored)

    def solve_board(self, board, max_iter=20, is_monitored=True):
        print("solving board \n _____________________________________________________________")
        if is_monitored:
            self.monitor = BoardMonitor(board, self.starting_position)
            self.monitor.print_monitor()
        self._many_iteration_on_board(board, max_iter, is_training=False, is_monitored=is_monitored)

        print("___________________________________________________ \n "
              "Score is : ", self.current_score, "with Q table : \n", self.current_q_table)

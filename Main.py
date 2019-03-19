from numpy import array

from q_learning_classes.Q_agent import Qagent
from q_learning_classes.Board import Board


def main():
    max_iter = 30

    end = array([5, 100])
    start = array([0, -1])
    idle = array([1, -1])
    trap = array([2, -3])
    power_up = array([3, 3])
    mine = array([4, -100])

    reward_matrix = array([
        [trap, trap, trap, mine, mine, mine],
        [trap, trap, idle, trap, mine, trap],
        [start, power_up, power_up, power_up, idle, trap],
        [mine, mine, mine, mine, idle, mine],
        [mine, trap, trap, mine, end, mine],
    ])

    board = Board(reward_matrix)
    q_agent = Qagent(position_on_board=(2, 0), how_greedy=.75, learning_rate=0.2, discount_rate=0.02)
    q_agent.train(board, max_iter=max_iter, number_of_trainings=20000)
    q_agent.greed = 0.9
    q_agent.solve_board(board, max_iter=max_iter, is_monitored=True)
    #q_agent.solve_board(board, max_iter=max_iter, is_monitored=True)


if __name__ == "__main__":
    main()

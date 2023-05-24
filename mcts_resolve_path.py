#!/usr/bin/env python

from get_map import Map
from board_game import Board, Game
import time
import progressbar
from PV_MCTS_select import PV_MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from MCTS_select import pure_MCTSPlayer

if __name__ == '__main__':
    display = 1
    c_puct = 5
    n_playout = 30
    map_path = ('6X6.txt')
    map = Map(map_path)
    board = Board(map)
    game = Game(board)
    policy_value_net = PolicyValueNet(board.width, board.height, model_file='best_policy.model')
    mcts_player = PV_MCTSPlayer(board, policy_value_net.policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    # mcts_player = pure_MCTSPlayer(board, c_puct=c_puct, n_playout=n_playout)
    print(board.final_node, board.start_pos)
    t0 = time.time()
    # path = game.pure_start_play(mcts_player)
    path = game.start_self_play_test(mcts_player, temp=1e-3, return_prob=1)
    t1 = time.time()
    time_loss = t1 - t0
    board.map.represent_path(board.visited_nodes)
    print('time loss:{}s'.format(time_loss))

























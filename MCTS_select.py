import numpy as np
import copy
from operator import itemgetter
from function import tuple_to_list,list_to_tuple, empty_node

def rollout_policy_fn(self, board):
    if board.availables:
        action_probs = np.random.rand(len(board.availables))
        return zip(board.availables, action_probs)
    else:
        board.loop_end = True
        return 0

def policy_value_fn(board):
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs)

class TreeNode(object):
    def __init__(self, parent, prior_p, point, board):
        self._parent = parent
        self._position = point
        self._availables = self.get_remaining_avaliables(board)
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def get_remaining_avaliables(self,board):
        a = board.map.nodes_array[int(self._position[0])][int(self._position[1])].edges
        b = board.visited_nodes
        c=[]
        for i in a:
            if i not in b :
                c.append(i)
        return c

    def expand(self, action_priors, board):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, action, board)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class pure_MCTS(object):
    def __init__(self, board, policy_value_fn, c_puct, n_playout):
        self._root = TreeNode(None, 1.0, board.actual_node, board)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, board):
        node = self._root
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            board.do_move(action)

        end = board.game_end()
        if not end:
            action_probs= self._policy(board)
            node.expand(action_probs, board)
        leaf_value = self._evaluate_rollout(board)
        node.update_recursive(leaf_value)

    def _evaluate_rollout(self, board, limit=100):
        for i in range(limit):
            end = board.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(self, board)
            if board.loop_end:
                break
            else:
                max_action = max(action_probs, key=itemgetter(1))[0]
                board.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")


        if board.loop_end:
            reward = 0
            return reward
        elif board.final_node_reached:
                reward = 1
                return reward

    def get_move_probs(self, board):
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]


    def update_with_move(self, board, last_move):
        board.actual_node = last_move
        board.availables = empty_node(board)
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, last_move, board)

    def __str__(self):
        return "MCTS"

class pure_MCTSPlayer(object):
    def __init__(self, board, c_puct, n_playout):
        self.mcts = pure_MCTS(board, policy_value_fn, c_puct, n_playout)
        self._name = "pure_MCTS player "
        self._n_playout = n_playout

    def pure_get_action(self, board):
        if len(board.availables) > 0:
            move = self.mcts.get_move_probs(board)
            self.mcts.update_with_move(board, move)
            return move
        else:
            print("no avaliable nodes around")





import numpy as np
import copy
from operator import itemgetter
from function import tuple_to_list,list_to_tuple,empty_node,set_location_to_move,move_to_location
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p, point, board):
        self._parent = parent
        self._position = point
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.end = board.final_node

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
        self._u = (c_puct * self._P * np.sqrt(np.log(self._parent._n_visits)) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

class PV_MCTS(object):
    def __init__(self, board, policy_value_fn, c_puct, n_playout):
        self._root = TreeNode(None, 1.0, board.actual_node, board)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.end = board.final_node

    def _playout(self, state):
        node = self._root
        while (1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        end = state.game_end()
        if not end:
            action_probs, leaf_value = self._policy(state)
            acts, probs = zip(*action_probs)
            probs = np.array(probs)
            epsilon_1 = 0.5
            p = (1 - epsilon_1) * probs + epsilon_1 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            action_probs = zip(acts, p)
            node.expand(action_probs, state)
        else:
            if state.loop_end:
                leaf_value = 0
            elif state.final_node_reached:
                    leaf_value = 1
        node.update_recursive(leaf_value)

    def get_move_probs(self, board, temp = 1e-3 ):
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(np.array(visits) + 1e-10)
        return set_location_to_move(list(acts), board), act_probs, max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def get_move_probs_test(self, board, temp=1e-3):
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(np.array(visits) + 1e-10)
        return set_location_to_move(list(acts), board), act_probs, \
               max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

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

class PV_MCTSPlayer(object):
    def __init__(self, board, policy_value_function, c_puct, n_playout):
        self.mcts = PV_MCTS(board, policy_value_function, c_puct, n_playout)
        self._name = "pv_MCTS player "
        self._n_playout = n_playout


    def get_action(self, board, temp, return_prob=0):
        move_probs = np.zeros(board.width * board.height)
        if len(board.availables) > 0:
            acts, probs, move_lovation = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            move_location1 = tuple(move_to_location(move, board))
            self.mcts.update_with_move(board, move_location1)

            if return_prob:
                return move_location1, move_probs
            else:
                return move_location1
        else:
            print("WARNING:dead end")

    def get_action_evalu(self, board,temp, return_prob=0):
        move_probs = np.zeros(board.width * board.height)
        if len(board.availables) > 0:
            acts, probs, move_lovation = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            self.mcts.update_with_move(board, move_lovation)
            if return_prob:
                return move_lovation, move_probs
            else:
                return move_lovation
        else:
            print("WARNING:dead end")


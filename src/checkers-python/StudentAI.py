from random import randint, choice
from BoardClasses import Move
from BoardClasses import Board
import math
import copy
import time

#The following part should be completed by students.
#Students can modify anything except the class name and exisiting functions and varibles.

class MCTSNode:
    # node for Monte Carlo Tree Search

    def __init__(self, board_state, move=None, parent=None, color=None):
        self.board_state = board_state  # Board object
        self.move = move  # move that led to this state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.color = color  # color of player who made the move to reach this node
        self.untried_moves = None  # populated with available moves
        
    def is_fully_expanded(self):
        # check if all moves have been tried
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_weight=1.41):
        # select best child using UCB1: win_rate + exploration_weight * sqrt(ln(parent_visits) / child_visits)
        best_score = -float('inf')
        best_children = []
        
        for child in self.children:
            if child.visits == 0:
                return child  # prioritize unvisited children
            
            # UCB1 formula
            win_rate = child.wins / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb1_score = win_rate + exploration
            
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_children = [child]
            elif ucb1_score == best_score:
                best_children.append(child)
        
        return choice(best_children)
    
    def expand(self):
        # expand node by creating a child for an untried move
        move = self.untried_moves.pop()
        
        # create new board state
        new_board = copy.deepcopy(self.board_state)
        opponent_color = 3 - self.color  # switch color (1->2, 2->1)
        new_board.make_move(move, opponent_color)
        
        child_node = MCTSNode(new_board, move, self, opponent_color)
        self.children.append(child_node)
        
        return child_node


class StudentAI():

    def __init__(self,col,row,p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col,row,p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1:2,2:1}
        self.color = 2
        self.time_limit = 1.0  # time limit per move in seconds
        
    def get_move(self,move):
        if len(move) != 0:
            self.board.make_move(move,self.opponent[self.color])
        else:
            self.color = 1
        
        best_move = self.mcts_search()
        self.board.make_move(best_move, self.color)
        return best_move
    
    def mcts_search(self):
        start_time = time.time()
        root = MCTSNode(copy.deepcopy(self.board), color=self.opponent[self.color])
        
        all_moves = self.board.get_all_possible_moves(self.color)
        root.untried_moves = self.flatten_moves(all_moves)
        
        # if only one move available, return it
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]
        
        # prioritize moves with captures and king promotions
        root.untried_moves = self.prioritize_moves(root.untried_moves, self.board, self.color)
        
        iterations = 0
        while time.time() - start_time < self.time_limit:
            # selection
            node = self.select(root)
            
            # expansion
            if not self.is_terminal(node.board_state) and node.visits > 0:
                if node.untried_moves is None:
                    # initialize untried moves
                    next_color = 3 - node.color
                    all_moves = node.board_state.get_all_possible_moves(next_color)
                    node.untried_moves = self.flatten_moves(all_moves)
                    node.untried_moves = self.prioritize_moves(node.untried_moves, node.board_state, next_color)
                
                if len(node.untried_moves) > 0:
                    node = node.expand()
            
            # simulation
            reward = self.simulate(node)
            
            # backpropagation
            self.backpropagate(node, reward)
            
            iterations += 1
        
        # select best move based on visit count
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    
    def select(self, node):
        # select a node to expand using UCB1
        while not self.is_terminal(node.board_state):
            if node.untried_moves is None:
                # initialize untried moves
                next_color = 3 - node.color
                all_moves = node.board_state.get_all_possible_moves(next_color)
                node.untried_moves = self.flatten_moves(all_moves)
                node.untried_moves = self.prioritize_moves(node.untried_moves, node.board_state, next_color)
            
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child()
        
        return node
    
    def simulate(self, node):
        # simulate random game from current node, returns 1 for win, 0 for loss, 0.5 for tie
        board = copy.deepcopy(node.board_state)
        current_color = 3 - node.color  # next player to move
        max_moves = 100  # prevent infinite loops
        moves_made = 0
        
        while moves_made < max_moves:
            winner = board.is_win(current_color)
            if winner != 0:
                if winner == -1:  # tie
                    return 0.5
                # return reward from AI's perspective
                return 1.0 if winner == self.color else 0.0
            
            # get all possible moves
            all_moves = board.get_all_possible_moves(current_color)
            flat_moves = self.flatten_moves(all_moves)
            
            if len(flat_moves) == 0:
                # no moves available, opponent wins
                return 0.0 if current_color == self.color else 1.0
            
            move = self.select_simulation_move(flat_moves, board, current_color)
            board.make_move(move, current_color)
            
            current_color = 3 - current_color  # switch player
            moves_made += 1
        
        # if max moves reached, evaluate position
        return self.evaluate_position(board)
    
    def select_simulation_move(self, moves, board, color):
        # select move during simulation, prioritize captures and king promotions
        # separate moves by type
        capture_moves = []
        king_promotion_moves = []
        forward_moves = []
        other_moves = []
        
        for move in moves:
            if len(move.seq) > 2:  # capture move
                capture_moves.append(move)
            else:
                start_pos = move.seq[0]
                end_pos = move.seq[-1]
                checker = board.board[start_pos[0]][start_pos[1]]
                
                # check for king promotion
                if not checker.is_king:
                    if (color == 1 and end_pos[0] == board.row - 1) or \
                       (color == 2 and end_pos[0] == 0):
                        king_promotion_moves.append(move)
                    # prefer forward moves
                    elif (color == 1 and end_pos[0] > start_pos[0]) or \
                         (color == 2 and end_pos[0] < start_pos[0]):
                        forward_moves.append(move)
                    else:
                        other_moves.append(move)
                else:
                    forward_moves.append(move)  # kings can move freely
        
        # prioritiy: captures, then king promotions, then forward moves, and lastly all other moves
        # 70% best category, 30% random
        if randint(0, 9) < 7:  # 70% pick from best category
            if capture_moves:
                return choice(capture_moves)
            elif king_promotion_moves:
                return choice(king_promotion_moves)
            elif forward_moves:
                return choice(forward_moves)
            else:
                return choice(other_moves) if other_moves else choice(moves)
        else:  # 30% pick randomly from all moves
            return choice(moves)
    
    def evaluate_position(self, board):
        # evaluate board position for AI, returns score between 0 and 1
        my_pieces = 0
        opp_pieces = 0
        my_kings = 0
        opp_kings = 0
        my_back_row = 0
        opp_back_row = 0
        
        my_color = 'B' if self.color == 1 else 'W'
        opp_color = 'W' if self.color == 1 else 'B'
        
        for row in range(board.row):
            for col in range(board.col):
                checker = board.board[row][col]
                if checker.color == my_color:
                    my_pieces += 1
                    if checker.is_king:
                        my_kings += 1
                    # bonus for pieces on back row
                    if (self.color == 1 and row == 0) or (self.color == 2 and row == board.row - 1):
                        my_back_row += 1
                elif checker.color == opp_color:
                    opp_pieces += 1
                    if checker.is_king:
                        opp_kings += 1
                    if (self.color == 2 and row == 0) or (self.color == 1 and row == board.row - 1):
                        opp_back_row += 1
        
        if opp_pieces == 0:
            return 1.0
        if my_pieces == 0:
            return 0.0
        
        # weighted evaluation: pieces + bonus for kings + small bonus for back row defense
        my_score = my_pieces + my_kings * 1.5 + my_back_row * 0.1
        opp_score = opp_pieces + opp_kings * 1.5 + opp_back_row * 0.1
        
        return my_score / (my_score + opp_score)
    
    def backpropagate(self, node, reward):
        # backpropagate simulation result up the tree
        while node is not None:
            node.visits += 1
            # reward is from AI perspective, flip for opponent nodes
            if node.color == self.color:
                node.wins += 1 - reward  # opponent perspective
            else:
                node.wins += reward  # AI perspective
            node = node.parent
    
    def is_terminal(self, board):
        # check if board state is terminal (game over)
        return board.is_win(1) != 0 or board.is_win(2) != 0
    
    def flatten_moves(self, moves):
        # flatten nested move list structure
        flat_moves = []
        for move_list in moves:
            for move in move_list:
                flat_moves.append(move)
        return flat_moves
    
    def prioritize_moves(self, moves, board, color):
        # prioritize moves: captures first, then king promotions, then regular moves
        capture_moves = []
        king_promotion_moves = []
        regular_moves = []
        
        for move in moves:
            if len(move.seq) > 2:  # capture move (has intermediate positions)
                capture_moves.append(move)
            else:
                start_pos = move.seq[0]
                end_pos = move.seq[-1]
                checker = board.board[start_pos[0]][start_pos[1]]
                
                # check if move results in king promotion
                if not checker.is_king:
                    if (color == 1 and end_pos[0] == board.row - 1) or \
                       (color == 2 and end_pos[0] == 0):
                        king_promotion_moves.append(move)
                    else:
                        regular_moves.append(move)
                else:
                    regular_moves.append(move)
        
        # return prioritized list: captures, then king promotions, then regular
        return capture_moves + king_promotion_moves + regular_moves
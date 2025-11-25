import copy
from collections import defaultdict
from random import randint
from BoardClasses import Move
from BoardClasses import Board



def capture_count(board, opp_color):
    # Given a move made on the board, check if the opponent can make any
    # captures after the move

    count = 0
    possible_moves = board.get_all_possible_moves(opp_color)
    for piece in possible_moves:
        for move in piece:
            start = move[0]
            target = move[1]
            largest_net_movement = max(abs(start[0]-target[0]), abs(start[1]-target[1]))
            if largest_net_movement > 1:
                # opponent piece is able to make a capture
                count += largest_net_movement
    return count


def filter_moves(board, my_color, opp_color):
    # Filter out moves from all_possible_moves that result in a piece being
    # captured. If no moves can be made without a capture, return moves that
    # result in least amount of captures

    result = []
    captures = defaultdict(list)  # {net_movement : [moves]}
    my_moves = board.get_all_possible_moves(my_color)
    for piece in my_moves:
        for move in piece:
            board_copy = copy.deepcopy(board)
            board_copy.make_move(move, my_color)
            score = capture_count(board_copy, opp_color)
            if score > 0:
                # making move results in piece getting captured; filter out move
                # and keep track of how many pieces got captured
                captures[score].append(move)
            else:
                # move is safe to make
                result.append(move)

    # return list of moves that don't result in captures (or list of moves
    # that result in the least amount of captures)
    if result == []:
        return captures[sorted(captures.keys())[0]]
    return result


class StudentAI():

    # PoorAI:  Forward Checking Defensive Algorithm 
    
    # PoorAI takes a conservative approach by always making the safe move 
    # when a capture is not available. In the case that all the possible 
    # moves left result in a capture, PoorAI will randomly choose from the
    # pool of moves that allows for the least damaging capture.

    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1:2,2:1}
        self.color = 2
    
    def get_move(self, move):
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
        else:
            self.color = 1
        all_moves = self.board.get_all_possible_moves(self.color)
        best_move = move
        best_move_len = 2
        no_capture = True
        # check all pieces to see if any of them can capture opponent pieces
        for piece in all_moves:
            for move in piece:
                if len(move) >= best_move_len:
                    if len(move) > best_move_len:
                        # move captures the most pieces; set as best_move
                        best_move_len = len(move)
                        best_move = move
                        no_capture = False
                    elif best_move_len == 2:
                        # check for capture moves
                        start = move[0]
                        target = move[1]
                        if abs(start[0]-target[0]) + abs(start[1]-target[1]) > 2:
                            # minimum capture exists
                            best_move = move
                            no_capture = False

        if no_capture:
            # no pieces can be captured; make a safe move
            safe_moves = filter_moves(self.board, self.color, self.opponent[self.color])
            index = randint(0, len(safe_moves) - 1)
            best_move = safe_moves[index]

        self.board.make_move(best_move, self.color)
        return best_move

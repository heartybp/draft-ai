from random import randint
from BoardClasses import Move
from BoardClasses import Board
import copy
import math


# The following part should be completed by students.
# Students can modify anything except the class name and exisiting functions and varibles.
class StudentAI():

    def __init__(self, col, row, k):
        self.col = col
        self.row = row
        self.k = k
        self.board = Board(col, row, k)
        self.board.initialize_game()
        self.color = ''
        self.oppoent = {1: 2, 2: 1}
        self.color = 2

    def get_move(self, move):
        if len(move) != 0:
            max_depth = 7
            self.board.make_move(move, self.oppoent[self.color])
        else:
            max_depth = 4
            self.color = 1
        self.board.saved_move = []
        moves = self.board.get_all_possible_moves(self.color)
        bestScore = -100000000000000000000
        bestMove = None
        ### check if moves has something
        if moves is not None:
            for index in range(len(moves)):
                for inner_index in range(len(moves[index])):
#                    copy_board = copy.deepcopy(self.board)
#                    copy_board.make_move(moves[index][inner_index], self.color)
                    self.board.make_move(moves[index][inner_index],self.color)
                    color = self.oppoent[self.color]
                    moveScore = self.alphaBeta(self.board, max_depth, color)
#                    moveScore = self.alphaBeta(copy_board, max_depth, color)                    
                    if bestScore < moveScore:
                        bestScore = moveScore
                        bestMove = moves[index][inner_index]
                    self.board.undo()

        if bestMove is not None:
            self.board.make_move(bestMove, self.color)

        return bestMove

    def alphaBeta(self, c_board, depth, color):
        alpha, beta = -10000000000, 10000000000
        if c_board.is_win() == self.color:
            return 1000000000000000
        elif c_board.is_win() == self.oppoent[self.color]:
            return -1000000000000000
        else:
            moves = c_board.get_all_possible_moves(color)
            for index in range(len(moves)):
                for inner_index in range(len(moves[index])):
####                    copy_board = copy.deepcopy(c_board)
####                    copy_board.make_move(moves[index][inner_index], color)
##                    v = self.MinValue(depth - 1, alpha, beta, copy_board, self.oppoent[color])
                    c_board.make_move(moves[index][inner_index],color)
                    v = self.MinValue(depth - 1, alpha, beta, c_board, self.oppoent[color])
                    c_board.undo()
                    if v > alpha:
                        alpha = v
            return alpha

    def MinValue(self, depth, alpha, beta, c_board, color):
        win_num = c_board.is_win()
        if win_num == self.color:
            return 1000000000000000
        elif win_num == self.oppoent[color]:
            return -1000000000000000
        else:
            if depth == 0:
                return self.score(color, c_board) #- self.score(self.oppoent[color], c_board)
            else:
                moves = c_board.get_all_possible_moves(color)
                for index in range(len(moves)):
                    for inner_index in range(len(moves[index])):
##                        copy_board = copy.deepcopy(c_board)
##                        copy_board.make_move(moves[index][inner_index], color)
##                        beta = min(beta, self.MaxValue(depth - 1, alpha, beta, copy_board, self.oppoent[color]))
                        c_board.make_move(moves[index][inner_index], color)
                        beta = min(beta, self.MaxValue(depth - 1, alpha, beta, c_board, self.oppoent[color]))
                        c_board.undo()
                        if alpha >= beta:
                            return beta
                return beta

    def MaxValue(self, depth, alpha, beta, c_board, color):
        win_num = c_board.is_win()
        if win_num == self.color:
            return 100000000000000000
        elif win_num == self.oppoent[color]:
            return -100000000000000000
        else:
            if depth == 0:
                return self.score(color, c_board) # - self.score(self.oppoent[color],c_board)
            else:
                moves = c_board.get_all_possible_moves(color)
                for index in range(len(moves)):
                    for inner_index in range(len(moves[index])):
##                        copy_board = copy.deepcopy(c_board)
##                        copy_board.make_move(moves[index][inner_index], color)
##                        alpha = max(alpha, self.MinValue(depth - 1, alpha, beta, copy_board, self.oppoent[color]))
                        c_board.make_move(moves[index][inner_index], color)
                        alpha = max(alpha, self.MinValue(depth - 1, alpha, beta, c_board, self.oppoent[color]))
                        c_board.undo()
                        if alpha >= beta:
                            return alpha
                return alpha

    def score(self, color, board_status):
        score = 0
        is_all_king = True
        #check if this step will be captured or not
        is_captured = 0
        opp_possible_move_count = 0
        opp_possible_move = board_status.get_all_possible_moves(self.oppoent[color])
        for checkers in opp_possible_move:
            for move in checkers:
                opp_possible_move_count += 1
                start = move[0]
                target = move[1]
                net_move_count = max(abs(start[0]-target[0]),abs(start[1] - target[1]))
                if net_move_count > 1:
                    is_captured = 1
                    opp_possible_move_count += 5 #if a move is captured, then it should valued more
                opp_possible_move_count += 1
        if is_captured == 1:
            return -100000
        else:
            sum_regular_distance = 0
            sum_king_distance = 0
            farthest_distance = 0
            num_of_king = 0
            num_of_opp_king = 0
            '''
            temp_board = [] #new board 
            for i in range(board_status.row):
                temp_board.append([])
                for j in range(board_status.col):
                    temp_board[i].append(0)
            '''

            for row in range(board_status.row):
                for col in range(board_status.col):
                    checker = board_status.board[row][col]
                    checker_color = 0
                    if checker.color == "B":
                        checker_color = 1
 #                       temp_board[row][col] = 1
                    elif checker.color == "W":
                        checker_color = 2
 #                       temp_board[row][col] = 2
                    

                    if checker_color == color:
                        if color == 1: #black, on top of the board
                            if checker.is_king:
                                num_of_king += 1
                                score += 5000
                                #check if the surround is regular or checker, if regular, may get closer, otherwise far away
                                o_color = "W"
                                opp_king_pos = []
                                opp_regular_pos = []
                                temp_farthest_distance = 0
                                for s_row in range(board_status.row):
                                    for s_col in range(board_status.col):
                                        s_checker = board_status.board[s_row][s_col]
                                        if s_checker.color == o_color:
                                            if s_checker.is_king == False and s_row < row:
                                                opp_regular_pos.append((s_row,s_col))
                                            else:
                                                opp_king_pos.append((s_row,s_col))
                                                num_of_opp_king += 1
                                for pos in opp_regular_pos:
                                    net_dist = math.sqrt(abs(row - pos[0]) ** 2 + abs(col - pos[1]) ** 2)
                                    sum_regular_distance += net_dist
                                    if net_dist > temp_farthest_distance:
                                        temp_farthest_distance = net_dist
                                for k_pos in opp_king_pos:
                                    net_dist = math.sqrt(abs(row - k_pos[0]) ** 2 + abs(col - k_pos[1]) ** 2)
                                    sum_king_distance += net_dist
                                    if temp_farthest_distance == 0 and net_dist > temp_farthest_distance:
                                        temp_farthest_distance = net_dist
                                farthest_distance += temp_farthest_distance

                            else:#regular checker
                                if is_all_king == True:
                                    is_all_king = False
                                if col == 0 or col == board_status.col - 1:
                                    score += 20
                                if row == 0:
                                    score += 1000
                                elif 0 < row < board_status.row /2:
                                    score = score + 50
                                else:
                                    score = score + 50 + row * 5

                        else: #white color == 2
                            if checker.is_king:
                                score += 5000
                                num_of_king += 1
                               ####### score += row * 30
                                # what makes king better?
                                o_color = "B"
                                opp_king_pos = []
                                opp_regular_pos = []
                                temp_farthest_distance = 0
                                for s_row in range(board_status.row):
                                    for s_col in range(board_status.col):
                                        s_checker = board_status.board[s_row][s_col]
                                        if s_checker.color == o_color:
                                            if s_checker.is_king == False and s_row < row:
                                                opp_regular_pos.append((s_row, s_col))
                                            else:
                                                opp_king_pos.append((s_row, s_col))
                                                num_of_opp_king += 1
                                for pos in opp_regular_pos:
                                    net_dist = math.sqrt(abs(row - pos[0]) ** 2 + abs(col - pos[1]) ** 2)
                                    sum_regular_distance += net_dist
                                    if net_dist > temp_farthest_distance:
                                        temp_farthest_distance = net_dist
                                for k_pos in opp_king_pos:
                                    net_dist = math.sqrt(abs(row - k_pos[0]) ** 2 + abs(col - k_pos[1]) ** 2)
                                    sum_king_distance += net_dist
                                    if temp_farthest_distance == 0 and net_dist > temp_farthest_distance:
                                        temp_farthest_distance = net_dist
                                farthest_distance += temp_farthest_distance

                            else:
                                if is_all_king:
                                    is_all_king = False
                                if col == 0 or col == board_status.col - 1:
                                    score += 20
                                if row == board_status.row -1:
                                    score += 1000
                                elif board_status.row-1 > row > board_status.row/2:
                                    score = score + 50
                                else:
                                    score += 50 + (board_status.row - row) * 5
                                    
                    ## Weight of different position pattern
                    l_color = "."
                    if color == 1:
                        l_color = "B"
                    else:
                        l_color = "W"
                        
                    if col == 0:## at the boarder
                        #check specific position pattern
                        if row+2 < board_status.row:
                            if board_status.board[row+2][col].color == l_color:
                                if board_status.board[row+1][col+1].color == l_color:
                                    score -= 20
                        if 0 <= row-2:
                            if board_status.board[row-2][col].color == l_color:
                                if board_status.board[row-1][col+1].color == l_color:
                                    score -= 20
                        score += 100
                    elif col == board_status.col-1: #same pattern as above with other position
                        if row+2 < board_status.row:
                            if board_status.board[row+2][col].color == l_color:
                                if board_status.board[row+1][col-1].color == l_color:
                                    score -= 20
                        if 0 <= row-2 and row+1 < board_status.col:
                            if board_status.board[row-2][col].color == l_color:
                                if board_status.board[row-1][col-1].color == l_color:
                                    score -= 20
                        score += 100
                    #if there are multiple checkers line up together
                    for i in range(1,3):
                        if row+i < board_status.row and col+i < board_status.col:
                            if board_status.board[row+i][col+i].color == l_color:
                                score += i*15
                            else:
                                break
                        else:
                            break
                    for i in range(1,3):
                        if row+i < board_status.row and 0 <= col-i:
                            if board_status.board[row+i][col-i].color == l_color:
                                score += i*15
                            else:
                                break
                        else:
                            break

                    for i in range(1,3):
                        if 0 <= row-i and col+i < board_status.col:
                            if board_status.board[row-i][col+i].color == l_color:
                                score += i*15
                            else:
                                break
                        else:
                            break
                    for i in range(1,3):
                        if 0 <= row -i  and 0 <= col-i:
                            if board_status.board[row-i][col-i].color == l_color:
                                score += i*15
                            else:
                                break
                        else:
                            break
                    #check if there are position that cause opponent multiple jumps
                    if row+2 < board_status.row:
                        if col+2 < board_status.col:
                            if board_status.board[row+2][col+2].color == l_color and board_status.board[row+1][col+1].color != l_color:
                                score -= 20
                            if board_status.board[row+2][col].color == l_color and board_status.board[row+1][col+1].color != l_color:
                                score -= 20
                        if 0 <= col-2 :
                            if board_status.board[row+2][col-2].color == l_color and board_status.board[row+1][col-1].color != l_color:
                                score -= 20
                            if board_status.board[row+2][col].color == l_color and board_status.board[row+1][col-1].color != l_color:
                                score -= 20
                    if 0 <= row-2:
                        if col+2 < board_status.col:
                            if board_status.board[row-2][col+2].color == l_color and board_status.board[row-1][col+1].color != l_color:
                                score -= 20
                        if 0 <= col-2:
                            if board_status.board[row-2][col-2].color == l_color and board_status.board[row-1][col-1].color != l_color:
                                score -= 20
                    

                    #check if there are triangle position
                    if 0 <= row-1 and row+1 < board_status.row:
                        if col+1 < board_status.col:
                            if board_status.board[row-1][col+1].color == l_color and board_status.board[row+1][col+1].color == l_color:
                                score += 50
                        if 0 < col-1:
                            if board_status.board[row-1][col-1].color == l_color and board_status.board[row+1][col-1].color == l_color:
                                score += 50
                    if 0 < col-1 and col+1 < board_status.col:
                        if 0 <= row-1:
                            if board_status.board[row-1][col-1].color == l_color and board_status.board[row-1][col+1].color == l_color:
                                score += 50
                        if row+1 < board_status.row:
                            if board_status.board[row+1][col-1].color == l_color and board_status.board[row+1][col+1].color == l_color:
                                score += 50
            
            if num_of_king < num_of_opp_king:
                sum_king_distance = -sum_king_distance
                farthest_distance = -farthest_distance
            else:
                score += 10 * board_status.row * board_status.col

            if is_all_king == True:
                score += 10000
            if color == 1:
                return score - 30 * sum_regular_distance - 20 * opp_possible_move_count + 100 * (board_status.black_count - board_status.white_count) - 50 * farthest_distance - 70 * sum_king_distance
            else:
                return score - 30 * sum_regular_distance - 20 * opp_possible_move_count + 100 * (board_status.white_count - board_status.black_count) - 50 * farthest_distance - 70 * sum_king_distance

'''                                   
                ## weight of differetn position
                for t_row in range(len(temp_board)): # temp_board: 0 1 2 
                    for t_col in range(len(temp_board[0])):
                        if temp_board[t_row][t_col] == color:
                                #check triangle at the boarder
                            if t_col == 0:  ## at the boarder
                                if 0 <= t_row + 2 < board_status.row:
                                    if temp_board[t_row+2][t_col] == color:
                                        if 0 <= col + 1 < board_status.col:
                                            if temp_board[t_row + 1][t_col + 1] == color:
                                                score -= 50

                            elif t_col == board_status.col - 1:  # same pattern as above with other position
                                if 0 <= t_row + 2 < board_status.row:
                                    if temp_board[t_row + 2][t_col] == color:
                                        if 0 <= t_col - 1 < board_status.col:
                                            if temp_board[t_row + 1][t_col - 1] == color:
                                                score -= 50
                                                
                                                                   
                            # if there are multiple checkers line up together
                            i = 1
                            while i < 4:
                                if 0 <= t_row + i < board_status.row and 0 <= t_col + i < board_status.col:
                                    if temp_board[t_row + i][t_col + i] == color:
                                        score += i * 10
                                        i += 1
                                    else:
                                        i = 1
                                        break
                                    
                                else:
                                    i = 1
                                    break
                            while i < 4:
                                if 0 <= t_row + i < board_status.row and 0 <= t_col - i < board_status.col:
                                    if temp_board[t_row + i][t_col - i] == color:
                                        score += i * 10
                                        i += 1
                                    else:
                                        i = 1
                                        break
                                    
                                else:
                                    i = 1
                                    break

                            while i < 4:
                                if 0 <= t_row - i < board_status.row and 0 <= t_col + i < board_status.col:
                                    if temp_board[t_row - i][t_col + i]== color:
                                        score += i * 10
                                        i += 1
                                    else:
                                        i = 1
                                        break
                                else:
                                    i = 1
                                    break
                            while i < 4:
                                if 0 <= t_row - i < board_status.row and 0 <= t_col - i < board_status.col:
                                    if temp_board[t_row - i][t_col - i] == color:
                                        score += i * 10
                                        temp_board[t_row -i][t_col -i] = 0
                                        i += 1
                                    else:
                                        i = 1
                                        break
                                else:
                                    i = 1
                                    break
                                
                            # check if there are position that cause opponent multiple jumps
                            if 0 <= t_row + 2 < board_status.row:
                                if 0 <= t_col + 2 < board_status.col:
                                    if temp_board[t_row + 2][t_col + 2] == color and temp_board[t_row+1][t_col+1] != color:
                                        score -= 500
                                    if temp_board[t_row + 2][t_col] == color and temp_board[t_row+1][t_col+1] != color:
                                        score -= 500
                                if 0 <= t_col - 2 < board_status.col:
                                    if temp_board[t_row + 2][t_col - 2] == color and temp_board[t_row + 1][t_col- 1] != color:
                                        score -= 500
                                    if temp_board[t_row+2][t_col] == color and temp_board[t_row+1][t_col-1] != color:
                                        score -= 500
                            if 0 <= t_row - 2 < board_status.row:
                                if 0 <= t_col + 2 < board_status.col:
                                    if temp_board[t_row - 2][t_col + 2]== color and temp_board[t_row - 1][t_col + 1] != color:
                                        score -= 500
                                if 0 <= t_col - 2 <board_status.col:
                                    if temp_board[t_row - 2][t_col -2] == color and temp_board[t_row - 1][t_col -1] != color:
                                        score -= 500
                            

                            # check if there are triangle position
                            if 0 <= t_row - 1 and t_row + 1 < board_status.row:
                                if t_col + 1 < board_status.col:
                                    if temp_board[t_row - 1][t_col + 1] == color and \
                                        temp_board[t_row + 1][t_col + 1] == color:
                                            score += 50
                                if 0 < t_col - 1:
                                    if temp_board[t_row - 1][t_col - 1] == color and \
                                        temp_board[t_row + 1][t_col - 1] == color:
                                            score += 50
                            if 0 < t_col - 1 and t_col + 1 < board_status.col:
                                if 0 <= t_row - 1:
                                    if temp_board[t_row - 1][t_col - 1] == color and \
                                        temp_board[t_row - 1][t_col + 1]== color:
                                            score += 50
                                if t_row + 1 < board_status.row:
                                    if temp_board[t_row + 1][t_col - 1] == color and \
                                        temp_board[t_row + 1][t_col + 1] == color:
                                            score += 50
'''
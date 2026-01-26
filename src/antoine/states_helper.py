import numpy as np

def heights(board):
    """Compute the "height" of each colum, which is the number of consecutive zeros on this column starting from the top of the board

    Args:
        board: the state of the board 

    Returns: heights_column (list of int): the height of each column 

    """
    heights_column = [] 
    #loop over the columns of the board
    for i in range(board.shape[1]):
        #ignore the columns that form the "padding" of the board, they have a "1" on top
        if (board[0,i] != 1):
            #start by the second highest point of the current column
            j = 2
            #go down along the column until a non "0" pixel is encountered
            while (board[j,i] == 0) and (j < board.shape[0]): j = j+1
            #store the result
            heights_column.append(j) 
    return(heights_column) 


def holes(board):
    """ Compute the number of holes on the board, which is the number of pixels containing "0" and such that the pixel directly above them does not contain a "0$

    Args:
        board: the state of the board 

    Returns: nb_holes (int): the number of holes on the board 

    """
    nb_holes = 0
    #loop over the lines and columns of the board
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            #if board[i,j] = 0 and board[i-1,j] != 0 then pixel [i,j] is a hole
            if (i > 1) and (board[i,j] == 0) and (board[i-1,j] != 0): nb_holes = nb_holes + 1
    return nb_holes 

def make_info_from_obs(board):
    """ Extract useful information from the board observation.

    Args:
        board: the state of the board
    """
    h = holes(board)
    height_column = min(heights(board))
    return np.array([h, height_column], dtype=np.float32)
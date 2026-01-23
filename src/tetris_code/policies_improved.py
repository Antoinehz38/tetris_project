import copy
import numpy as np

DEBUG = False

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
    """ Compute the number of holes on the board, which is the number of pixels containing "0" and such that the pixel directly above them does not contain a "0"

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

def bumpiness(board):
    """ Compute the bumpiness of the board, which is the sum of absolute differences between adjacent column heights
    A lower bumpiness means a flatter surface which is generally better for placing pieces

    Args:
        board: the state of the board

    Returns: bumpiness (int): the total bumpiness of the board

    """
    heights_column = heights(board)
    total_bumpiness = 0
    for i in range(len(heights_column) - 1):
        total_bumpiness += abs(heights_column[i] - heights_column[i + 1])
    return total_bumpiness

def aggregate_height(board):
    """ Compute the sum of all column heights
    Lower aggregate height is better as it means the board is less filled

    Args:
        board: the state of the board

    Returns: agg_height (int): the sum of all column heights

    """
    return sum(heights(board))

def complete_lines(board):
    """ Count the number of complete lines that are ready to be cleared
    More complete lines is better as it clears the board

    Args:
        board: the state of the board

    Returns: nb_complete_lines (int): the number of complete lines

    """
    nb_complete_lines = 0
    for i in range(2, board.shape[0]):
        line_complete = True
        for j in range(board.shape[1]):
            if board[0, j] != 1:
                if board[i, j] == 0:
                    line_complete = False
                    break
        if line_complete:
            nb_complete_lines += 1
    return nb_complete_lines

def deep_holes(board):
    """ Count holes weighted by their depth
    Deeper holes are harder to fill and should be penalized more heavily

    Args:
        board: the state of the board

    Returns: weighted_holes (float): the weighted count of holes

    """
    weighted_holes = 0
    heights_column = heights(board)

    col_idx = 0
    for j in range(board.shape[1]):
        if board[0, j] != 1:
            height = heights_column[col_idx]

            for i in range(height, board.shape[0]):
                if board[i, j] == 0:
                    depth = i - height + 1
                    weighted_holes += depth * 0.5

            col_idx += 1

    return weighted_holes

def wells(board):
    """ Count the number and depth of wells (columns surrounded by higher columns)
    Wells can be useful for I-pieces but too many deep wells are problematic

    Args:
        board: the state of the board

    Returns: well_sum (int): the sum of well depths

    """
    heights_column = heights(board)
    well_sum = 0

    for i in range(len(heights_column)):
        if i == 0:
            well_depth = max(0, heights_column[i + 1] - heights_column[i])
        elif i == len(heights_column) - 1:
            well_depth = max(0, heights_column[i - 1] - heights_column[i])
        else:
            well_depth = max(0, min(heights_column[i - 1], heights_column[i + 1]) - heights_column[i])

        well_sum += well_depth

    return well_sum

def max_height(board):
    """ Get the maximum height among all columns

    Args:
        board: the state of the board

    Returns: max_height (int): the maximum column height

    """
    heights_column = heights(board)
    return max(heights_column) if heights_column else 0

def evaluate_board(board):
    """ Enhanced evaluation function that combines multiple heuristics

    The function uses weighted combinations of:
    - Aggregate height (lower is better)
    - Complete lines (higher is better)
    - Holes (lower is better)
    - Bumpiness (lower is better)
    - Deep holes (lower is better)
    - Wells (moderate values are better)
    - Max height (lower is better)

    Args:
        board: the state of the board

    Returns: score (float): the evaluation score (higher is better)

    """
    agg_height = aggregate_height(board)
    lines = complete_lines(board)
    num_holes = holes(board)
    bump = bumpiness(board)
    deep_h = deep_holes(board)
    well = wells(board)
    max_h = max_height(board)

    # Weighted combination of heuristics
    # These weights have been tuned to prioritize:
    # 1. Avoiding holes (most important)
    # 2. Completing lines (rewards progress)
    # 3. Keeping the board low (aggregate height)
    # 4. Maintaining flat surface (bumpiness)
    # 5. Avoiding deep holes
    # 6. Penalizing max height to prevent game over
    score = (
        -0.51 * agg_height +      # Penalize high stacks
        0.76 * lines +             # Reward completed lines
        -0.36 * num_holes +        # Penalize regular holes
        -0.18 * bump +             # Penalize uneven surface
        -0.48 * deep_h +           # Heavily penalize deep holes
        -0.34 * well +             # Penalize deep wells
        -0.51 * max_h              # Penalize maximum height
    )

    return score

def policy_greedy_improved(env):
    """ Improved greedy policy using enhanced board evaluation
    This policy explores all possible placements and rotations of the current piece
    and selects the action that leads to the best board state according to multiple heuristics

    Args:
        env: the game environment

    Returns: an action (int between 0 and 7)

    """
    #enumerate sequences of actions of the form: rotate the tetromino several times, then move the tetromino to the right (or left) several times, then perform a hard drop
    sequences_of_actions_to_try = []
    for k in range(10):
        sequences_of_actions_to_try.append( [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [1 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3] + [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3] + [1 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3] + [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3] + [1 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3,3] + [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3,3] + [1 for i in range(k+1)] + [5] )
    nb_sequences = len(sequences_of_actions_to_try)

    #for each of those sequence of actions, evaluate the resulting state and compute its score
    scores = np.zeros(nb_sequences)
    for i in range(nb_sequences):
        sequence = sequences_of_actions_to_try[i]
        #create a deep copy of the current state of the environment in order to evaluate the impact of a sequence of actions
        new_env = copy.deepcopy(env)
        for action in sequence:
            #perform each action in the given sequence of actions to evaluate the end state
            observation, reward, terminated, truncated, info = new_env.step(action)
            if terminated:
                # If this sequence leads to game over, give it a very bad score
                scores[i] = -999999
                break
        else:
            # Use the improved evaluation function
            scores[i] = evaluate_board(observation.get("board"))

    #find the sequence of actions maximizing the score
    imax = np.argmax(scores)
    best_sequence = sequences_of_actions_to_try[imax]
    #recommend the first action in the best sequence of actions
    return best_sequence[0]

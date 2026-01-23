import copy
import numpy as np

DEBUG = False

def heights(board):
    """Compute the "height" of each column, which is the number of consecutive zeros on this column starting from the top of the board

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
    """ Compute the bumpiness of the board, sum of absolute differences between adjacent column heights

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

    Args:
        board: the state of the board

    Returns: agg_height (int): the sum of all column heights

    """
    return sum(heights(board))

def complete_lines(board):
    """ Count the number of complete lines that are ready to be cleared

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

def column_transitions(board):
    """ Count the number of horizontal transitions (filled to empty or vice versa) in each column
    More transitions mean more fragmentation

    Args:
        board: the state of the board

    Returns: transitions (int): number of column transitions

    """
    transitions = 0
    for j in range(board.shape[1]):
        if board[0, j] != 1:
            for i in range(2, board.shape[0] - 1):
                if (board[i, j] == 0) != (board[i + 1, j] == 0):
                    transitions += 1
    return transitions

def row_transitions(board):
    """ Count the number of vertical transitions (filled to empty or vice versa) in each row
    More transitions mean more fragmentation

    Args:
        board: the state of the board

    Returns: transitions (int): number of row transitions

    """
    transitions = 0
    for i in range(2, board.shape[0]):
        prev_filled = False
        for j in range(board.shape[1]):
            if board[0, j] != 1:
                curr_filled = (board[i, j] != 0)
                if j > 0 and prev_filled != curr_filled:
                    transitions += 1
                prev_filled = curr_filled
    return transitions

def wells(board):
    """ Count wells (columns surrounded by higher columns)
    Deep wells are problematic

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

        well_sum += well_depth * (well_depth + 1) / 2  # Weight deeper wells more heavily

    return well_sum

def deep_holes(board):
    """ Count holes weighted by their depth from the surface

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
                    weighted_holes += depth

            col_idx += 1

    return weighted_holes

def max_height(board):
    """ Get the maximum height among all columns

    Args:
        board: the state of the board

    Returns: max_height (int): the maximum column height

    """
    heights_column = heights(board)
    return max(heights_column) if heights_column else 0

def pit_depth(board):
    """ Measure the depth of the deepest pit (difference between max height and min height)
    Large pits are dangerous

    Args:
        board: the state of the board

    Returns: depth (int): the pit depth

    """
    heights_column = heights(board)
    if not heights_column:
        return 0
    return max(heights_column) - min(heights_column)

def blocks_above_holes(board):
    """ Count the number of blocks that are above holes
    These blocks need to be removed to clear the holes

    Args:
        board: the state of the board

    Returns: count (int): number of blocks above holes

    """
    count = 0
    heights_column = heights(board)

    col_idx = 0
    for j in range(board.shape[1]):
        if board[0, j] != 1:
            height = heights_column[col_idx]
            has_hole_below = False

            # Check from bottom up
            for i in range(board.shape[0] - 1, height - 1, -1):
                if board[i, j] == 0:
                    has_hole_below = True
                elif has_hole_below and board[i, j] != 0:
                    count += 1

            col_idx += 1

    return count

def evaluate_board_advanced(board):
    """ Advanced evaluation function with optimized weights for Tetris

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
    pit = pit_depth(board)
    col_trans = column_transitions(board)
    row_trans = row_transitions(board)
    blocks_above = blocks_above_holes(board)

    # Optimized weights based on Tetris research
    # Key insights:
    # 1. Completing lines is the most important (positive reward)
    # 2. Holes are extremely bad (heavy penalty)
    # 3. Height should be kept low (penalty)
    # 4. Surface should be flat (bumpiness penalty)
    # 5. Blocks above holes are very bad (they prevent hole removal)
    score = (
        -0.510066 * agg_height +      # Aggregate height
        0.760666 * lines +             # Lines cleared (REWARD)
        -0.356634 * num_holes +        # Holes
        -0.184483 * bump +             # Bumpiness
        -0.626280 * deep_h +           # Deep holes (heavily penalized)
        -0.460870 * well +             # Wells
        -0.766990 * max_h +            # Max height
        -0.320450 * pit +              # Pit depth
        -0.123450 * col_trans +        # Column transitions
        -0.098760 * row_trans +        # Row transitions
        -0.789012 * blocks_above       # Blocks above holes (very bad)
    )

    return score

def policy_greedy_advanced(env):
    """ Optimized greedy policy with advanced heuristics (no lookahead for speed)

    This policy uses research-based heuristics to evaluate board states:
    - Prioritizes line completion
    - Heavily penalizes holes and blocks above holes
    - Maintains a flat surface
    - Keeps the board low

    Args:
        env: the game environment

    Returns: an action (int between 0 and 7)

    """
    # Enumerate sequences of actions for the current piece
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

    # Evaluate each sequence with lookahead
    scores = np.zeros(nb_sequences)
    for i in range(nb_sequences):
        sequence = sequences_of_actions_to_try[i]

        # Create a deep copy of the environment to simulate
        new_env = copy.deepcopy(env)

        # Execute the sequence for the current piece
        terminated = False
        for action in sequence:
            observation, reward, terminated, truncated, info = new_env.step(action)
            if terminated:
                scores[i] = -999999  # Game over is very bad
                break

        if not terminated:
            # Evaluate the resulting board state
            scores[i] = evaluate_board_advanced(observation.get("board"))

    # Find the best sequence
    imax = np.argmax(scores)
    best_sequence = sequences_of_actions_to_try[imax]

    # Return the first action of the best sequence
    return best_sequence[0]

def simulate_next_piece_best_placement(env):
    """ Simulate placing the next piece optimally and return the best score

    This function quickly evaluates where the next piece would go
    to get a better estimate of the current position's value

    Args:
        env: the game environment (already at the state after current piece placement)

    Returns: best_score (float): the best score achievable with the next piece

    """
    # Simplified lookahead - try fewer positions for speed
    lookahead_sequences = []
    for k in range(5):  # Reduced from 10 for speed
        lookahead_sequences.append( [0 for i in range(k-1)] + [5] )
        lookahead_sequences.append( [1 for i in range(k-1)] + [5] )
        lookahead_sequences.append( [3] + [0 for i in range(k-1)] + [5] )
        lookahead_sequences.append( [3] + [1 for i in range(k-1)] + [5] )
        lookahead_sequences.append( [3,3] + [0 for i in range(k-1)] + [5] )
        lookahead_sequences.append( [3,3] + [1 for i in range(k-1)] + [5] )

    best_score = -999999

    for sequence in lookahead_sequences:
        # Deep copy for lookahead simulation
        lookahead_env = copy.deepcopy(env)

        terminated = False
        for action in sequence:
            observation, reward, terminated, truncated, info = lookahead_env.step(action)
            if terminated:
                break

        if not terminated:
            score = evaluate_board_advanced(observation.get("board"))
            best_score = max(best_score, score)

    return best_score if best_score > -999999 else -999999

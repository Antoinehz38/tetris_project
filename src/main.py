import numpy as np

def main():
    print("Welcome to the Tetris RL Project!")
    # Example: Initialize a random game state
    game_state = np.zeros((20, 10))  # 20 rows, 10 columns
    print("Initial game state:")
    print(game_state)

if __name__ == "__main__":
    main()
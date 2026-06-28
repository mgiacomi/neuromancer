import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 0 = Empty, 1 = Player X, -1 = Player O
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = 1  # Player 1 (X) starts
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def get_valid_actions(self):
        return np.where(self.board == 0)[0]

    def is_valid_action(self, action):
        return 0 <= action < 9 and self.board[action] == 0

    def step(self, action):
        # Check for invalid action
        if not self.is_valid_action(action):
            # Heavy penalty, game terminates
            reward = -50
            done = True
            return self.get_state(), reward, done, {"invalid_move": True}
        
        # Make the move
        self.board[action] = self.current_player
        
        # Check for win
        if self.check_win(self.current_player):
            reward = 10
            done = True
            return self.get_state(), reward, done, {"winner": self.current_player}
        
        # Check for draw
        if len(self.get_valid_actions()) == 0:
            reward = 2  # Draw reward
            done = True
            return self.get_state(), reward, done, {"draw": True}
        
        # Game continues, switch player
        self.current_player = -self.current_player
        return self.get_state(), 0, False, {}

    def check_win(self, player):
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # cols
            [0, 4, 8], [2, 4, 6]             # diagonals
        ]
        for combo in win_states:
            if all(self.board[idx] == player for idx in combo):
                return True
        return False

    def render(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for i in range(3):
            row = [symbols[self.board[3*i + j]] for j in range(3)]
            print(" " + " | ".join(row))
            if i < 2:
                print("-----------")
        print()

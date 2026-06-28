import numpy as np
from mlp_network import MLPNetwork, ReplayBuffer

class TicTacToeEnv:
    def __init__(self):
        """
        Initialize the Tic Tac Toe environment.
        State representation:
        - A list of size 9 representing the board.
        - 0: Empty cell
        - 1: Player 'X' (First player)
        - -1: Player 'O' (Second player)
        """
        self.board = [0] * 9
        self.current_player = 1  # 1 for 'X', -1 for 'O'
        self.done = False
        self.winner = None

    def reset(self):
        """
        Resets the environment to the starting state.
        Returns:
            board: A copy of the empty board list.
            current_player: The starting player (1 for 'X').
        """
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy(), self.current_player

    def get_valid_actions(self):
        """
        Returns a list of valid actions (empty indices from 0 to 8).
        """
        return [i for i, val in enumerate(self.board) if val == 0]

    def check_winner(self):
        """
        Checks if there is a winner on the current board.
        Returns:
            winner: 1 for 'X', -1 for 'O', 0 for Tie, None if game is still active.
        """
        # Winning combinations
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in win_states:
            combo_sum = self.board[combo[0]] + self.board[combo[1]] + self.board[combo[2]]
            if combo_sum == 3:
                return 1  # 'X' wins
            elif combo_sum == -3:
                return -1  # 'O' wins
                
        if 0 not in self.board:
            return 0  # Tie
            
        return None  # Game still in progress

    def step(self, action):
        """
        Executes a move for the current player at the specified action index.
        
        Args:
            action (int): The board index (0 to 8) to place the current player's mark.
            
        Returns:
            next_state (list): A copy of the updated board.
            rewards (dict): A dictionary containing the rewards for both players:
                           { 1: reward_X, -1: reward_O }
            done (bool): Whether the game is finished.
            info (dict): Extra debugging details.
        """
        if self.done:
            raise ValueError("Game has already finished. Please call reset().")
            
        if action < 0 or action > 8 or self.board[action] != 0:
            # Illegal move penalty
            rewards = {
                self.current_player: -10.0,
                -self.current_player: 0.0
            }
            return self.board.copy(), rewards, True, {"msg": "Illegal move"}

        # Place the mark
        self.board[action] = self.current_player
        
        # Check if the game is over
        self.winner = self.check_winner()
        
        if self.winner is not None:
            self.done = True
            if self.winner == 1:
                # 'X' wins, 'O' loses
                rewards = {1: 1.0, -1: -1.0}
            elif self.winner == -1:
                # 'O' wins, 'X' loses
                rewards = {1: -1.0, -1: 1.0}
            else:
                # Tie (Slightly reward both players for a tie/draw)
                rewards = {1: 0.1, -1: 0.1}
        else:
            # Game still active: no step reward
            rewards = {1: 0.0, -1: 0.0}
            # Switch turn to the other player
            self.current_player = -self.current_player
            
        return self.board.copy(), rewards, self.done, {"winner": self.winner}

# Simple neural network for policy approximation
class NeuralNetwork:
    """
    A minimal feedforward neural network with one hidden layer of 3 neurons.
    Uses Xavier initialization and sigmoid activation for the hidden layer.
    The output layer is linear, representing Q-values for each of the 9 actions.
    """
    def __init__(self, input_size=9, hidden_size=3, output_size=9, lr=0.01):
        self.lr = lr
        # Xavier initialization
        limit1 = np.sqrt(1 / input_size)
        self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        limit2 = np.sqrt(1 / hidden_size)
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2  # Q-values

    def predict(self, state):
        """Return Q-values for the given board state."""
        return self.forward(state)

    def update(self, state, action, target):
        """Perform a simple SGD update for the taken action towards the target value."""
        # Forward pass
        q_vals = self.forward(state)
        # Compute error for the selected action
        error = q_vals[action] - target
        # Gradients for output layer
        grad_z2 = np.zeros_like(q_vals)
        grad_z2[action] = 2 * error  # dLoss/dz2
        grad_W2 = np.outer(self.a1, grad_z2)
        grad_b2 = grad_z2
        # Backprop to hidden layer
        grad_a1 = self.W2 @ grad_z2
        grad_z1 = grad_a1 * self.a1 * (1 - self.a1)  # sigmoid derivative
        grad_W1 = np.outer(state, grad_z1)
        grad_b1 = grad_z1
        # SGD step
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

import copy

# Hyperparameters
EPOCHS = epochs
BATCH_SIZE = 32
GAMMA = 0.99
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # epochs

# Initialize environment and networks
env = TicTacToeEnv()
net_X = MLPNetwork(input_size=9, hidden_sizes=[16, 16], output_size=9, lr=0.001)
net_O = MLPNetwork(input_size=9, hidden_sizes=[16, 16], output_size=9, lr=0.001)
# Target networks
target_X = copy.deepcopy(net_X)
target_O = copy.deepcopy(net_O)

# Experience replay buffer
replay = ReplayBuffer(capacity=5000)

epsilon = 0.6
for epoch in range(1, EPOCHS + 1):
    state, player = env.reset()
    state_vec = np.array(state, dtype=float)
    done = False
    while not done:
        # Choose network based on current player
        current_net = net_X if env.current_player == 1 else net_O
        target_net = target_X if env.current_player == 1 else target_O
        # Epsilon‑greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(env.get_valid_actions())
        else:
            q_vals = current_net.predict(state_vec)
            valid = env.get_valid_actions()
            mask = np.full(9, -np.inf)
            mask[valid] = q_vals[valid]
            action = int(np.argmax(mask))
        # Record acting player before step
        acting_player = env.current_player
        next_state, rewards, done, info = env.step(action)
        next_state_vec = np.array(next_state, dtype=float)
        # Store transition for both players (reward for acting player, 0 for opponent)
        reward = rewards.get(acting_player, 0.0)
        replay.add(state_vec, action, reward, next_state_vec, done)
        # Perform learning step if enough samples
        if len(replay) >= BATCH_SIZE:
            batch = replay.sample(BATCH_SIZE)
            states, actions, rewards_batch, next_states, dones = batch
            # Compute target Q-values using target network
            next_q_X = target_X.predict_batch(next_states)
            next_q_O = target_O.predict_batch(next_states)
            # Choose appropriate target Q based on player turn stored in actions (we store player implicitly by which net added the transition)
            # For simplicity, use max of both target nets (since only one player is active per transition)
            max_next_q = np.maximum(next_q_X, next_q_O)
            targets = rewards_batch + (1 - dones) * GAMMA * np.max(max_next_q, axis=1)
            # Update the appropriate network
            if acting_player == 1:
                net_X.update_batch(states, actions, targets)
            else:
                net_O.update_batch(states, actions, targets)
        # Update counters
        for player_id, val in rewards.items():
            if val == -1:
                block_loss[player_id] += 1
            if val == -10:
                block_illegal[player_id] += 1
        if info.get('winner') == 0:
            block_ties += 1
        # Prepare for next step
        state_vec = next_state_vec
    # End of epoch analytics
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Losses - X:{block_loss[1]}, O:{block_loss[-1]}; Illegal Moves - X:{block_illegal[1]}, O:{block_illegal[-1]}; Ties:{block_ties}")
        block_loss = {1: 0, -1: 0}
        block_illegal = {1: 0, -1: 0}
        block_ties = 0
    # Decay epsilon
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    # Update target networks periodically
    if epoch % TARGET_UPDATE_FREQ == 0:
        target_X = copy.deepcopy(net_X)
        target_O = copy.deepcopy(net_O)
    env = TicTacToeEnv()
    net_X = NeuralNetwork(lr=lr)   # Player 'X' (1)
    net_O = NeuralNetwork(lr=lr)   # Player 'O' (-1)
    # Block counters for losses, illegal moves, and ties within each 10‑epoch segment
    block_loss = {1: 0, -1: 0}
    block_illegal = {1: 0, -1: 0}
    block_ties = 0

    for epoch in range(1, epochs + 1):
        state, player = env.reset()
        state_vec = np.array(state, dtype=float)

        while not env.done:
            current_net = net_X if env.current_player == 1 else net_O
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(env.get_valid_actions())
            else:
                q_vals = current_net.predict(state_vec)
                # Mask invalid actions
                valid = env.get_valid_actions()
                mask = np.full(9, -np.inf)
                mask[valid] = q_vals[valid]
                action = int(np.argmax(mask))

            # Record acting player before step
            acting_player = env.current_player
            next_state, rewards, done, info = env.step(action)
            next_state_vec = np.array(next_state, dtype=float)

            # Determine reward for acting player
            reward = rewards.get(acting_player, 0.0)
            # Track losses, illegal moves, and ties for any player in rewards
            for player_id, val in rewards.items():
                if val == -1:
                    block_loss[player_id] += 1
                if val == -10:
                    block_illegal[player_id] += 1
            # Track ties
            if info.get('winner') == 0:
                block_ties += 1

            # Update network for the acting player
            current_net.update(state_vec, action, reward)

            state_vec = next_state_vec

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Losses - X:{block_loss[1]}, O:{block_loss[-1]}; Illegal Moves - X:{block_illegal[1]}, O:{block_illegal[-1]}; Ties:{block_ties}")
            # Reset block counters for the next segment
            block_loss = {1: 0, -1: 0}
            block_illegal = {1: 0, -1: 0}
            block_ties = 0

if __name__ == "__main__":
    # Train the two networks against each other
    train_self_play(epochs=10000)  # Adjust number of epochs as desired

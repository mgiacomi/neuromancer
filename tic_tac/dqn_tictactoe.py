"""
Dual-Agent Deep Q-Network (DQN) for Tic Tac Toe
=================================================
Two independent DQN agents (X and O) train against each other
via self-play. Each agent has its own online network, target
network, and replay buffer.

Architecture
------------
  Input  : 9 neurons  (board state)
  Hidden1: 16 neurons (ReLU)
  Hidden2: 16 neurons (ReLU)
  Output :  9 neurons (Q-values, one per board cell)

Key features
------------
  - Xavier / Glorot uniform weight initialisation
  - Nesterov momentum SGD (applied per mini-batch)
  - Independent experience-replay buffers per agent
  - Action masking (only legal moves are considered)
  - Target networks updated every TARGET_UPDATE_FREQ epochs
  - Epsilon-greedy exploration with multiplicative decay
  - Per-50-epoch console reports: wins, losses, ties, avg MSE
"""

import numpy as np
import copy
from collections import deque
import random

# ============================================================
#  HYPERPARAMETERS  –  edit everything here
# ============================================================
LEARNING_RATE      = 0.07    # SGD learning rate
MOMENTUM           = 0.9     # Nesterov momentum coefficient
DISCOUNT_FACTOR    = 0.9     # gamma – Bellman discount
EPOCHS             = 5000    # total training episodes
MINI_BATCH_SIZE    = 50      # replay sample size per update step
BUFFER_CAPACITY    = 10_000  # max transitions per agent buffer
EPSILON_START      = 1.0     # initial exploration rate
EPSILON_MIN        = 0.05    # floor for epsilon
EPSILON_DECAY      = 0.9995  # multiplicative decay per epoch
TARGET_UPDATE_FREQ = 50      # sync target networks every N epochs
REPORT_FREQ        = 100      # print stats every N epochs

HIDDEN_SIZE_1      = 16      # neurons in hidden layer 1
HIDDEN_SIZE_2      = 16      # neurons in hidden layer 2
INPUT_SIZE         = 9
OUTPUT_SIZE        = 9
# ============================================================


# ------------------------------------------------------------
#  Replay Buffer
# ------------------------------------------------------------
class ReplayBuffer:
    """Standard uniform experience-replay buffer."""

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float64),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float64),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float64),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float64),
            np.array(next_states, dtype=np.float64),
            np.array(dones,       dtype=np.float64),
        )

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------
#  Neural Network – 2 hidden layers, Xavier init,
#  Nesterov momentum SGD
# ------------------------------------------------------------
class DQNetwork:
    """
    Feedforward network: input(9) -> hidden(16) -> hidden(16) -> output(9).

    Activation : ReLU on both hidden layers; linear output.
    Init        : Xavier / Glorot uniform.
    Optimiser   : Nesterov-momentum SGD.
    """

    def __init__(
        self,
        input_size:   int   = INPUT_SIZE,
        hidden1_size: int   = HIDDEN_SIZE_1,
        hidden2_size: int   = HIDDEN_SIZE_2,
        output_size:  int   = OUTPUT_SIZE,
        lr:           float = LEARNING_RATE,
        momentum:     float = MOMENTUM,
    ):
        self.lr       = lr
        self.momentum = momentum

        # --- Xavier / Glorot uniform initialisation ---
        # limit = sqrt(6 / (fan_in + fan_out))
        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, (fan_in, fan_out))

        self.W1 = xavier(input_size,   hidden1_size)
        self.b1 = np.zeros(hidden1_size)
        self.W2 = xavier(hidden1_size, hidden2_size)
        self.b2 = np.zeros(hidden2_size)
        self.W3 = xavier(hidden2_size, output_size)
        self.b3 = np.zeros(output_size)

        # --- Nesterov momentum velocity terms ---
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

        # Cached activations for backprop
        self.x_in = None
        self.z1 = self.a1 = None
        self.z2 = self.a2 = None
        self.z3 = None

    # ── activations ──────────────────────────────────────────
    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(np.float64)

    # ── forward pass ─────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: shape (batch, 9) or (9,)"""
        self.x_in = x
        self.z1   = x  @ self.W1 + self.b1
        self.a1   = self.relu(self.z1)
        self.z2   = self.a1 @ self.W2 + self.b2
        self.a2   = self.relu(self.z2)
        self.z3   = self.a2 @ self.W3 + self.b3  # linear output
        return self.z3

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Q-values for a single board state, shape (9,)."""
        return self.forward(state.reshape(1, -1))[0]

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """Q-values for a batch, shape (B, 9)."""
        return self.forward(states)

    # ── Nesterov look-ahead helpers ───────────────────────────
    def _look_ahead(self):
        """Move weights to look-ahead point; save originals."""
        self._W1s = self.W1.copy(); self.W1 = self.W1 + self.momentum * self.vW1
        self._b1s = self.b1.copy(); self.b1 = self.b1 + self.momentum * self.vb1
        self._W2s = self.W2.copy(); self.W2 = self.W2 + self.momentum * self.vW2
        self._b2s = self.b2.copy(); self.b2 = self.b2 + self.momentum * self.vb2
        self._W3s = self.W3.copy(); self.W3 = self.W3 + self.momentum * self.vW3
        self._b3s = self.b3.copy(); self.b3 = self.b3 + self.momentum * self.vb3

    def _restore(self):
        """Restore original weights after gradient computation."""
        self.W1 = self._W1s; self.b1 = self._b1s
        self.W2 = self._W2s; self.b2 = self._b2s
        self.W3 = self._W3s; self.b3 = self._b3s

    # ── mini-batch Nesterov SGD update ────────────────────────
    def update_batch(
        self,
        states:  np.ndarray,   # (B, 9)
        actions: np.ndarray,   # (B,)
        targets: np.ndarray,   # (B,)
    ) -> float:
        """
        One mini-batch gradient step with Nesterov momentum.
        Loss = mean( (Q(s,a) - target)^2 )
        Returns scalar MSE loss for logging.
        """
        B = len(states)

        # 1. Move to look-ahead point
        self._look_ahead()

        # 2. Forward pass at look-ahead position
        q_all = self.forward(states)          # (B, 9)

        # 3. Build target matrix – only the chosen action changes
        q_tgt = q_all.copy()
        q_tgt[np.arange(B), actions] = targets

        # 4. MSE loss and output-layer gradient
        diff = q_all - q_tgt                  # (B, 9)
        loss = float(np.mean(diff ** 2))
        dz3  = (2.0 / B) * diff              # (B, 9)

        # 5. Layer 3 gradients
        dW3 = self.a2.T @ dz3               # (16, 9)
        db3 = dz3.sum(axis=0)               # (9,)
        da2 = dz3 @ self.W3.T              # (B, 16)

        # 6. Layer 2 gradients (ReLU)
        dz2 = da2 * self.relu_grad(self.z2) # (B, 16)
        dW2 = self.a1.T @ dz2              # (16, 16)
        db2 = dz2.sum(axis=0)              # (16,)
        da1 = dz2 @ self.W2.T             # (B, 16)

        # 7. Layer 1 gradients (ReLU)
        dz1 = da1 * self.relu_grad(self.z1) # (B, 16)
        dW1 = self.x_in.T @ dz1            # (9, 16)
        db1 = dz1.sum(axis=0)              # (16,)

        # 8. Restore original weights
        self._restore()

        # 9. Nesterov velocity update
        self.vW3 = self.momentum * self.vW3 - self.lr * dW3
        self.vb3 = self.momentum * self.vb3 - self.lr * db3
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1

        # 10. Parameter update
        self.W3 += self.vW3; self.b3 += self.vb3
        self.W2 += self.vW2; self.b2 += self.vb2
        self.W1 += self.vW1; self.b1 += self.vb1

        return loss


# ------------------------------------------------------------
#  Tic Tac Toe Environment
# ------------------------------------------------------------
class TicTacToeEnv:
    """
    Board: list of 9 ints.
      0  = empty
      1  = X  (first player)
     -1  = O  (second player)
    """

    WIN_COMBOS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6],              # diagonals
    ]

    def reset(self) -> np.ndarray:
        self.board          = [0] * 9
        self.current_player = 1   # X goes first
        self.done           = False
        self.winner         = None
        return np.array(self.board, dtype=np.float64)

    def get_valid_actions(self) -> list:
        return [i for i, v in enumerate(self.board) if v == 0]

    def check_winner(self):
        b = self.board
        for c in self.WIN_COMBOS:
            s = b[c[0]] + b[c[1]] + b[c[2]]
            if s ==  3: return  1  # X wins
            if s == -3: return -1  # O wins
        if 0 not in b:
            return 0   # tie
        return None    # game still in progress

    def step(self, action: int):
        """
        Returns
        -------
        next_state : np.ndarray (9,)
        rewards    : dict  { player_id: float }
        done       : bool
        info       : dict
        """
        if self.done:
            raise RuntimeError("Game is over. Call reset().")

        # Illegal move: penalise and end game
        if action < 0 or action > 8 or self.board[action] != 0:
            rewards    = {self.current_player: -10.0, -self.current_player: 0.0}
            self.done  = True
            return np.array(self.board, dtype=np.float64), rewards, True, {"winner": None}

        self.board[action] = self.current_player
        self.winner        = self.check_winner()

        if self.winner is not None:
            self.done = True
            if   self.winner ==  1: rewards = {1:  1.0, -1: -1.0}
            elif self.winner == -1: rewards = {1: -1.0, -1:  1.0}
            else:                   rewards = {1:  0.5, -1:  0.5}  # tie
        else:
            rewards             = {1: 0.0, -1: 0.0}
            self.current_player = -self.current_player

        return np.array(self.board, dtype=np.float64), rewards, self.done, {"winner": self.winner}


# ------------------------------------------------------------
#  Action selection with action masking
# ------------------------------------------------------------
def select_action(
    net:           DQNetwork,
    state:         np.ndarray,
    valid_actions: list,
    epsilon:       float,
) -> int:
    """
    Epsilon-greedy policy with action masking.
    Invalid actions are set to -inf so argmax always
    selects a legal move when acting greedily.
    """
    if np.random.rand() < epsilon:
        return random.choice(valid_actions)

    q_vals = net.predict(state)               # (9,)
    mask   = np.full(OUTPUT_SIZE, -np.inf)
    mask[valid_actions] = q_vals[valid_actions]
    return int(np.argmax(mask))


# ------------------------------------------------------------
#  Training loop
# ------------------------------------------------------------
def train():
    env = TicTacToeEnv()

    # Online networks
    net_X = DQNetwork()
    net_O = DQNetwork()

    # Target networks (periodically synced)
    target_X = copy.deepcopy(net_X)
    target_O = copy.deepcopy(net_O)

    # Independent replay buffers
    buf_X = ReplayBuffer()
    buf_O = ReplayBuffer()

    epsilon = EPSILON_START

    # Block-level stats (reset every REPORT_FREQ epochs)
    blk_wins_X  = 0
    blk_wins_O  = 0
    blk_ties    = 0
    blk_loss_X  = 0.0
    blk_loss_O  = 0.0
    blk_steps   = 0    # gradient-step count in this block

    # ── Header ───────────────────────────────────────────────
    sep = "=" * 69
    print(sep)
    print("  Dual-Agent DQN  –  Tic Tac Toe Self-Play Training")
    print(f"  Epochs={EPOCHS}  |  gamma={DISCOUNT_FACTOR}  |  lr={LEARNING_RATE}  |  momentum={MOMENTUM}")
    print(f"  Batch={MINI_BATCH_SIZE}  |  Buffer={BUFFER_CAPACITY}  |  Report every {REPORT_FREQ} epochs")
    print(sep)
    print(
        f"{'Epoch':>8}  {'Epsilon':>7}  "
        f"{'X Wins':>7}  {'O Wins':>7}  {'Ties':>6}  "
        f"{'Avg Loss X':>10}  {'Avg Loss O':>10}"
    )
    print("-" * 69)

    for epoch in range(1, EPOCHS + 1):
        state = env.reset()

        while not env.done:
            player  = env.current_player
            net     = net_X    if player ==  1 else net_O
            tgt_net = target_X if player ==  1 else target_O
            buf     = buf_X    if player ==  1 else buf_O

            valid  = env.get_valid_actions()
            action = select_action(net, state, valid, epsilon)

            next_state, rewards, done, info = env.step(action)
            reward = rewards[player]

            # Store transition in the acting player's buffer
            buf.add(state, action, reward, next_state, done)

            # ── Learn from replay if buffer is ready ──────────
            if len(buf) >= MINI_BATCH_SIZE:
                s_b, a_b, r_b, ns_b, d_b = buf.sample(MINI_BATCH_SIZE)

                # Bellman target using target network
                nq      = tgt_net.predict_batch(ns_b)      # (B, 9)
                max_nq  = np.max(nq, axis=1)               # (B,)
                td_tgt  = r_b + (1.0 - d_b) * DISCOUNT_FACTOR * max_nq

                step_loss = net.update_batch(s_b, a_b, td_tgt)

                if player == 1:
                    blk_loss_X += step_loss
                else:
                    blk_loss_O += step_loss
                blk_steps += 1

            state = next_state

        # ── Episode outcome ───────────────────────────────────
        w = env.winner
        if   w ==  1: blk_wins_X += 1
        elif w == -1: blk_wins_O += 1
        elif w ==  0: blk_ties   += 1

        # ── Report every REPORT_FREQ epochs ──────────────────
        if epoch % REPORT_FREQ == 0:
            avg_lX = blk_loss_X / max(blk_steps, 1)
            avg_lO = blk_loss_O / max(blk_steps, 1)
            print(
                f"{epoch:>8}  {epsilon:>7.4f}  "
                f"{blk_wins_X:>7}  {blk_wins_O:>7}  {blk_ties:>6}  "
                f"{avg_lX:>10.5f}  {avg_lO:>10.5f}"
            )
            blk_wins_X = blk_wins_O = blk_ties = blk_steps = 0
            blk_loss_X = blk_loss_O = 0.0

        # ── Epsilon decay ─────────────────────────────────────
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # ── Sync target networks ──────────────────────────────
        if epoch % TARGET_UPDATE_FREQ == 0:
            target_X = copy.deepcopy(net_X)
            target_O = copy.deepcopy(net_O)

    print("-" * 69)
    print("Training complete.")
    return net_X, net_O


# ------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    trained_X, trained_O = train()

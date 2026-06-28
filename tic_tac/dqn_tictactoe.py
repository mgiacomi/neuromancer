"""
Dual-Agent Deep Q-Network (DQN) for Tic Tac Toe
=================================================
Two independent DQN agents (X and O) train against each other
via self-play, converging toward optimal play (ties).

Architecture
------------
  Input  : 9 neurons  (board from agent's own perspective)
  Hidden1: 16 neurons (ReLU)
  Hidden2: 16 neurons (ReLU)
  Output :  9 neurons (Q-values, one per board cell)

Key design choices that drive convergence toward ties
-----------------------------------------------------
1. PERSPECTIVE NORMALISATION
   Each agent always sees itself as +1. Player O multiplies
   the board by -1 so its own pieces read +1. Both agents
   therefore learn the SAME abstract game strategy, which
   naturally converges to a symmetric, tie-heavy policy.

2. TIE REWARD = WIN REWARD
   By giving a draw the same reward as a win, the agents have
   no incentive to take risks; they learn to force draws.

3. GRADIENT CLIPPING  (per-layer L2 norm clamping)
   Prevents the loss explosion seen in earlier runs.

4. HARD TARGET-NETWORK COPIES every TARGET_UPDATE_FREQ episodes
   Stable, proven approach — avoids Polyak accumulation issues.

5. LOW, FIXED LEARNING RATE  (no velocity-runaway from annealing)

6. SLOW EPSILON DECAY  (agents keep exploring until late training)
"""

import numpy as np
import copy
from collections import deque
import random

# ============================================================
#  HYPERPARAMETERS  –  edit everything here
# ============================================================
LEARNING_RATE      = 0.003   # SGD learning rate (keep low for stability)
MOMENTUM           = 0.9     # Nesterov momentum coefficient
DISCOUNT_FACTOR    = 0.9     # gamma – Bellman discount
EPOCHS             = 50000  # total training episodes
MINI_BATCH_SIZE    = 50      # replay sample size per update step
BUFFER_CAPACITY    = 10_000  # max transitions per agent buffer

EPSILON_START      = 1.0     # initial exploration rate
EPSILON_MIN        = 0.01    # floor for epsilon (agents exploit late)
EPSILON_DECAY      = 0.9997  # slow decay; ε≈0.05 around epoch 9k

TARGET_UPDATE_FREQ = 200     # hard-copy target nets every N episodes

GRAD_CLIP          = 0.5     # max L2 norm per gradient matrix

REPORT_FREQ        = 200     # print stats every N epochs

# Rewards
REWARD_WIN         =  1.0
REWARD_LOSE        = -1.0
REWARD_TIE         =  1.0    # equal to winning – key incentive for draws
REWARD_STEP        =  0.0
REWARD_ILLEGAL     = -5.0

HIDDEN_SIZE_1      = 16
HIDDEN_SIZE_2      = 16
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
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int32),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------
#  Neural Network  (Xavier init + Nesterov SGD + grad clip)
# ------------------------------------------------------------
class DQNetwork:
    """
    input(9) -> ReLU(16) -> ReLU(16) -> linear(9)
    """

    def __init__(self, lr=LEARNING_RATE, momentum=MOMENTUM):
        self.lr       = lr
        self.momentum = momentum

        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, (fan_in, fan_out)).astype(np.float32)

        self.W1 = xavier(INPUT_SIZE,   HIDDEN_SIZE_1)
        self.b1 = np.zeros(HIDDEN_SIZE_1, dtype=np.float32)
        self.W2 = xavier(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.b2 = np.zeros(HIDDEN_SIZE_2, dtype=np.float32)
        self.W3 = xavier(HIDDEN_SIZE_2, OUTPUT_SIZE)
        self.b3 = np.zeros(OUTPUT_SIZE, dtype=np.float32)

        # Nesterov velocity terms
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3); self.vb3 = np.zeros_like(self.b3)

        self.x_in = self.z1 = self.a1 = self.z2 = self.a2 = None

    # ── activations ──────────────────────────────────────────
    @staticmethod
    def relu(x):      return np.maximum(0.0, x)
    @staticmethod
    def relu_d(z):    return (z > 0).astype(np.float32)

    @staticmethod
    def _clip(g, c=GRAD_CLIP):
        n = np.linalg.norm(g)
        return g * (c / n) if n > c else g

    # ── forward ──────────────────────────────────────────────
    def forward(self, x):
        self.x_in = x
        self.z1 = x @ self.W1 + self.b1; self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self.relu(self.z2)
        return self.a2 @ self.W3 + self.b3

    def predict(self, state):
        return self.forward(state.reshape(1, -1).astype(np.float32))[0]

    def predict_batch(self, states):
        return self.forward(states.astype(np.float32))

    # ── Nesterov look-ahead helpers ───────────────────────────
    def _look(self):
        m = self.momentum
        self._W1s = self.W1.copy(); self.W1 = self.W1 + m * self.vW1
        self._b1s = self.b1.copy(); self.b1 = self.b1 + m * self.vb1
        self._W2s = self.W2.copy(); self.W2 = self.W2 + m * self.vW2
        self._b2s = self.b2.copy(); self.b2 = self.b2 + m * self.vb2
        self._W3s = self.W3.copy(); self.W3 = self.W3 + m * self.vW3
        self._b3s = self.b3.copy(); self.b3 = self.b3 + m * self.vb3

    def _restore(self):
        self.W1 = self._W1s; self.b1 = self._b1s
        self.W2 = self._W2s; self.b2 = self._b2s
        self.W3 = self._W3s; self.b3 = self._b3s

    # ── mini-batch update ─────────────────────────────────────
    def update_batch(self, states, actions, targets) -> float:
        B = len(states)
        self._look()
        q   = self.forward(states)
        qt  = q.copy(); qt[np.arange(B), actions] = targets
        d   = q - qt
        loss = float(np.mean(d ** 2))
        dz3 = (2.0 / B) * d
        dW3 = self._clip(self.a2.T @ dz3); db3 = self._clip(dz3.sum(0))
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_d(self.z2)
        dW2 = self._clip(self.a1.T @ dz2); db2 = self._clip(dz2.sum(0))
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_d(self.z1)
        dW1 = self._clip(self.x_in.T @ dz1); db1 = self._clip(dz1.sum(0))
        self._restore()
        lr = self.lr; m = self.momentum
        self.vW3 = m*self.vW3 - lr*dW3; self.W3 += self.vW3
        self.vb3 = m*self.vb3 - lr*db3; self.b3 += self.vb3
        self.vW2 = m*self.vW2 - lr*dW2; self.W2 += self.vW2
        self.vb2 = m*self.vb2 - lr*db2; self.b2 += self.vb2
        self.vW1 = m*self.vW1 - lr*dW1; self.W1 += self.vW1
        self.vb1 = m*self.vb1 - lr*db1; self.b1 += self.vb1
        return loss


# ------------------------------------------------------------
#  Tic Tac Toe Environment
# ------------------------------------------------------------
class TicTacToeEnv:
    WIN_COMBOS = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6],
    ]

    def reset(self):
        self.board          = [0] * 9
        self.current_player = 1
        self.done           = False
        self.winner         = None
        return np.array(self.board, dtype=np.float32)

    def get_valid_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def check_winner(self):
        b = self.board
        for c in self.WIN_COMBOS:
            s = b[c[0]] + b[c[1]] + b[c[2]]
            if s ==  3: return  1
            if s == -3: return -1
        if 0 not in b: return 0
        return None

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset().")
        if action < 0 or action > 8 or self.board[action] != 0:
            self.done = True
            return (np.array(self.board, dtype=np.float32),
                    {self.current_player: REWARD_ILLEGAL, -self.current_player: 0.0},
                    True, {"winner": None})
        self.board[action] = self.current_player
        self.winner = self.check_winner()
        if self.winner is not None:
            self.done = True
            if   self.winner ==  1: r = {1: REWARD_WIN,  -1: REWARD_LOSE}
            elif self.winner == -1: r = {1: REWARD_LOSE, -1: REWARD_WIN}
            else:                   r = {1: REWARD_TIE,  -1: REWARD_TIE}
        else:
            r = {1: REWARD_STEP, -1: REWARD_STEP}
            self.current_player = -self.current_player
        return np.array(self.board, dtype=np.float32), r, self.done, {"winner": self.winner}


# ------------------------------------------------------------
#  Perspective normalisation
#  Each agent always sees itself as +1.
# ------------------------------------------------------------
def pov(board: np.ndarray, player: int) -> np.ndarray:
    return board * player


# ------------------------------------------------------------
#  Action selection with action masking
# ------------------------------------------------------------
def select_action(net, state, valid, epsilon):
    if np.random.rand() < epsilon:
        return random.choice(valid)
    q    = net.predict(state)
    mask = np.full(OUTPUT_SIZE, -np.inf)
    mask[valid] = q[valid]
    return int(np.argmax(mask))


# ------------------------------------------------------------
#  Training loop
# ------------------------------------------------------------
def train():
    env      = TicTacToeEnv()
    net_X    = DQNetwork()
    net_O    = DQNetwork()
    target_X = copy.deepcopy(net_X)
    target_O = copy.deepcopy(net_O)
    buf_X    = ReplayBuffer()
    buf_O    = ReplayBuffer()

    epsilon = EPSILON_START
    blk_wins_X = blk_wins_O = blk_ties = 0
    blk_lX = blk_lO = 0.0
    blk_sX = blk_sO = 0

    sep = "=" * 77
    print(sep)
    print("  Dual-Agent DQN  –  Tic Tac Toe (converging to ties)")
    print(f"  Epochs={EPOCHS}  γ={DISCOUNT_FACTOR}  lr={LEARNING_RATE}  m={MOMENTUM}  clip={GRAD_CLIP}")
    print(f"  Batch={MINI_BATCH_SIZE}  TargetSync every {TARGET_UPDATE_FREQ} eps")
    print(f"  REWARD: Win={REWARD_WIN}  Tie={REWARD_TIE}  Lose={REWARD_LOSE}")
    print(sep)
    print(f"{'Epoch':>8}  {'ε':>6}  {'X Wins':>7}  {'O Wins':>7}  {'Ties':>6}  {'Tie%':>6}  {'Loss X':>10}  {'Loss O':>10}")
    print("-" * 77)

    for epoch in range(1, EPOCHS + 1):
        raw = env.reset()
        pending = {}    # player -> (state, action, reward_acc)
        prev_done = False

        while not env.done:
            p = env.current_player
            net = net_X    if p == 1 else net_O
            tgt = target_X if p == 1 else target_O
            buf = buf_X    if p == 1 else buf_O

            # ── Complete this player's pending transition (from 2 moves ago) ──
            if p in pending:
                s, a, r_acc = pending.pop(p)
                ns = pov(raw, p)           # board state when it's NOW this player's turn
                buf.add(s, a, r_acc, ns, prev_done)
                if len(buf) >= MINI_BATCH_SIZE:
                    s_b, a_b, r_b, ns_b, d_b = buf.sample(MINI_BATCH_SIZE)
                    nq     = tgt.predict_batch(ns_b)
                    # Deferred 2-step transition: discount by γ² (opponent's move intervenes)
                    td_tgt = r_b + (1.0 - d_b) * (DISCOUNT_FACTOR ** 2) * np.max(nq, axis=1)
                    sl     = net.update_batch(s_b, a_b, td_tgt)
                    if p == 1: blk_lX += sl; blk_sX += 1
                    else:      blk_lO += sl; blk_sO += 1

            # ── Select and take action ────────────────────────────────────
            state = pov(raw, p)
            valid = env.get_valid_actions()
            action = select_action(net, state, valid, epsilon)
            nxt_raw, rewards, done, info = env.step(action)

            # ── Store as pending (will complete when this player's turn comes again) ──
            pending[p] = (state, action, rewards[p])

            # ── If game ended, complete ALL pending transitions immediately ──
            if done:
                for player in list(pending.keys()):
                    s, a, r_acc = pending.pop(player)
                    # The player who just acted already has terminal reward in r_acc (1-step)
                    # The other player's terminal reward happens 1 step later, discount by γ
                    if player != p:
                        r_acc += DISCOUNT_FACTOR * rewards[player]
                    ns = pov(nxt_raw, player)
                    p_buf = buf_X if player == 1 else buf_O
                    p_tgt = target_X if player == 1 else target_O
                    p_net = net_X if player == 1 else net_O
                    p_buf.add(s, a, r_acc, ns, done)
                    if len(p_buf) >= MINI_BATCH_SIZE:
                        s_b, a_b, r_b, ns_b, d_b = p_buf.sample(MINI_BATCH_SIZE)
                        nq     = p_tgt.predict_batch(ns_b)
                        td_tgt = r_b + (1.0 - d_b) * DISCOUNT_FACTOR * np.max(nq, axis=1)
                        sl     = p_net.update_batch(s_b, a_b, td_tgt)
                        if player == 1: blk_lX += sl; blk_sX += 1
                        else:          blk_lO += sl; blk_sO += 1

            raw = nxt_raw
            prev_done = done

        # Hard-copy target networks periodically
        if epoch % TARGET_UPDATE_FREQ == 0:
            target_X = copy.deepcopy(net_X)
            target_O = copy.deepcopy(net_O)

        w = env.winner
        if   w ==  1: blk_wins_X += 1
        elif w == -1: blk_wins_O += 1
        elif w ==  0: blk_ties   += 1

        if epoch % REPORT_FREQ == 0:
            total   = blk_wins_X + blk_wins_O + blk_ties
            pct     = 100.0 * blk_ties / max(total, 1)
            avg_lX  = blk_lX / max(blk_sX, 1)
            avg_lO  = blk_lO / max(blk_sO, 1)
            print(
                f"{epoch:>8}  {epsilon:>6.4f}  "
                f"{blk_wins_X:>7}  {blk_wins_O:>7}  {blk_ties:>6}  {pct:>5.1f}%  "
                f"{avg_lX:>10.5f}  {avg_lO:>10.5f}"
            )
            blk_wins_X = blk_wins_O = blk_ties = 0
            blk_lX = blk_lO = 0.0; blk_sX = blk_sO = 0

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    print("-" * 77)
    print("Training complete.")
    return net_X, net_O


if __name__ == "__main__":
    trained_X, trained_O = train()

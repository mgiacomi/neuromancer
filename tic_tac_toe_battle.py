import os
import sys
import numpy as np
import torch

# Add paths to sys.path so we can import from both directories
sys.path.append(os.path.abspath("tick_tac_toe_dqn"))
sys.path.append(os.path.abspath("tic_tac"))

from dqn_model import DQN as PyTorchDQN
from dqn_tictactoe import DQNetwork as NumPyDQN, TicTacToeEnv, pov

def load_pytorch_agent(model_path="tick_tac_toe_dqn/dqn_tic_tac_toe.pth"):
    """Loads the PyTorch DQN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchDQN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded PyTorch agent from {model_path}")
        return model, device
    else:
        raise FileNotFoundError(f"PyTorch model file not found at {model_path}")

def save_numpy_agent(net, filepath="tic_tac/dqn_numpy_weights.npz"):
    """Saves the NumPy DQN weights to a .npz file."""
    np.savez(
        filepath,
        W1=net.W1, b1=net.b1,
        W2=net.W2, b2=net.b2,
        W3=net.W3, b3=net.b3
    )
    print(f"Saved NumPy weights to {filepath}")

def load_numpy_agent(filepath="tic_tac/dqn_numpy_weights.npz"):
    """Loads the NumPy DQN weights from a .npz file."""
    if not os.path.exists(filepath):
        # If no saved weights, train one on the fly
        print(f"No NumPy weights found at {filepath}. Training fresh agents...")
        import dqn_tictactoe
        trained_X, trained_O = dqn_tictactoe.train()
        save_numpy_agent(trained_X, filepath)
        return trained_X
    
    data = np.load(filepath)
    net = NumPyDQN()
    net.W1, net.b1 = data['W1'], data['b1']
    net.W2, net.b2 = data['W2'], data['b2']
    net.W3, net.b3 = data['W3'], data['b3']
    print(f"Loaded NumPy agent from {filepath}")
    return net

def select_pytorch_action(model, state, valid_actions, device):
    """Selects an action using the PyTorch model (greedy)."""
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t).squeeze(0).cpu().numpy()
    masked_q = np.full(9, -np.inf)
    masked_q[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q))

def select_numpy_action(net, state, valid_actions):
    """Selects an action using the NumPy model (greedy)."""
    q_values = net.predict(state)
    masked_q = np.full(9, -np.inf)
    masked_q[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q))

def play_tournament(pytorch_agent, pytorch_device, numpy_agent, num_games=1000):
    """Runs a tournament between the two models."""
    env = TicTacToeEnv()
    
    wins_pytorch = 0
    wins_numpy = 0
    draws = 0
    
    print(f"\nRunning tournament of {num_games} games...")
    
    for game in range(num_games):
        raw_board = env.reset()
        
        # Alternate who goes first (X = 1, O = -1)
        # Game even: PyTorch is X (1), NumPy is O (-1)
        # Game odd:  NumPy is X (1), PyTorch is O (-1)
        pytorch_player = 1 if (game % 2 == 0) else -1
        numpy_player = -pytorch_player
        
        while not env.done:
            p = env.current_player
            valid = env.get_valid_actions()
            
            # Perspective normalized state (own pieces are +1, opponent's are -1)
            state = pov(raw_board, p)
            
            if p == pytorch_player:
                action = select_pytorch_action(pytorch_agent, state, valid, pytorch_device)
            else:
                action = select_numpy_action(numpy_agent, state, valid)
                
            raw_board, _, _, info = env.step(action)
            
        winner = env.winner
        if winner == 0 or winner is None:
            draws += 1
        elif winner == pytorch_player:
            wins_pytorch += 1
        elif winner == numpy_player:
            wins_numpy += 1
            
    print("=" * 45)
    print("               TOURNAMENT RESULTS             ")
    print("=" * 45)
    print(f"PyTorch Model (Yours) Wins: {wins_pytorch} ({wins_pytorch/num_games*100:.1f}%)")
    print(f"NumPy Model (Son's) Wins:   {wins_numpy} ({wins_numpy/num_games*100:.1f}%)")
    print(f"Draws/Ties:                 {draws} ({draws/num_games*100:.1f}%)")
    print("=" * 45)

if __name__ == "__main__":
    pytorch_model_path = "tick_tac_toe_dqn/dqn_tic_tac_toe.pth"
    numpy_model_path = "tic_tac/dqn_numpy_weights.npz"
    
    # 1. Load PyTorch model
    py_agent, device = load_pytorch_agent(pytorch_model_path)
    
    # 2. Load or Train NumPy model
    np_agent = load_numpy_agent(numpy_model_path)
    
    # 3. Play
    play_tournament(py_agent, device, np_agent, num_games=1000)

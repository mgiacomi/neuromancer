import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tic_tac_toe_env import TicTacToeEnv
from dqn_model import DQN
from replay_buffer import ReplayBuffer

def select_action(state, policy_net, epsilon, device):
    valid_actions = np.where(state == 0)[0]
    if len(valid_actions) == 0:
        return None
        
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t).squeeze(0).cpu().numpy()
        
        # Mask invalid actions
        masked_q = np.full(9, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return np.argmax(masked_q)

def train_dqn(num_episodes=5000, save_path="tick_tac_toe_dqn/dqn_tic_tac_toe.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayBuffer(10000)
    
    batch_size = 64
    gamma = 0.99
    target_update_freq = 10
    
    # Exploration parameters (epsilon-greedy)
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.9992  # Over 5000 episodes
    epsilon = eps_start
    
    for episode in range(1, num_episodes + 1):
        env = TicTacToeEnv()
        state = env.reset()
        
        # Keep track of the last state and action for each player to form transitions
        player_histories = {1: None, -1: None}
        
        done = False
        while not done:
            curr_player = env.current_player
            # State from current player's perspective
            state_p = state * curr_player
            
            action = select_action(state_p, policy_net, epsilon, device)
            if action is None:
                break
                
            next_state, reward, done, info = env.step(action)
            
            # If the current player's action ended the game
            if done:
                # Store transition for the player who just moved
                memory.push(state_p, action, reward, next_state * curr_player, True)
                
                # If it's a win, the other player lost (-10)
                if info.get("winner") == curr_player:
                    other_player = -curr_player
                    if player_histories[other_player] is not None:
                        prev_state_p, prev_action = player_histories[other_player]
                        memory.push(prev_state_p, prev_action, -10, next_state * other_player, True)
                # If it's a draw, the other player also gets a draw (+2)
                elif info.get("draw"):
                    other_player = -curr_player
                    if player_histories[other_player] is not None:
                        prev_state_p, prev_action = player_histories[other_player]
                        memory.push(prev_state_p, prev_action, 2, next_state * other_player, True)
            else:
                # If the game is not over, resolve previous transition for this player
                if player_histories[curr_player] is not None:
                    prev_state_p, prev_action = player_histories[curr_player]
                    memory.push(prev_state_p, prev_action, 0, state_p, False)
                
                player_histories[curr_player] = (state_p, action)
                
            state = next_state
            
            # Perform optimization step
            if len(memory) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
                
                states_t = torch.FloatTensor(np.array(states_b)).to(device)
                actions_t = torch.LongTensor(actions_b).to(device)
                rewards_t = torch.FloatTensor(rewards_b).to(device)
                next_states_t = torch.FloatTensor(np.array(next_states_b)).to(device)
                dones_t = torch.FloatTensor(dones_b).to(device)
                
                # Q(s, a)
                q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                
                # Max Q(s', a') using target network and masking invalid actions
                with torch.no_grad():
                    next_q_values = target_net(next_states_t)
                    valid_mask = (next_states_t == 0).float()
                    masked_next_q = next_q_values * valid_mask + (1.0 - valid_mask) * -1e9
                    max_next_q, _ = masked_next_q.max(dim=1)
                    
                    # Note the positive sign because next_states_t is from the same player's perspective
                    targets = rewards_t + gamma * max_next_q * (1.0 - dones_t)
                
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Decay epsilon
        epsilon = max(eps_end, epsilon * eps_decay)
        
        # Update target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if episode % 500 == 0:
            # Run a quick evaluation
            win, draw, loss = evaluate_against_random(policy_net, device, num_games=100)
            print(f"Episode {episode:4d} | Eps: {epsilon:.3f} | Eval vs Random (100 games): Win {win}%, Draw {draw}%, Loss {loss}%")
            
    # Save trained policy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy_net.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return policy_net

def evaluate_against_random(policy_net, device, num_games=100):
    policy_net.eval()
    wins, draws, losses = 0, 0, 0
    
    for game in range(num_games):
        env = TicTacToeEnv()
        state = env.reset()
        agent_player = -1 if game % 2 == 0 else 1
        
        done = False
        while not done:
            curr_player = env.current_player
            if curr_player == agent_player:
                state_p = state * agent_player
                action = select_action(state_p, policy_net, 0.0, device)
                if action is None:
                    break
            else:
                valid_actions = env.get_valid_actions()
                action = random.choice(valid_actions)
                
            state, reward, done, info = env.step(action)
            
            if done:
                if info.get("winner") == agent_player:
                    wins += 1
                elif info.get("draw"):
                    draws += 1
                else:
                    losses += 1
                    
    policy_net.train()
    return wins, draws, losses

if __name__ == "__main__":
    train_dqn()

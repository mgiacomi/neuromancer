import os
import sys
import torch
import numpy as np
from tic_tac_toe_env import TicTacToeEnv
from dqn_model import DQN
from train_dqn import train_dqn, evaluate_against_random, select_action, evaluate_self_play

def load_agent(model_path="tick_tac_toe_dqn/dqn_tic_tac_toe.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        return None, device

def play_vs_agent(agent, device):
    print("\n--- Tic-Tac-Toe vs Deep Q-Network Agent ---")
    while True:
        choice = input("Do you want to play as X (goes first) or O (goes second)? [X/O]: ").strip().upper()
        if choice in ['X', 'O']:
            break
        print("Invalid choice. Please enter X or O.")
        
    human_player = 1 if choice == 'X' else -1
    env = TicTacToeEnv()
    state = env.reset()
    
    print("\nInitial Board (indices 0 to 8 starting from top-left):")
    # Print indices helper
    print(" 0 | 1 | 2")
    print("-----------")
    print(" 3 | 4 | 5")
    print("-----------")
    print(" 6 | 7 | 8\n")
    
    done = False
    while not done:
        curr_player = env.current_player
        env.render()
        
        if curr_player == human_player:
            while True:
                try:
                    action_input = input(f"Your turn ({choice}). Enter board index (0-8): ").strip()
                    action = int(action_input)
                    if env.is_valid_action(action):
                        break
                    else:
                        print("Invalid move. Square is occupied or index out of range.")
                except ValueError:
                    print("Invalid input. Please enter an integer from 0 to 8.")
        else:
            print("Agent's turn...")
            state_p = state * curr_player
            action = select_action(state_p, agent, 0.0, device) # Exploit (epsilon=0)
            if action is None:
                print("Agent could not find a valid action! Game over.")
                break
            print(f"Agent chose index {action}")
            
        state, reward, done, info = env.step(action)
        
        if done:
            env.render()
            if info.get("winner") == human_player:
                print("Congratulations! You won!")
            elif info.get("winner") == -human_player:
                print("Agent won! Better luck next time.")
            else:
                print("It's a draw!")
            break

def main():
    model_path = "tick_tac_toe_dqn/dqn_tic_tac_toe.pth"
    print("Tic-Tac-Toe DQN Self-Play RL Agent")
    print("==================================")
    
    agent, device = load_agent(model_path)
    
    if agent is None:
        print("No trained agent found. Starting training for 5000 episodes...")
        agent = train_dqn(num_episodes=5000, save_path=model_path)
        agent, device = load_agent(model_path)
        
    while True:
        print("\nChoose an option:")
        print("1. Play against the DQN Agent")
        print("2. Evaluate Agent vs Random Player (100 games)")
        print("3. Evaluate Agent vs Self (100 games, no epsilon)")
        print("4. Retrain Agent")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ").strip()
        if choice == '1':
            play_vs_agent(agent, device)
        elif choice == '2':
            print("Evaluating agent vs Random...")
            wins, draws, losses = evaluate_against_random(agent, device, num_games=100)
            print(f"\nEvaluation Results (100 games):")
            print(f"Wins:   {wins:3d}%")
            print(f"Draws:  {draws:3d}%")
            print(f"Losses: {losses:3d}%")
            print(f"Win/Draw Rate: {wins + draws}%")
        elif choice == '3':
            print("Evaluating agent vs Self...")
            wins_X, wins_O, draws = evaluate_self_play(agent, device, num_games=100)
            print(f"\nSelf-Play Results (100 games):")
            print(f"X Wins: {wins_X:3d}%")
            print(f"O Wins: {wins_O:3d}%")
            print(f"Draws:  {draws:3d}%")
            print(f"Tie Rate: {draws}%")
        elif choice == '4':
            confirm = input("Are you sure you want to retrain the agent? This will overwrite the existing model. [y/n]: ").strip().lower()
            if confirm == 'y':
                agent = train_dqn(num_episodes=5000, save_path=model_path)
                agent, device = load_agent(model_path)
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please choose from 1 to 5.")

if __name__ == "__main__":
    main()

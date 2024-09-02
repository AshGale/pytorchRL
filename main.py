import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import random
import math

# Constants
GRID_SIZE = 8
CELL_SIZE = 50
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
SEARCH_RANGE = 2
MAX_STEPS_PER_EPISODE = 200

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)

class GridEnvironment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.man_pos = [0, 0]
        self.food_pos = None
        self.energy = 100
        self.work_progress = 0
        self.steps = 0
        self.last_action = None
        self.searched_area = set()
        self.current_search_area = set()
        self.is_searching = False
        self.reset()

    def reset(self):
        self.man_pos = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        self.food_pos = None
        self.energy = 100
        self.work_progress = 0
        self.steps = 0
        self.last_action = None
        self.searched_area.clear()
        self.current_search_area.clear()
        self.is_searching = False
        return self.get_state()

    def generate_food(self):
        if self.work_progress >= 100 and self.food_pos is None:
            while True:
                x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                if [x, y] != self.man_pos and not self.is_work_area(x, y):
                    self.food_pos = [x, y]
                    self.work_progress = 0
                    break

    def is_work_area(self, x, y):
        return 3 <= x <= 4 and 3 <= y <= 4

    def get_state(self):
        state = np.zeros((7, GRID_SIZE, GRID_SIZE))
        state[0, self.man_pos[0], self.man_pos[1]] = 1  # Man's position
        if self.food_pos and tuple(self.food_pos) in self.searched_area:
            state[1, self.food_pos[0], self.food_pos[1]] = 1  # Food's position (if discovered)
        state[2, :, :] = self.energy / 100  # Energy level
        state[3, :, :] = self.work_progress / 100  # Work progress
        state[4, :, :] = self.steps / 1000  # Normalized step count
        state[5, 3:5, 3:5] = 1  # Work area
        for x, y in self.searched_area:
            state[6, x, y] = 1  # Searched areas
        return state

    def step(self, action):
        self.steps += 1
        reward = 0
        done = False
        self.last_action = action
        self.is_searching = False
        self.current_search_area.clear()

        if action < 4:  # Move
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            new_x, new_y = self.man_pos[0] + dx, self.man_pos[1] + dy
            if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                self.man_pos = [new_x, new_y]
            self.energy -= 1
        elif action == 4:  # Work
            if self.is_work_area(self.man_pos[0], self.man_pos[1]):
                self.energy -= 5
                self.work_progress += 20
                reward += 0.5  # Small reward for working in the correct area
                if self.work_progress >= 100:
                    reward += 5  # Reward for completing work
            else:
                reward -= 1  # Penalty for attempting to work outside work area
        elif action == 5:  # Search
            self.energy -= 2
            reward += self.search()
            self.is_searching = True

        self.generate_food()

        if self.food_pos and self.man_pos == self.food_pos:
            reward += 10
            self.energy = min(100, self.energy + 50)  # Recover energy, cap at 100
            self.food_pos = None
            self.searched_area.clear()  # Clear searched area after eating

        if self.energy <= 0:
            reward -= 10
            done = True

        return self.get_state(), reward, done

    def search(self):
        reward = 0
        self.current_search_area.clear()
        for dx in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
            for dy in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
                x, y = self.man_pos[0] + dx, self.man_pos[1] + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    self.current_search_area.add((x, y))
                    self.searched_area.add((x, y))
                    if self.food_pos and [x, y] == self.food_pos:
                        reward += 2  # Reward for discovering food
        return reward

class DQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * GRID_SIZE * GRID_SIZE, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, load_model_path=None):
        self.device = torch.device("cpu")  # Use CPU
        print(f"Using device: {self.device}")
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = DQN(state_dim[0], action_dim).to(self.device)
        self.target_net = DQN(state_dim[0], action_dim).to(self.device)
        
        if load_model_path:
            self.policy_net.load_state_dict(torch.load(load_model_path))
            print(f"Loaded pre-trained model from {load_model_path}")
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayBuffer(10000)

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# def draw_grid(screen, env):
    # screen.fill(WHITE)
    # for i in range(GRID_SIZE):
    #     for j in range(GRID_SIZE):
    #         if env.is_work_area(i, j):
    #             pygame.draw.rect(screen, YELLOW, (i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    #         elif env.is_searching and (i, j) in env.current_search_area:
    #             pygame.draw.rect(screen, LIGHT_BLUE, (i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    #         pygame.draw.rect(screen, BLACK, (i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
    
    # # Draw man with color based on last action
    # man_color = BLUE
    # if env.last_action == 4:  # Work action
    #     man_color = PURPLE
    # elif env.last_action == 5:  # Search action
    #     man_color = ORANGE
    # elif env.last_action is not None:  # Move actions
    #     man_color = GREEN
    # pygame.draw.rect(screen, man_color, (env.man_pos[0]*CELL_SIZE, env.man_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # # Draw food (only if in searched area)
    # if env.food_pos and tuple(env.food_pos) in env.searched_area:
    #     pygame.draw.rect(screen, RED, (env.food_pos[0]*CELL_SIZE, env.food_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # # Draw energy bar
    # pygame.draw.rect(screen, GREEN, (0, WINDOW_SIZE, int(WINDOW_SIZE * env.energy / 100), 10))

    # # Draw work progress bar
    # pygame.draw.rect(screen, BLUE, (0, WINDOW_SIZE + 10, int(WINDOW_SIZE * env.work_progress / 100), 10))

    # # Draw step counter
    # font = pygame.font.Font(None, 36)
    # text = font.render(f"Steps: {env.steps}", True, BLACK)
    # screen.blit(text, (10, WINDOW_SIZE + 30))

    # pygame.display.flip()

total_work_progress = 0
total_food_generated = 0
total_food_eaten = 0
total_searches = 0

def main(load_model_path=None):
    pygame.init()
    # screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
    pygame.display.set_caption("Grid RL Game")

    # Initialize total counters
    total_steps = 0
    total_work_progress = 0
    total_food_generated = 0
    total_food_eaten = 0
    total_searches = 0

    env = GridEnvironment()
    state_dim = env.get_state().shape
    agent = Agent(state_dim, 6, load_model_path=load_model_path)  # 6 actions: up, right, down, left, work, search
    
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 50000
    batch_size = 32
    update_target_every = 100

    episode_rewards = []

    for episode in range(100):  # reduced from 10000 to get output
        state = env.reset()
        episode_reward = 0
        episode_work_progress = 0
        episode_food_generated = 0
        episode_food_eaten = 0
        episode_searches = 0
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                math.exp(-1. * total_steps / epsilon_decay)

        for step in range(MAX_STEPS_PER_EPISODE):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.get_action(state, epsilon)
            next_state, reward, done = env.step(action)
            
            # Update counters
            if action == 4 and env.is_work_area(env.man_pos[0], env.man_pos[1]):
                episode_work_progress += 1
            if action == 5:
                episode_searches += 1
            if env.food_pos is None and env.work_progress >= 100:
                episode_food_generated += 1
            if next_state[1].sum() == 0 and state[1].sum() > 0:
                episode_food_eaten += 1
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % update_target_every == 0:
                agent.update_target_network()

            if done or step == MAX_STEPS_PER_EPISODE - 1:
                break

        # Update total counters
        total_work_progress += episode_work_progress
        total_food_generated += episode_food_generated
        total_food_eaten += episode_food_eaten
        total_searches += episode_searches

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        
        # Enhanced logging
        print(f"Episode {episode + 1:4d}, Reward: {episode_reward:6.2f}, Avg Reward: {avg_reward:6.2f}, "
            f"Epsilon: {epsilon:.2f}, Steps: {step + 1:3d}/{MAX_STEPS_PER_EPISODE}, "
            f"Work: {episode_work_progress:3d}, Food Gen: {episode_food_generated:2d}, "
            f"Food Eaten: {episode_food_eaten:2d}, Searches: {episode_searches:3d} | ")
            # f"Total - Work: {total_work_progress:5d}, Food Gen: {total_food_generated:4d}, "
            # f"Food Eaten: {total_food_eaten:4d}, Searches: {total_searches:5d}")

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'trained_model.pth')
    print("Trained model saved as 'trained_model.pth'")

    # Export the model to ONNX format
    dummy_input = torch.randn(1, 7, GRID_SIZE, GRID_SIZE)  # Adjust input size as per your model
    torch.onnx.export(agent.policy_net, dummy_input, "trained_model.onnx", verbose=True)
    print("Model exported to ONNX format as 'trained_model.onnx'")

    # Plot the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('reward_plot.png')
    print("Reward plot saved as 'reward_plot.png'")

    # Print overall statistics
    print(f"\nTraining completed. Overall statistics:")
    print(f"Total episodes: {episode + 1}")
    print(f"Total steps: {total_steps}")
    print(f"Total work progress: {total_work_progress}")
    print(f"Total food generated: {total_food_generated}")
    print(f"Total food eaten: {total_food_eaten}")
    print(f"Total searches: {total_searches}")

    # Print final model summary
    print("\nFinal Model Summary:")
    print(agent.policy_net)

    pygame.quit()

if __name__ == "__main__":
    # To start a new job with a saved model, uncomment the next line and specify the path:
    # main(load_model_path='trained_model.pth')
    
    # To start a new job without a saved model, use:
    main()

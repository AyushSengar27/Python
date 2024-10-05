import numpy as np
import random
import matplotlib.pyplot as plt

# Define the environment: a 5x5 grid world
GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)
OBSTACLE_STATE = [(2, 2), (1, 3), (3, 1)]

# Define actions: up, down, left, right
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}

# Define the learning rate, discount factor, and exploration rate (epsilon)
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.95
EPSILON = 0.1  # Probability of taking a random action (exploration)

# Initialize the Q-Table (size: grid size x grid size x number of actions)
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Function to check if a state is valid (within the grid and not an obstacle)
def is_valid_state(state):
    x, y = state
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return False
    if state in OBSTACLE_STATE:
        return False
    return True

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < EPSILON:  # Exploration
        return random.choice(ACTIONS)
    else:  # Exploitation
        x, y = state
        return ACTIONS[np.argmax(q_table[x, y])]

# Function to take a step in the environment
def take_action(state, action):
    x, y = state
    if action == 'up':
        next_state = (x - 1, y)
    elif action == 'down':
        next_state = (x + 1, y)
    elif action == 'left':
        next_state = (x, y - 1)
    elif action == 'right':
        next_state = (x, y + 1)

    if not is_valid_state(next_state):
        next_state = state  # If next state is invalid, remain in the same state

    return next_state

# Function to get the reward for a state transition
def get_reward(state, next_state):
    if next_state == GOAL_STATE:
        return 100  # High reward for reaching the goal
    elif next_state == state:
        return -1  # Penalty for hitting a wall/obstacle or no movement
    else:
        return -0.1  # Small negative reward for each move to encourage reaching the goal

# Q-Learning algorithm
def q_learning(episodes=1000):
    for episode in range(episodes):
        state = START_STATE  # Start at the initial state

        while state != GOAL_STATE:
            action = choose_action(state)  # Choose an action
            next_state = take_action(state, action)  # Take the action and get next state
            reward = get_reward(state, next_state)  # Get the reward

            # Update Q-Table using the Q-Learning formula
            x, y = state
            next_x, next_y = next_state
            action_idx = ACTION_TO_IDX[action]
            
            # Q-Learning update rule
            q_table[x, y, action_idx] = q_table[x, y, action_idx] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(q_table[next_x, next_y]) - q_table[x, y, action_idx]
            )

            state = next_state  # Move to the next state

# Run the Q-Learning algorithm
q_learning(episodes=1000)

# Display the learned Q-Table
print("Learned Q-Table:")
print(q_table)

# Visualize the path using the learned Q-Table
def visualize_policy():
    state = START_STATE
    path = [state]

    while state != GOAL_STATE:
        x, y = state
        action_idx = np.argmax(q_table[x, y])
        action = ACTIONS[action_idx]
        state = take_action(state, action)
        path.append(state)

    return path

# Get the path found by the learned Q-Table
optimal_path = visualize_policy()

# Visualize the optimal path on the grid
def plot_grid_with_path(optimal_path):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    # Mark the start, goal, obstacles, and path
    grid[START_STATE] = 1  # Start state (green)
    grid[GOAL_STATE] = 2  # Goal state (red)
    for obs in OBSTACLE_STATE:
        grid[obs] = -1  # Obstacles (black)

    plt.imshow(grid, cmap='coolwarm', origin='upper')

    for i, (x, y) in enumerate(optimal_path):
        if i == 0:
            plt.text(y, x, 'S', ha='center', va='center', color='white', fontsize=12)
        elif (x, y) == GOAL_STATE:
            plt.text(y, x, 'G', ha='center', va='center', color='white', fontsize=12)
        else:
            plt.text(y, x, '.', ha='center', va='center', color='black', fontsize=10)

    plt.title("Optimal Path Found by Q-Learning")
    plt.show()

# Plot the optimal path
plot_grid_with_path(optimal_path)

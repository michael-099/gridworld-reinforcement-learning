import numpy as np
import random

# Grid world parameters
grid_size = 5
goal_state = (4, 4)
actions = ['up', 'down', 'left', 'right']
alpha = 0.5    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Epsilon-greedy exploration
episodes = 1000

# Initialize Q-table: 5x5 grid, 4 actions
Q = np.zeros((grid_size, grid_size, len(actions)))

# Step function
def step(state, action):
    i, j = state
    if action == 0:  # up
        i = max(i - 1, 0)
    elif action == 1:  # down
        i = min(i + 1, grid_size - 1)
    elif action == 2:  # left
        j = max(j - 1, 0)
    elif action == 3:  # right
        j = min(j + 1, grid_size - 1)
    reward = 10 if (i, j) == goal_state else -1
    return (i, j), reward

# Q-learning algorithm
for ep in range(episodes):
    state = (0, 0)  # Start at top-left corner
    while state != goal_state:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state[0], state[1], :])
        
        next_state, reward = step(state, action)
        
        # Q-learning update
        best_next = np.max(Q[next_state[0], next_state[1], :])
        Q[state[0], state[1], action] += alpha * (reward + gamma * best_next - Q[state[0], state[1], action])
        
        state = next_state

print("Learned Q-values:")
for i in range(grid_size):
    for j in range(grid_size):
        print(f"State ({i},{j}): {Q[i,j]}")

import numpy as np

# Define the grid world size
grid_size = 5

# Define rewards: -1 for each step, +10 for the goal state
rewards = -1 * np.ones((grid_size, grid_size))
goal_state = (4, 4)
rewards[goal_state] = 10  # Reward for reaching the goal

# Define possible actions
actions = ['up', 'down', 'left', 'right']

# Discount factor
gamma = 0.9

# Initialize the value function to zeros
V = np.zeros((grid_size, grid_size))

# Helper function to move in the grid
def step(state, action):
    i, j = state
    if action == 'up':
        i = max(i - 1, 0)
    elif action == 'down':
        i = min(i + 1, grid_size - 1)
    elif action == 'left':
        j = max(j - 1, 0)
    elif action == 'right':
        j = min(j + 1, grid_size - 1)
    return (i, j)

# Value Iteration algorithm
def value_iteration(V, rewards, gamma, theta=1e-4):
    while True:
        delta = 0
        new_V = V.copy()
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == goal_state:
                    continue  # Skip the goal state
                # Compute value for each action
                action_values = []
                for action in actions:
                    next_state = step((i, j), action)
                    reward = rewards[next_state]
                    action_values.append(reward + gamma * V[next_state])
                # Update value function with maximum action value
                new_V[i, j] = max(action_values)
                delta = max(delta, abs(new_V[i, j] - V[i, j]))
        V = new_V
        if delta < theta:
            break
    return V

# Compute the optimal value function
V_opt = value_iteration(V, rewards, gamma)

# Derive the optimal policy
policy = np.empty((grid_size, grid_size), dtype=str)
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) == goal_state:
            policy[i, j] = 'G'  # Goal
            continue
        action_values = []
        for action in actions:
            next_state = step((i, j), action)
            reward = rewards[next_state]
            action_values.append(reward + gamma * V_opt[next_state])
        best_action = actions[np.argmax(action_values)]
        policy[i, j] = best_action

print("Optimal Value Function:")
print(V_opt)
print("\nOptimal Policy:")
print(policy)

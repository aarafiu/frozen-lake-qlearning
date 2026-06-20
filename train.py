import numpy as np
from environment import FrozenLakeEnv
from agent import QLearningAgent

# ============================================================
# Training Setup
# ============================================================
# Note: epsilon_decay = 0.9998 was chosen after debugging — an initial
# decay rate of 0.995 caused epsilon to become negligible before the
# agent could ever find the Goal by chance, leading to 0% success.
# Slowing the decay gives the agent a long enough exploration window
# to find the goal at least once before switching to exploitation.
#
# Because training involves randomness, it is still possible (though
# unlikely) for a single run to get unlucky and never find the goal
# during exploration. To guard against this, training is retried
# automatically if a run ends with zero successes.

env = FrozenLakeEnv()

n_episodes = 50000
max_steps = 100          # max moves per episode

attempt = 0

while True:
    attempt += 1

    agent = QLearningAgent(
        n_states=64,
        n_actions=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9998,
        epsilon_min=0.01
    )

    episode_rewards = []       # total reward earned in each episode
    episode_successes = []     # True/False — did this episode reach the Goal?
    epsilon_history = []       # epsilon value after each episode

    # ============================================================
    # Training Loop
    # ============================================================
    for episode in range(n_episodes):

        state = env.reset()        # start a new episode
        total_reward = 0           # track reward for this episode
        done = False

        for step in range(max_steps):

            action = agent.choose_action(state)            # agent picks an action
            next_state, reward, done = env.step(action)    # environment responds
            agent.update(state, action, reward, next_state, done)  # agent learns

            state = next_state
            total_reward += reward

            if done:
                break   # episode ended — either Goal or Hole

        agent.decay_epsilon()   # reduce exploration slightly

        episode_rewards.append(total_reward)
        episode_successes.append(total_reward > 0)
        epsilon_history.append(agent.epsilon)

    total_successes = sum(episode_successes)
    print(f"Attempt {attempt}: total successes = {total_successes}")

    if total_successes > 0:
        break   # learning happened — keep this run

# ============================================================
# Diagnostics
# ============================================================
first_5000 = np.mean(episode_successes[:5000]) * 100
last_5000 = np.mean(episode_successes[-5000:]) * 100

print(f"Success rate, first 5000 episodes: {first_5000:.2f}%")
print(f"Success rate, last 5000 episodes: {last_5000:.2f}%")

# ============================================================
# Save trained Q-table and training statistics
# ============================================================
np.save("q_table.npy", agent.q_table)
np.save("episode_rewards.npy", np.array(episode_rewards))
np.save("episode_successes.npy", np.array(episode_successes))
np.save("epsilon_history.npy", np.array(epsilon_history))

print("Training complete!")
print("Q-table saved to q_table.npy")
print("Training statistics saved")

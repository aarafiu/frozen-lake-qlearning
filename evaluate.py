import numpy as np
from environment import FrozenLakeEnv

# ============================================================
# Load the trained Q-table saved by train.py
# ============================================================
q_table = np.load("q_table.npy")

# create a fresh environment to test against
env = FrozenLakeEnv()


# ============================================================
# Policy Extraction (Part D)
# ============================================================
ACTION_SYMBOLS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

def extract_policy(q_table, env):
    policy_grid = []   # will hold 8 rows, each a list of symbols

    for row in range(env.nrows):
        row_symbols = []   # symbols for this row

        for col in range(env.ncols):
            cell = env.map[row][col]

            if cell == "H":
                row_symbols.append("H")        # hole stays H
            elif cell == "G":
                row_symbols.append("G")        # goal stays G
            else:
                state = row * env.ncols + col            # convert row,col to state number
                best_action = np.argmax(q_table[state])  # find best action
                row_symbols.append(ACTION_SYMBOLS[best_action])  # convert to arrow

        policy_grid.append(row_symbols)

    return policy_grid


def print_policy(policy_grid):
    print("")
    for row in policy_grid:
        print(" ".join(row))
    print("")


# ============================================================
# Evaluation (Part E)
# ============================================================
def evaluate_agent(q_table, env, n_eval_episodes=100, max_steps=100):
    successes = 0
    failures = 0
    total_rewards = []

    for episode in range(n_eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        for step in range(max_steps):
            # always pick the best known action — no exploration
            action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

        if episode_reward > 0:
            successes += 1
        else:
            failures += 1

    success_rate = (successes / n_eval_episodes) * 100
    average_reward = np.mean(total_rewards)

    return {
        "success_rate": success_rate,
        "average_reward": average_reward,
        "successes": successes,
        "failures": failures
    }


if __name__ == "__main__":
    print("Learned Policy:")
    policy = extract_policy(q_table, env)
    print_policy(policy)

    results = evaluate_agent(q_table, env, n_eval_episodes=100)

    print("Success Rate: {:.2f}%".format(results["success_rate"]))
    print("Average Reward: {:.4f}".format(results["average_reward"]))
    print("Number of Successful Runs:", results["successes"])
    print("Number of Failures:", results["failures"])

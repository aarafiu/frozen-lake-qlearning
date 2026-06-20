import numpy as np

class QLearningAgent:

    def __init__(self, n_states, n_actions, alpha, gamma, epsilon, epsilon_decay, epsilon_min):

        self.n_states = n_states                          # total number of states (64)
        self.n_actions = n_actions                        # total number of actions (4)
        self.alpha = alpha                                # learning rate
        self.gamma = gamma                                # discount factor
        self.epsilon = epsilon                            # starting epsilon exploration rate
        self.epsilon_decay = epsilon_decay                # epsilon decreasing rate
        self.epsilon_min = epsilon_min                    # epsilon should never go below this

        # create the Q table starting with 0s
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):

        # generate a random number between 0 and 1
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)           # explore (pick a random action)
        else:
            return np.argmax(self.q_table[state])              # exploit (pick the action with highest Q value)
                                                                  # argmax returns the index of the highest value

    def update(self, state, action, reward, next_state, done):

        current_q = self.q_table[state, action]                   # current Q value for this state-action pair
        best_next_q = np.max(self.q_table[next_state])            # best possible Q value from the next state

        # if episode is done, there is no future reward
        if done:
            target = reward
        else:
            target = reward + self.gamma * best_next_q

        self.q_table[state, action] = current_q + self.alpha * (target - current_q)  # Q-learning update equation

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)      # never go below epsilon_min

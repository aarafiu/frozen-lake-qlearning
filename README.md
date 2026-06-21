
## Frozen Lake from First Principles Using Q-Learning

### Introduction

#### Reinforcement Learning
Reinforcement Learning (RL) is a branch of machine learning where an agent learns to make decisions by interacting with an environment through trial and error. Rather than being told the correct answer, the agent takes actions, receives feedback in the form of rewards, and gradually learns which actions lead to better outcomes over time.

#### The Frozen Lake
Frozen Lake is a grid-world problem in which an agent must navigate from a starting position to a goal position on a frozen lake, while avoiding holes in the ice. Each cell on the grid is either safe to walk on (Frozen), a hole that ends the episode in failure, the starting point, or the goal. The challenge is that the agent starts with no knowledge of where the holes are and must learn a safe path purely through experience.

----

### Environment Design

#### State Representation
The environment is an 8×8 grid, giving 64 possible states. Each state can be represented as a single integer, calculated from the agent's row and column position using the formula:

```
state = row × 8 + column
```

For example, the agent's starting position (row 0, column 0) is state 0, and the Goal at (row 7, column 7) is state 63. This flattening makes it simple to index directly into the Q-table, which is structured as a 64×4 array.

### Action Representation
The agent can take one of four actions at each step:

| Action | Number | Effect |
|---|---|---|
| Left | 0 | column − 1 |
| Down | 1 | row + 1 |
| Right | 2 | column + 1 |
| Up | 3 | row − 1 |

If an action would move the agent outside the grid boundaries, the agent simply remains in its current position instead.

### Reward Structure

| Outcome | Reward |
|---|---|
| Reaching the Goal (G) | +1 |
| Falling into a Hole (H) | 0 |
| Any other move (F) | 0 |

Both Holes and the Goal are terminal states — they end the episode. Although falling into a Hole gives the same numeric reward (0) as a normal move, it differs in that the episode terminates immediately, meaning the agent forfeits any chance of future reward. This implicit penalty is what discourages the agent from repeating hole-falling behavior, even without an explicit negative reward.

---

### Q-Learning Algorithm

#### Description
Q-Learning is a reinforcement learning algorithm that learns the value of taking a given action in a given state. It works by maintaining a **Q table** which is a lookup table or grid of (number of states × number of actions) — where each entry, Q(s,a), estimates how good it is to take action (a) while in state (s). Over many episodes of trial and error, these estimates are refined until they converge to values that reflect an optimal policy.

### The Update Equation
After every action, the Q-table is updated using the equation:

```
Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]
```

| Symbol | Meaning |
|---|---|
| Q(s,a) | Current estimated value of taking action (a) in state (s) |
| α (alpha) | Learning rate: to control how much new information overrides old estimates |
| r | Reward received after taking the action |
| γ (gamma) | Discount factor — controls how much future rewards matter relative to immediate ones |
| max Q(s',a') | The best estimated value achievable from the next state |

In simple terms: the agent compares what it currently believes an action is worth against what it just experienced (the reward received, plus the best possible value from where it ended up), and updates its belief slightly toward that new information, scaled by the learning rate.

### Exploration Strategy — Epsilon-Greedy with Decay
To balance trying new actions (**exploration**) against using what it has already learned (**exploitation**), the agent uses an **epsilon-greedy** strategy:

- With probability ε (epsilon), the agent picks a random action (explore)
- With probability 1 − ε, the agent picks the action with the highest known Q value (exploit)

Epsilon starts at 1.0 (fully random) and decays after every episode by a fixed decay rate, gradually shifting the agent from exploration toward exploitation as training progresses. A minimum epsilon value (epsilon_min) ensures the agent never stops exploring completely, even late in training.

---

### Training Procedure

#### Hyperparameters

| Hyperparameter | Value | Purpose | Why This Value |
|---|---|---|---|
| Learning rate (α) | 0.1 | Controls how quickly the Q values update based on new information | Standard default for Q Learning from; not exhaustively tuned. |
| Discount factor (γ) | 99% | Gives strong weight to future rewards.| Deliberately chosen high because the only reward exists at the Goal; a high γ is needed for that reward to propagate backward and still be substancial across many cells or states during training |
| Initial epsilon | 1 | Agent starts by exploring purely at random | Standard default starting point. becasue the Q table is all zeros at the start, so there is nothing to exploit |
| Epsilon decay | 0.9998 | Controls how quickly exploration is phased out | Calculated analytically after debugging. An initial value of `0.995` caused epsilon to hit its low under 1000 episodes which was too early for the agent to realistically reach the Goal. `0.9998` was used to extend the exploration window further |
| Minimum epsilon | 0.01 | Ensures a small amount of exploration always remains | Standard default; something small to avoids the agent from becoming permanently stuck on a suboptimal policy late in training |
| Episodes | 50,000 | Number of training episodes | Originally an arbitrary starting choice of 10,000 was increased to 50,000 while debugging the epsilon decay issue. Increasing episodes alone did not fix the underlying bug, but 50,000 was retained because, once epsilon decay was corrected. |
| Max steps per episode | 100 | Prevents an episode from running indefinitely | Set generously above the minimum 14 steps needed to reach the goal, giving the agent room to wander during early random exploration without being cut off too abruptly |

#### Debugging the Epsilon Decay Rate
The `epsilon_decay = 0.9998` value shown above was not the first one tried. An initial value of `0.995` was tested first, but caused training to fail with a 0% success rate even after increasing episodes from 10,000 to 50,000. Diagnosis revealed that epsilon was reaching its minimum value under 1000 episodes which was too early for the agent to realistically reach the Goal at least once by random exploration. Once epsilon was decayed more slowly, the agent's longer exploration window allowed it to find the Goal early in training, after which the learned reward propagated through the Q table to improve the success rate.

#### Handling Random Variance Between Runs
Because exploration relies on randomness, individual training runs can occasionally fail to find the Goal even once, purely by chance. To guard against this, `train.py` automatically retries training if a run completes with zero recorded successes, ensuring a usable Q table is always produced regardless of random seed.

---

### Results

#### Final Success Rate
After training for 50,000 episodes, the agent achieved a **100% success rate** over 100 evaluation episodes, with an average reward of 1.0 and zero failures. During training, the success rate climbed from 0% to around **98%** by the last 5,000 episodes, as shown in the training graphs in `results/`.

#### Learned Policy

```
↓ ↓ ↓ ↓ ↓ → ↓ ↓
→ → → → ↓ ↓ ↓ ↓
↑ ↑ ↑ H ↓ ↓ ↓ ↓
↑ ↑ ↑ H ↓ ↓ ↓ ↓
↑ ↑ ↑ H → → → ↓
↑ H H → ↑ ↑ H ↓
↑ H ← ← H ↑ H ↓
↑ ← ← H ← ↑ → G
```

#### Discussion of Performance
The learned policy guides the agent safely from the Start to the Goal, avoiding every Hole along the way. Since the environment is deterministic; meaning every action always does exactly what it's supposed to, with no randomness in movement. As such a working policy will succeed every single time. That's why the evaluation result is 100%, with no failures at all.

A few arrows on the grid, especially in squares far from the agent's actual path, may look a little odd at first glance. This happens because the agent rarely visits those squares once it finds a good route, so it never gets the chance to fully learn the best action there. It doesn't affect performance, since the agent never actually needs to pass through those squares.

---

### Execution Instructions

#### Requirements

- Python 3.x
- numpy
- matplotlib` (only required for generating training visualizations)

Install dpendencies with:
```bash
pip install -r requirements.txt
```

### Running the Project

**1. Train the agent:**

```bash
python train.py
```
This trains the agent for 50,000 episodes and saves the trained Q-table (`q_table.npy`) along with training statistics (`episode_rewards.npy`, `episode_successes.npy`, `epsilon_history.npy`).

**2. Evaluate the trained agent:**

```bash
python evaluate.py
```
This loads the saved Q-table, displays the learned policy in grid form, and evaluates the agent over 100 episodes, reporting success rate, average reward, and failure count.

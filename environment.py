# The lake map as given

MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
]

ACTION = [
    (0,-1),   # 0: Left
    (1,0),    # 1: Down
    (0,1),    # 2: Right
    (-1,0)    # 3: Up
]

class FrozenLakeEnv:

    def __init__(self):
        self.map = MAP                # store the lake map
        self.nrows = 8                # number of rows
        self.ncols = 8                # number of columns
        self.state = 0                # agent starts at position 0 (row 0, col 0)

    def reset(self):
        self.state = 0                # send agent back to the start
        return self.state             # return the starting state

    def get_state(self):
        return self.state             # report the current state

    def is_terminal(self):
        row = self.state // 8         # convert state number to row
        col = self.state % 8          # convert state number to column
        cell = self.map[row][col]     # look up the character at that position
        return cell in ("H", "G")     # return True if Hole or Goal, False otherwise

    def step(self, action):

        # converting current state back into row and column
        row = self.state // 8
        col = self.state % 8

        # calculating the new row and column based on the action
        row_change, col_change = ACTION[action]
        new_row = row + row_change
        new_col = col + col_change

        # enforcing boundaries so the agent stays inside grid
        new_row = max(0, min(self.nrows - 1, new_row))
        new_col = max(0, min(self.ncols - 1, new_col))

        # update the agent's state
        self.state = new_row * self.ncols + new_col

        # calculate the reward
        cell = self.map[new_row][new_col]

        if cell == "G":
            reward = 1    # reached the goal
        else:
            reward = 0    # anywhere else gives 0

        # check if episode is over (that is when we enter a hole or reach the goal)
        done = self.is_terminal()

        # return everything the agent needs to know
        return self.state, reward, done

    def render(self):
        agent_row = self.state // 8
        agent_col = self.state % 8

        print("")

        for row in range(self.nrows):
            row_display = ""

            for col in range(self.ncols):
                if row == agent_row and col == agent_col:
                    row_display += "* "
                else:
                    row_display += self.map[row][col] + " "

            print(row_display)

        print("")

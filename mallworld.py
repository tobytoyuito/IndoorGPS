import numpy as np
from utils import check_random_state


# Maze state is represented as a 2-element NumPy array: (Y, X). Increasing Y is South.

# Possible actions, expressed as (delta-y, delta-x).
maze_actions = {
    'N': np.array([0, -1, 0]),
    'S': np.array([0, 1, 0]),
    'E': np.array([0, 0, 1]),
    'W': np.array([0, 0, -1]),
    'K': np.array([0, 0, 0])
}

def parse_topology(topology):
    return np.array([[list(col) for col in row] for row in topology])


class Maze(object):
    """
    Simple wrapper around a NumPy 3D array to handle flattened indexing and staying in bounds.
    """
    def __init__(self, topology):
        self.topology = parse_topology(topology)
        self.flat_topology = self.topology.ravel()
        self.shape = self.topology.shape

    def in_bounds_flat(self, position):
        return 0 <= position < np.product(self.shape)

    def in_bounds_unflat(self, position):
        return 0 <= position[0] < self.shape[0] and 0 <= position[1] < self.shape[1] and 0 <= position[2] < self.shape[2]

    def get_flat(self, position):
        if not self.in_bounds_flat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.flat_topology[position]

    def get_unflat(self, position):
        if not self.in_bounds_unflat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.topology[tuple(position)]

    def flatten_index(self, index_tuple):
        return np.ravel_multi_index(index_tuple, self.shape)

    def unflatten_index(self, flattened_index):
        return np.unravel_index(flattened_index, self.shape)

    def flat_positions_containing(self, x):
        return list(np.nonzero(self.flat_topology == x)[0])

    def flat_positions_not_containing(self, x):
        return list(np.nonzero(self.flat_topology != x)[0])
    
    def flat_change(self, position, element):
        if not self.in_bounds_flat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        self.flat_topology[position] = element
        self.topology[tuple(self.unflatten_index(position))] = element

    def __str__(self):
        return '\n\n'.join('\n'.join(''.join(row) for row in level) for level in self.topology.tolist())

    def __repr__(self):
        return 'Maze({})'.format(repr(self.topology.tolist()))

#***********************************************************************
def move_avoiding_walls(maze, position, action, climb):
    """
    Return the new position after moving, and the event that happened ('hit-wall' or 'moved').

    Works with the position and action as a (row, column) array.
    """
    # Compute new position
    new_position = position + action
    #print(position, action, new_position)
    # Compute collisions with walls, including implicit walls at the ends of the world.
    if not maze.in_bounds_unflat(new_position) or maze.get_unflat(new_position) == '#':
        return position, 'hit-wall'

    # Go to the stairs
    if maze.get_unflat(new_position) == '%':
        flat_new_position = climb[maze.flatten_index(new_position)]
        new_position = maze.unflatten_index(flat_new_position)
        return new_position, 'stair'

    # GO to people and back
    if maze.get_unflat(new_position) == 'a':
        return position, 'runinto-people'
    
    if action[1]==action[2]:
        return new_position, 'stay'

    return new_position, 'moved'


def move_adversary(maze, position, action):
    """
    Return the new position of one adversary.
    
    Works with the positon and action as a (row, column) array.
    """
    
    new_position = position + action
    
    # Compute collisions with walls, including implicit walls at the ends of the world.
    if not maze.in_bounds_unflat(new_position) or maze.get_unflat(new_position) == '#' or maze.get_unflat(new_position) == '%':
        return position,1
    
    # GO to people and back
    if maze.get_unflat(new_position) == 'a':
        return position,0
    
    # GO to people and back
    if maze.get_unflat(new_position) == 'I':
        return position,0
    
    return new_position,0

#***********************************************************************




class MallWorld(object):
    """
    A simple task in a maze: get to the goal.

    Parameters
    ----------

    maze : list of strings or lists
        maze topology (see below)

    rewards: dict of string to number. default: {'*': 10}.
        Rewards obtained by being in a maze grid with the specified contents,
        or experiencing the specified event (either 'hit-wall' or 'moved'). The
        contributions of content reward and event reward are summed. For
        example, you might specify a cost for moving by passing
        rewards={'*': 10, 'moved': -1}.

    terminal_markers: sequence of chars, default '*'
        A grid cell containing any of these markers will be considered a
        "terminal" state.

    action_error_prob: float
        With this probability, the requested action is ignored and a random
        action is chosen instead.

    random_state: None, int, or RandomState object
        For repeatable experiments, you can pass a random state here. See
        http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html

    Notes
    -----

    Maze topology is expressed textually. Key:
     '#': wall
     '.': open (really, anything that's not '#')
     '*': goal
     'o': origin
    """

    def __init__(self, maze, num_advs=3, rewards={'*': 10}, terminal_markers='*', action_error_prob=0, random_state=None, directions="NSEWK"):

        self.maze = Maze(maze) if not isinstance(maze, Maze) else maze
        self.rewards = rewards
        self.terminal_markers = terminal_markers
        self.action_error_prob = action_error_prob
        self.random_state = check_random_state(random_state)
        self.num_advs = num_advs
        
        # find all available position
        self.available_state = self.maze.flat_positions_containing('.')
                
        # randomly choose position of adversary
        self.adversaries_position = np.random.choice(self.available_state, replace=False, size=self.num_advs)
        
        # find stairs
        self.stairs = self.maze.flat_positions_containing('%')
        
        # dict to upstair or downstair
        self.climb = {self.stairs[0]:self.stairs[1], self.stairs[1]: self.stairs[0]}

        self.actions = [maze_actions[direction] for direction in directions]
        self.num_actions = len(self.actions)
        self.state = None
        self.reset()
        self.num_states = self.maze.shape[0] * self.maze.shape[1] * self.maze.shape[2]
        self.current_situation = None
        self.adversary_actions = None
        self.total_hit = 0

    def __repr__(self):
        return 'GridWorld(maze={maze!r}, rewards={rewards}, terminal_markers={terminal_markers}, action_error_prob={action_error_prob})'.format(**self.__dict__)

    def reset(self):
        """
        Reset the position of agent and adversaries to a starting position (an 'o'), chosen at random.
        """
        options = self.maze.flat_positions_containing('o')
        self.state = options[self.random_state.choice(len(options))]
        self.current_situation = self.get_current_situation()
        self.adversaries_position = np.random.choice(self.available_state, replace=False, size=self.num_advs)
        self.total_hit = 0

    def is_terminal(self, state):
        """Check if the given state is a terminal state."""
        return self.maze.get_flat(state) in self.terminal_markers
    
    def get_current_situation(self):
        f,x,y = self.maze.unflatten_index(self.observe()[0])
        adv_temp = np.zeros((3,1))
        pos_dict = {(-2,-2): 1, (-2,-1): 2, (-2,0): 3, (-2,1): 4, (-2,2): 5,(-1,-2): 6, (-1,-1): 7, (-1,0): 8, (-1,1): 9, (-1,2): 10,(0,-2): 11, (0,-1): 12, (0,1): 13, (0,2): 14,(1,-2): 15, (1,-1): 16, (1,0): 17, (1,1): 18, (1,2): 19,(2,-2): 20, (2,-1): 21, (2,0): 22, (2,1): 23, (2,2): 24}
        for i in xrange(3):
            f1,x1,y1 = self.maze.unflatten_index(self.adversaries_position[i])
            if(((x1-x,y1-y) in pos_dict)and (f==f1)): 
                adv_temp[i] = pos_dict[(x1-x,y1-y)]
            else: 
                adv_temp[i] = 0    
        self.current_situation = self.observe()[0]+162*adv_temp[0]+162*25*adv_temp[1]+162*25*25*adv_temp[2]
        return self.current_situation    

    def observe(self):
        """
        Return the current state and position of adversaries as integers.

        The state and position is the index into the flattened maze.
        """
        return self.state, self.adversaries_position
    
    def add_adv_maze(self):
        """
        Return the gridworld with adversary
        """
        tmp = Maze(self.maze.topology)
        for adver_p in self.adversaries_position:
            tmp.flat_change(adver_p, 'a')
    
        return tmp
        
    
    def visualize(self):
        """
        Print the gridworld with adversary and agent
        """
        tmp = self.add_adv_maze()
        tmp.flat_change(self.state, 'I')
        
        print str(tmp)
        

    def perform_action_adversary_rand(self, action_idx):
        """Perform an action (specified by index), yielding a new state and reward."""
        # In the absorbing end state, nothing does anything.
        if self.is_terminal(self.state):
            return self.observe(), 0
        
        # move adversary first
        for idx in xrange(self.num_advs):
            action_idx_a = self.random_state.choice(self.num_actions)
            action = self.actions[action_idx_a]
            new_position = move_adversary(self.add_adv_maze(),
                                          self.maze.unflatten_index(self.adversaries_position[idx]),
                                          action)
            self.adversaries_position[idx] = self.maze.flatten_index(new_position[0])
            
        # move agent
        if self.action_error_prob and self.random_state.rand() < self.action_error_prob:
            action_idx = self.random_state.choice(self.num_actions)
        action = self.actions[action_idx]
        new_state_tuple, result = move_avoiding_walls(self.add_adv_maze(), 
                                                      self.maze.unflatten_index(self.state), 
                                                      action, self.climb)
        hit = 0
        if result=='runinto-people':
            hit = 1
            self.total_hit += 1
        self.state = self.maze.flatten_index(new_state_tuple)
        self.current_situation = self.get_current_situation()

        reward = self.rewards.get(self.maze.get_flat(self.state), 0) + self.rewards.get(result, 0)
        return self.observe(), reward, hit

    
    def perform_action_adversary_w_policy(self, action_idx):
        """Perform an action (specified by index), yielding a new state and reward."""
        # In the absorbing end state, nothing does anything.
        if self.is_terminal(self.state):
            return self.observe(), 0
        
        # move adversary first
        if self.adversary_actions is None:
            self.adversary_actions = []
            for idx in xrange(self.num_advs):
                action_idx_a = self.random_state.choice(self.num_actions)
                action = self.actions[action_idx_a]
                self.adversary_actions.append(action)
                new_position = move_adversary(self.add_adv_maze(),
                                          self.maze.unflatten_index(self.adversaries_position[idx]),
                                          action)
                self.adversaries_position[idx] = self.maze.flatten_index(new_position[0])
                if (new_position[1]==1):
                    action_idx_a = self.random_state.choice(self.num_actions)
                    action = self.actions[action_idx_a]
                    self.adversary_actions[idx] = action
                          
        for idx in xrange(self.num_advs):
            action = self.adversary_actions[idx]
            
            if(np.random.uniform(0,1) < 0.2):
                action_idx_a = np.random.random_integers(0,self.num_actions-1)  
                action = self.actions[action_idx_a]
                self.adversary_actions[idx] = action
            
            new_position = move_adversary(self.add_adv_maze(),
                                          self.maze.unflatten_index(self.adversaries_position[idx]),
                                          action)   
            self.adversaries_position[idx] = self.maze.flatten_index(new_position[0])
            if (new_position[1]==1):
                    action_idx_a = self.random_state.choice(self.num_actions)
                    action = self.actions[action_idx_a]
                    self.adversary_actions[idx] = action
                          
                    
        # move agent
        if self.action_error_prob and self.random_state.rand() < self.action_error_prob:
            action_idx = self.random_state.choice(self.num_actions)
        action = self.actions[action_idx]
        new_state_tuple, result = move_avoiding_walls(self.add_adv_maze(), 
                                                      self.maze.unflatten_index(self.state), 
                                                      action, self.climb)
        hit = 0
        if result=='runinto-people':
            hit = 1
            self.total_hit += 1
        self.state = self.maze.flatten_index(new_state_tuple)
        self.current_situation = self.get_current_situation()

        reward = self.rewards.get(self.maze.get_flat(self.state), 0) + self.rewards.get(result, 0)
        return self.observe(), reward, hit

    def num_total_hit(self):
        return self.total_hit

    def as_mdp(self):
        transition_probabilities = np.zeros((self.num_states, self.num_actions, self.num_states))
        rewards = np.zeros((self.num_states, self.num_actions, self.num_states))
        action_rewards = np.zeros((self.num_states, self.num_actions))
        destination_rewards = np.zeros(self.num_states)

        for state in range(self.num_states):
            destination_rewards[state] = self.rewards.get(self.maze.get_flat(state), 0)

        is_terminal_state = np.zeros(self.num_states, dtype=np.bool)

        for state in range(self.num_states):
            if self.is_terminal(state):
                is_terminal_state[state] = True
                transition_probabilities[state, :, state] = 1.
            else:
                for action in range(self.num_actions):
                    new_state_tuple, result = move_avoiding_walls(self.maze, self.maze.unflatten_index(state), self.actions[action])
                    new_state = self.maze.flatten_index(new_state_tuple)
                    transition_probabilities[state, action, new_state] = 1.
                    action_rewards[state, action] = self.rewards.get(result, 0)

        # Now account for action noise.
        transitions_given_random_action = transition_probabilities.mean(axis=1, keepdims=True)
        transition_probabilities *= (1 - self.action_error_prob)
        transition_probabilities += self.action_error_prob * transitions_given_random_action

        rewards_given_random_action = action_rewards.mean(axis=1, keepdims=True)
        action_rewards = (1 - self.action_error_prob) * action_rewards + self.action_error_prob * rewards_given_random_action
        rewards = action_rewards[:, :, None] + destination_rewards[None, None, :]
        rewards[is_terminal_state] = 0

        return transition_probabilities, rewards

    def get_max_reward(self):
        transition_probabilities, rewards = self.as_mdp()
        return rewards.max()

    ### Old API, where terminal states were None.

    def observe_old(self):
        return None if self.is_terminal(self.state) else self.state

    def perform_action_old(self, action_idx):
        new_state, reward = self.perform_action(action_idx)
        if self.is_terminal(new_state):
            return None, reward
        else:
            return new_state, reward


    samples = {
        'trivial': [
            '###',
            '#o#',
            '#.#',
            '#*#',
            '###'],

        'larger': [
            '#########',
            '#..#....#',
            '#..#..#.#',
            '#..#..#.#',
            '#..#.##.#',
            '#....*#.#',
            '#######.#',
            '#o......#',
            '#########'],

        'two level': [
            [
            '#########',
            '#......%#',
            '#.......#',
            '#..#....#',
            '#..#....#',
            '#.......#',
            '##......#',
            '#o......#',
            '#########'],
            [
            '#########',
            '#......%#',
            '#.......#',
            '#..#....#',
            '#..#....#',
            '#.......#',
            '##......#',
            '#*......#',
            '#########']
        ]
    }


def construct_cliff_task(width, height, goal_reward=50, move_reward=-1, cliff_reward=-100, **kw):
    """
    Construct a 'cliff' task, a GridWorld with a "cliff" between the start and
    goal. Falling off the cliff gives a large negative reward and ends the
    episode.

    Any other parameters, like action_error_prob, are passed on to the
    GridWorld constructor.
    """

    maze = ['.' * width] * (height - 1)  # middle empty region
    maze.append('o' + 'X' * (width - 2) + '*') # bottom goal row

    rewards = {
        '*': goal_reward,
        'moved': move_reward,
        'hit-wall': move_reward,
        'X': cliff_reward
    }

    return GridWorld(maze, rewards=rewards, terminal_markers='*X', **kw)

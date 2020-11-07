"""
COMP7702 2020 Assignment 3 Solution
Author: Joel Thomas
Student Number: 44793203
"""

from laser_tank import LaserTankMap, DotDict
import time
import numpy as np


class Solver:
    def __init__(self, game_map):
        """
        Solves a given LaserTankMap problem using either one synchronous VI, modified PI, or MCTS. Stores a list of
        all possible states if using VI or PI to solve the problem. Also computes a list of all transition probabilities
        for a MOVE_FORWARD only policy.

        Parameters:
            game_map (LaserMapTank): A LaserTankMap instance.
        """
        self.game_map = game_map
        # Update these when solving using VI or PI
        self.values = None
        self.policy = None

        if game_map.method in ["vi", "pi"]:
            # Form all possible states (x, y, d) where d = heading of the player tank
            self.states = []
            for x in range(1, self.game_map.x_size - 1):
                for y in range(1, self.game_map.y_size - 1):
                    for d in range(4):
                        self.states.append((x, y, d))

        # Form transition probabilities according to MOVE_FORWARD only policy
        self.transition_ps = [[[0 for _ in LaserTankMap.DIRECTIONS]
                              for __ in range(1, self.game_map.y_size - 1)]
                              for ___ in range(1, self.game_map.x_size - 1)]
        for y in range(1, self.game_map.y_size - 1):
            for x in range(1, self.game_map.x_size - 1):
                for d in range(4):
                    s = (x, y, d)
                    self.transition_ps[x - 1][y - 1][d] = self.compute_transition_p(s)

    def compute_transition_p(self, s):
        """
        Computes a dictionary of all possible transitions from a given state when performing hte action MOVE_FORWARD.

        Parameters:
            s <tuple(int)>: A length-3 tuple denoting a state (x, y, d) where d = heading of the player tank.

        Returns:
            transition_p <dict(tuple(int): float)>: A dictionary of transition probabilities based on each of the next
            states (because MOVE_FORWARD is non-deterministic in this assignment).
        """
        x, y, d = s
        # Entries are in order of MOVE_FORWARD_LEFT, MOVE_FORWARD, MOVE_FORWARD_RIGHT, TURN_LEFT, NO_CHANGE, TURN_RIGHT.
        if d == LaserTankMap.UP:
            transition_p = {(x - 1, y - 1, d): self.game_map.t_error_prob/5,
                            (x, y - 1, d): self.game_map.t_success_prob,
                            (x + 1, y - 1, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.LEFT): self.game_map.t_error_prob/5,
                            (x, y, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.RIGHT): self.game_map.t_error_prob/5}
        elif d == LaserTankMap.DOWN:
            transition_p = {(x + 1, y + 1, d): self.game_map.t_error_prob/5,
                            (x, y + 1, d): self.game_map.t_success_prob,
                            (x - 1, y + 1, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.RIGHT): self.game_map.t_error_prob/5,
                            (x, y, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.LEFT): self.game_map.t_error_prob/5}
        elif d == LaserTankMap.LEFT:
            transition_p = {(x - 1, y + 1, d): self.game_map.t_error_prob/5,
                            (x - 1, y, d): self.game_map.t_success_prob,
                            (x - 1, y - 1, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.DOWN): self.game_map.t_error_prob/5,
                            (x, y, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.UP): self.game_map.t_error_prob/5}
        # d == LaserTankMap.RIGHT
        else:
            transition_p = {(x + 1, y - 1, d): self.game_map.t_error_prob/5,
                            (x + 1, y, d): self.game_map.t_success_prob,
                            (x + 1, y + 1, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.UP): self.game_map.t_error_prob/5,
                            (x, y, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.DOWN): self.game_map.t_error_prob/5}

        return transition_p

    def apply_move_deterministic(self, s, action):
        """
        Similar to LaserMapTank.apply_move() but for deterministic only actions i.e. TURN_LEFT, TURN_RIGHT and
        SHOOT_LASER. Changes a state from (x, y, d) to (x, y, d_dash) where d_dash is the new heading based on the
        deterministic action (know that none of the deterministic actions can change (x, y)).

        Parameters:
            s <tuple(int)>: A length-3 tuple denoting a state (x, y, d) where d = heading of the player tank.
            action <str>: either TURN_LEFT ("l"), TURN_RIGHT ("r") or SHOOT_LASER ("s").

        Returns:
            <tuple(int)>: A length-3 tuple denoting the next state (x, y, d) where d = heading of the player tank.
        """
        x, y, d = s
        d_dash = d
        if action == self.game_map.TURN_LEFT:
            if d == self.game_map.UP:
                d_dash = self.game_map.LEFT
            elif d == self.game_map.DOWN:
                d_dash = self.game_map.RIGHT
            elif d == self.game_map.LEFT:
                d_dash = self.game_map.DOWN
            else:
                d_dash = self.game_map.UP
        elif action == self.game_map.TURN_RIGHT:
            if d == self.game_map.UP:
                d_dash = self.game_map.RIGHT
            elif d == self.game_map.DOWN:
                d_dash = self.game_map.LEFT
            elif d == self.game_map.LEFT:
                d_dash = self.game_map.UP
            else:
                d_dash = self.game_map.DOWN

        return x, y, d_dash

    def run_value_iteration(self):
        """
        Build a value table and a policy table, using Synchronous Value Iteration, which is stored inside self.values
        and self.policy.
        """
        # Default initialisation for VI
        self.values = [[[0 for _ in self.game_map.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]
        self.policy = [[[-1 for _ in self.game_map.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]

        # Number of total iterations
        num_iter = 0
        # Record runtime
        start_time = time.time()
        while True:
            num_iter += 1
            # Use in stopping criteria later - max difference between old and new value
            max_delta = 0
            for s in self.states:
                x, y, d = s
                # Skip states that are goal state or blocked states
                if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                    continue

                # Make clone so not mutating original game_map
                game_map = self.game_map.make_clone()
                game_map.player_x = x
                game_map.player_y = y
                game_map.player_heading = d
                transition_p = self.transition_ps[x - 1][y - 1][d]
                # Use this for determining optimal value given the optimal action
                max_action_value = -1e6
                max_action_move = None
                for action in LaserTankMap.MOVES:
                    total = 0
                    # Handle non-deterministic MOVE_FORWARD action
                    if action == LaserTankMap.MOVE_FORWARD:
                        # Compute R(s, a) rather than R(s, a, s') since possible to arrive at same state using different
                        # actions.
                        expected_reward = 0
                        # Increase excess probability for NO_CHANGE transition if unreachable state encountered
                        excess_p = 0
                        unreachable_states = []
                        # Calculate expected immediate reward for current state from all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            # Goal reward
                            if (x_dash, y_dash) == (game_map.flag_x, game_map.flag_y):
                                expected_reward += p * game_map.goal_reward
                            # Collision cost
                            elif game_map.cell_is_blocked(y_dash, x_dash):
                                expected_reward += p * game_map.collision_cost
                                excess_p += p
                                unreachable_states.append(s_dash)
                            # Game over cost
                            elif game_map.cell_is_game_over(y_dash, x_dash):
                                expected_reward += p * game_map.game_over_cost
                            # Normal move cost
                            else:
                                expected_reward += p * game_map.move_cost

                        total = expected_reward
                        # Compute expected future reward for current state over all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            if s_dash in unreachable_states:
                                continue

                            # Increased probability of NO_CHANGE transition if some s' are unreachable
                            if s == s_dash:
                                total += game_map.gamma * (p + excess_p) * self.values[x - 1][y - 1][d]
                            else:
                                total += game_map.gamma * p * self.values[x_dash - 1][y_dash - 1][d_dash]
                    # Handle deterministic actions - TURN_LEFT, TURN_RIGHT, SHOOT_LASER
                    else:
                        x_dash, y_dash, d_dash = self.apply_move_deterministic(s, action)
                        # Expected reward + discounted future reward:
                        # Since only one s' possible --> P(s'|s,a) = 1
                        total += game_map.move_cost + game_map.gamma * self.values[x_dash - 1][y_dash - 1][d_dash]

                    # Update optimal value and action
                    if total > max_action_value:
                        max_action_value = total
                        max_action_move = action

                # Update stopping criteria
                if abs(max_action_value - self.values[x - 1][y - 1][d]) > max_delta:
                    max_delta = abs(max_action_value - self.values[x - 1][y - 1][d])
                # Update optimal values and policy
                self.values[x - 1][y - 1][d] = max_action_value
                self.policy[x - 1][y - 1][d] = max_action_move

            # # Q2.c)
            # if time.time() >= start_time + 1:
            #     break

            # Q2.d)
            if num_iter == 10:
                break

        # RESULTS
        # print(self.values)
        # print(self.policy)
        print(f"Number of iterations: {num_iter}")
        print(f"Elapsed runtime: {time.time() - start_time}")

    def run_policy_iteration(self):
        """
        Build a value table and a policy table, using Modified Policy Iteration, which is stored inside self.values and
        self.policy.
        """
        # Default initialisation for PI
        self.values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]
        self.policy = [[[LaserTankMap.MOVE_FORWARD for _ in LaserTankMap.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]
        # Set invalid action for goal and blocked states
        for y in range(1, self.game_map.y_size - 1):
            for x in range(1, self.game_map.x_size - 1):
                if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                    for d in range(4):
                        self.policy[x - 1][y - 1][d] = -1

        # Number of total iterations
        num_iter = 0
        # Record runtime
        start_time = time.time()
        while True:
            num_iter += 1
            # POLICY EVALUATION
            for k in range(10):
                for s in self.states:
                    x, y, d = s
                    # Skip states that are goal state or blocked states
                    if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                        continue

                    # Make clone so not mutating original game_map
                    game_map = self.game_map.make_clone()
                    game_map.player_x = x
                    game_map.player_y = y
                    game_map.player_heading = d
                    transition_p = self.transition_ps[x - 1][y - 1][d]
                    # Use current policy to determine optimal action given state
                    action = self.policy[x - 1][y - 1][d]
                    total = 0
                    # Handle non-deterministic MOVE_FORWARD action
                    if action == LaserTankMap.MOVE_FORWARD:
                        # Compute R(s, a) rather than R(s, a, s') since possible to arrive at same state using different
                        # actions.
                        expected_reward = 0
                        # Increase excess probability for NO_CHANGE transition if unreachable state encountered
                        excess_p = 0
                        unreachable_states = []
                        # Calculate expected immediate reward for current state from all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            # Goal reward
                            if (x_dash, y_dash) == (game_map.flag_x, game_map.flag_y):
                                expected_reward += p * game_map.goal_reward
                            # Collision cost
                            elif game_map.cell_is_blocked(y_dash, x_dash):
                                expected_reward += p * game_map.collision_cost
                                excess_p += p
                                unreachable_states.append(s_dash)
                            # Game over cost
                            elif game_map.cell_is_game_over(y_dash, x_dash):
                                expected_reward += p * game_map.game_over_cost
                            # Normal move cost
                            else:
                                expected_reward += p * game_map.move_cost

                        total = expected_reward
                        # Compute expected future reward for current state over all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            if s_dash in unreachable_states:
                                continue

                            # Increased probability of NO_CHANGE transition if some s' are unreachable
                            if s == s_dash:
                                total += game_map.gamma * (p + excess_p) * self.values[x - 1][y - 1][d]
                            else:
                                total += game_map.gamma * p * self.values[x_dash - 1][y_dash - 1][d_dash]
                    # Handle deterministic actions - TURN_LEFT, TURN_RIGHT, SHOOT_LASER
                    else:
                        x_dash, y_dash, d_dash = self.apply_move_deterministic(s, action)
                        # Expected reward + discounted future reward:
                        # Since only one s' possible --> P(s'|s,a) = 1
                        total += game_map.move_cost + game_map.gamma * self.values[x_dash - 1][y_dash - 1][d_dash]

                    # Update optimal values
                    self.values[x - 1][y - 1][d] = total

            # POLICY IMPROVEMENT
            for s in self.states:
                x, y, d = s
                # Skip states that are goal state or blocked states
                if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                    continue

                # Make clone so not mutating original game_map
                game_map = self.game_map.make_clone()
                game_map.player_x = x
                game_map.player_y = y
                game_map.player_heading = d
                transition_p = self.transition_ps[x - 1][y - 1][d]
                # Use this for determining optimal value given the optimal action
                max_action_value = -1e6
                max_action_move = None
                for action in LaserTankMap.MOVES:
                    total = 0
                    # Handle non-deterministic MOVE_FORWARD action
                    if action == LaserTankMap.MOVE_FORWARD:
                        # Compute R(s, a) rather than R(s, a, s') since possible to arrive at same state using different
                        # actions.
                        expected_reward = 0
                        # Increase excess probability for NO_CHANGE transition if unreachable state encountered
                        excess_p = 0
                        unreachable_states = []
                        # Calculate expected immediate reward for current state from all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            # Goal reward
                            if (x_dash, y_dash) == (game_map.flag_x, game_map.flag_y):
                                expected_reward += p * game_map.goal_reward
                            # Collision cost
                            elif game_map.cell_is_blocked(y_dash, x_dash):
                                expected_reward += p * game_map.collision_cost
                                excess_p += p
                                unreachable_states.append(s_dash)
                            # Game over cost
                            elif game_map.cell_is_game_over(y_dash, x_dash):
                                expected_reward += p * game_map.game_over_cost
                            # Normal move cost
                            else:
                                expected_reward += p * game_map.move_cost

                        total = expected_reward
                        # Compute expected future reward for current state over all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            if s_dash in unreachable_states:
                                continue

                            # Increased probability of NO_CHANGE transition if some s' are unreachable
                            if s == s_dash:
                                total += game_map.gamma * (p + excess_p) * self.values[x - 1][y - 1][d]
                            else:
                                total += game_map.gamma * p * self.values[x_dash - 1][y_dash - 1][d_dash]
                    # Handle deterministic actions - TURN_LEFT, TURN_RIGHT, SHOOT_LASER
                    else:
                        x_dash, y_dash, d_dash = self.apply_move_deterministic(s, action)
                        # Expected reward + discounted future reward:
                        # Since only one s' possible --> P(s'|s,a) = 1
                        total += game_map.move_cost + game_map.gamma * self.values[x_dash - 1][y_dash - 1][d_dash]

                    # Update optimal value and action
                    if total > max_action_value:
                        max_action_value = total
                        max_action_move = action

                # Update optimal policy
                self.policy[x - 1][y - 1][d] = max_action_move

            # # Q2.c)
            # if time.time() >= start_time + 1:
            #     break

            # Q2.d)
            if num_iter == 10:
                break

        # RESULTS
        # print(self.values)
        # print(self.policy)
        print(f"Number of iterations: {num_iter}")
        print(f"Elapsed runtime: {time.time() - start_time}")

    def get_offline_value(self, state):
        """
        Get the value of the given state.

        Parameters:
            state (LaserMapTank): A LaserTankMap instance.

        Returns:
            (float): V(s).
        """
        return self.values[state.player_x - 1][state.player_y - 1][state.player_heading]

    def get_offline_policy(self, state):
        """
        Get the policy for the given state (i.e. the action that should be performed at this state).

        Parameters:
            state (LaserMapTank): A LaserTankMap instance.

        Returns:
            (str): pi(s) - an element of LaserTankMap.Moves.
        """
        return self.policy[state.player_x - 1][state.player_y - 1][state.player_heading]

    def get_mcts_policy(self, state):
        """
        Choose an action to be performed using online MCTS.

        Parameters:
            state (LaserMapTank): A LaserTankMap instance.

        Returns:
            (str): pi(s) - an element of LaserTankMap.Moves.
        """
        return self.ucb_search(state)

    def ucb_search(self, state):
        """
        Conducts Monte-Carlo Tree Search starting from the given state to find the most optimal action to perform.
        Incorporates the UCB1 bandit sampling algorithm for the selection step.

        Parameters:
            state (LaserMapTank): A LaserTankMap instance.

        Returns:
            best_action (str): pi(s) - an element of LaserTankMap.Moves.
        """
        root_node = Node(state)
        last_node = self.expand(root_node)
        reward = self.default_policy(last_node.state)
        self.backup(last_node, reward)
        for t in range(500):
            # Selection/expansion steps
            last_node = self.tree_policy(root_node)
            # Simulation step
            reward = self.default_policy(last_node.state)
            # Backpropagation step
            self.backup(last_node, reward)
        best_action = self.best_child(root_node).action
        # print(best_action)
        return best_action

    def tree_policy(self, node):
        """
        Covers the selection/expansion steps by using the UCB1 bandit sampling algorithm to either explore (expand) or
        exploit (select) in a graph structure with nodes = states and edges = actions.

        Parameters:
            node (Node): A node in the graph representing a state.

        Returns:
            (Node): Another node in the graph representing a different (or same) state.
        """
        while node.children:
            # Node not fully expanded - exploration
            if len(node.children) < 4:
                return self.expand(node)
            # Select best child of node - exploitation
            else:
                node = self.best_child(node)
        return self.expand(node)

    def expand(self, node):
        """
        Expands a given node by adding a new node to its children.

        Parameters:
            node (Node): A node in the graph representing a state.

        Returns:
            (Node): Another node in the graph representing a different (or same) state.
        """
        x = node.state.player_x
        y = node.state.player_y
        d = node.state.player_heading

        # Store all new children in a node as a list - this is because the non-deterministic action MOVE_FORWARD can
        # have up to 6 children.
        children = []
        # Select an untried action
        action = node.untried_actions.pop()
        # Handle non-deterministic MOVE_FORWARD action
        if action == LaserTankMap.MOVE_FORWARD:
            transition_p = self.transition_ps[x - 1][y - 1][d]
            for s_dash in transition_p:
                x_dash, y_dash, d_dash = s_dash
                # Blocked cells --> stay in original state
                if node.state.cell_is_blocked(y_dash, x_dash) or x == x_dash and y == y_dash and d == d_dash:
                    # Only create a new node for the original node's state if its not already included in the children.
                    if node.state not in [other_nodes.state for other_nodes in children]:
                        child = Node(node.state)
                        # Update node attributes (see Node class for more info)
                        child.action = action
                        child.parent = node
                        children.append(child)
                    else:
                        continue
                # Unblocked cells --> move to another state
                else:
                    # Make clone so not mutating original state
                    cloned_state = node.state.make_clone()
                    cloned_state.player_x = x_dash
                    cloned_state.player_y = y_dash
                    cloned_state.player_heading = d_dash
                    child = Node(cloned_state)
                    # Update node attributes (see Node class for more info)
                    child.action = action
                    child.parent = node
                    children.append(child)
        # Handle deterministic actions - TURN_LEFT, TURN_RIGHT, SHOOT_LASER
        else:
            # Make clone so not mutating original state
            cloned_state = node.state.make_clone()
            cloned_state.apply_move(action)
            child = Node(cloned_state)
            # Update node attributes (see Node class for more info)
            child.action = action
            child.parent = node
            children.append(child)

        node.children.append(children)
        # If action was MOVE_FORWARD --> possible up to 6 children, so return a random child
        return children[np.random.randint(0, len(children))]

    @staticmethod
    def best_child(node):
        """
        Finds the best child of a given node based on the child with the highest UCB value. Note that c = sqrt(2) is
        used here.

        Parameters:
            node (Node): A node in the graph representing a state.

        Returns:
            max_ucb_child (Node): The best child node in the graph representing a different (or same) state.
        """
        # Default initialisation
        max_ucb_value = -1e4
        max_ucb_child = None
        for children in node.children:
            for child in children:
                # Calculate UCB value for each child using formula and update the highest value child
                ucb_value = child.avg_reward + np.sqrt(2) + np.sqrt(np.log(node.n)/child.n)
                if ucb_value > max_ucb_value:
                    max_ucb_value = ucb_value
                    max_ucb_child = child

        return max_ucb_child

    @staticmethod
    def default_policy(state):
        """
        Covers the simulation step by simulating from the current state to a terminal state (when the game ends because
        the player lands on a game_over_cell or the goal_cell) using a default policy i.e. random actions.

        Parameters:
            state (LaserMapTank): A LaserTankMap instance.

        Returns:
            last_reward (float): The final terminal state reward (either LaserMapTank.game_over_cost or
            LaserMapTank.goal_reward).
        """
        cloned_state = state.make_clone()
        last_reward = None
        while last_reward != cloned_state.game_over_cost and last_reward != cloned_state.goal_reward:
            action = LaserTankMap.MOVES[np.random.randint(0, 4)]
            last_reward = cloned_state.apply_move(action)

        return last_reward

    @staticmethod
    def backup(node, reward):
        """
        Covers the backpropagation step by backing up the simulation result from default_policy(). Updates the average
        reward and number of times visited statistics for each node involved in the simulation until the root node is
        reached.

        Parameters:
            node (Node): A node in the graph representing a state.
            reward (float): The final terminal state reward (either LaserMapTank.game_over_cost or
            LaserMapTank.goal_reward).
        """
        while node is not None:
            node.avg_reward = (node.avg_reward * node.n + reward)/(node.n + 1)
            node.n += 1
            node = node.parent


class Node:
    def __init__(self, state):
        """
        Initialise a new node in the graph representing a complete state of the problem.

        Parameters:
            state (LaserMapTank): A LaserTankMap instance.
        """
        self.state = state
        # Action performed by parent node to reach this node
        self.action = None
        # Parent of this node
        self.parent = None
        # Children of this node:
        # Minimum 0 for terminal node and maximum 9 for all actions tried and MOVE_FORWARD yielding full 6 different
        # children.
        self.children = []
        # List of actions that have not yet been tried for this node
        self.untried_actions = [action for action in LaserTankMap.MOVES]
        # Average reward received for traversing this node
        self.avg_reward = 0
        # Can't use 0 as default value since eventually results in ZeroDivisionError during backpropagation so instead
        # use a really small value.
        self.n = state.epsilon

from laser_tank import LaserTankMap, DotDict
import time
from copy import deepcopy
import numpy as np

"""
Template file for you to implement your solution to Assignment 3. You should implement your solution by filling in the
following method stubs:
    run_value_iteration()
    run_policy_iteration()
    get_offline_value()
    get_offline_policy()
    get_mcts_policy()
    
You may add to the __init__ method if required, and can add additional helper methods and classes if you wish.

To ensure your code is handled correctly by the autograder, you should avoid using any try-except blocks in your
implementation of the above methods (as this can interfere with our time-out handling).

COMP3702 2020 Assignment 3 Support Code
"""


class Solver:
    def __init__(self, game_map):
        #
        # TODO
        # Write any environment preprocessing code you need here (e.g. storing teleport locations).
        #
        # You may also add any instance variables (e.g. root node of MCTS tree) here.
        #
        # The allowed time for this method is 1 second, so your Value Iteration or Policy Iteration implementation
        # should be in the methods below, not here.
        #

        self.game_map = game_map
        self.values = None
        self.policy = None
        self.root_node = None

        # Form all states
        self.states = []
        for x in range(1, self.game_map.x_size - 1):
            for y in range(1, self.game_map.y_size - 1):
                for heading in range(4):
                    self.states.append((x, y, heading))

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
        x, y, d = s

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
        elif d == LaserTankMap.RIGHT:
            transition_p = {(x + 1, y - 1, d): self.game_map.t_error_prob/5,
                            (x + 1, y, d): self.game_map.t_success_prob,
                            (x + 1, y + 1, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.UP): self.game_map.t_error_prob/5,
                            (x, y, d): self.game_map.t_error_prob/5,
                            (x, y, LaserTankMap.DOWN): self.game_map.t_error_prob/5}

        return transition_p

    def apply_move_deterministic(self, state, action):
        x, y, d = state
        if action == self.game_map.TURN_LEFT:
            # no collision or game over possible
            if d == self.game_map.UP:
                d = self.game_map.LEFT
            elif d == self.game_map.DOWN:
                d = self.game_map.RIGHT
            elif d == self.game_map.LEFT:
                d = self.game_map.DOWN
            else:
                d = self.game_map.UP
        elif action == self.game_map.TURN_RIGHT:
            # no collision or game over possible
            if d == self.game_map.UP:
                d = self.game_map.RIGHT
            elif d == self.game_map.DOWN:
                d = self.game_map.LEFT
            elif d == self.game_map.LEFT:
                d = self.game_map.UP
            else:
                d = self.game_map.DOWN
        return x, y, d

    def run_value_iteration(self):
        """
        Build a value table and a policy table using value iteration, and store inside self.values and self.policy.
        """
        self.values = [[[0 for _ in self.game_map.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]
        self.policy = [[[-1 for _ in self.game_map.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]

        num_runs = 0
        start_time = time.time()
        while True:
            num_runs += 1
            max_delta = 0
            for s in self.states:
                # print(f"State: {s}")
                x, y, d = s
                # Skips states that are goal state, blocked states or game-over states
                if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                    continue

                game_map = self.game_map.make_clone()
                game_map.player_x = x
                game_map.player_y = y
                game_map.player_heading = d
                transition_p = self.transition_ps[x - 1][y - 1][d]
                max_action_value = -1e6
                max_action_move = None
                for action in LaserTankMap.MOVES:
                    total = 0

                    if action == LaserTankMap.MOVE_FORWARD:
                        expected_reward = 0
                        excess_p = 0
                        unreachable_states = []
                        # Calculate expected immediate reward for current state from all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            # print(s_dash)
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
                        # Compute  expected future reward for current state over all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            if s_dash in unreachable_states:
                                continue

                            if s == s_dash:
                                total += game_map.gamma * (p + excess_p) * self.values[x - 1][y - 1][d]
                            else:
                                total += game_map.gamma * p * self.values[x_dash - 1][y_dash - 1][d_dash]
                    # Determine case for deterministic actions separately
                    else:
                        x_dash, y_dash, d_dash = self.apply_move_deterministic(s, action)
                        # Expected reward + discounted future reward
                        total += game_map.move_cost + game_map.gamma * self.values[x_dash - 1][y_dash - 1][d_dash]
                        # print(f"Action : {action}")
                        # print(f"Next state: {(game_map_clone.player_x, game_map_clone.player_y,
                        #                       game_map_clone.player_heading)}")

                    if total > max_action_value:
                        max_action_value = total
                        max_action_move = action

                if abs(max_action_value - self.values[x - 1][y - 1][d]) > max_delta:
                    max_delta = abs(max_action_value - self.values[x - 1][y - 1][d])
                self.values[x - 1][y - 1][d] = max_action_value
                self.policy[x - 1][y - 1][d] = max_action_move

                # print(f"Values: {self.values}")
                # print(f"Policy: {self.policy}")
                # print("\n")
                # time.sleep(1)

            if max_delta <= self.game_map.epsilon * 50:
                break

        # Computed values and policy
        print(self.values)
        print(self.policy)
        print(f"Number of iterations: {num_runs}")
        print(f"Elapsed runtime: {time.time() - start_time}")

    def run_policy_iteration(self):
        """
        Build a value table and a policy table using policy iteration, and store inside self.values and self.policy.
        """
        self.values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]
        self.policy = [[[LaserTankMap.MOVE_FORWARD for _ in LaserTankMap.DIRECTIONS]
                        for __ in range(1, self.game_map.y_size - 1)]
                       for ___ in range(1, self.game_map.x_size - 1)]
        for y in range(1, self.game_map.y_size - 1):
            for x in range(1, self.game_map.x_size - 1):
                if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                    for d in range(4):
                        self.policy[x - 1][y - 1][d] = -1

        num_runs = 0
        start_time = time.time()
        while True:
            num_runs += 1
            # Policy evaluation
            for k in range(10):
                for s in self.states:
                    # print(f"State: {s}")
                    x, y, d = s
                    # Skips states that are goal state, blocked states or game-over states
                    if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                        continue

                    game_map = self.game_map.make_clone()
                    game_map.player_x = x
                    game_map.player_y = y
                    game_map.player_heading = d
                    transition_p = self.transition_ps[x - 1][y - 1][d]
                    action = self.policy[x - 1][y - 1][d]
                    total = 0

                    if action == LaserTankMap.MOVE_FORWARD:
                        expected_reward = 0
                        excess_p = 0
                        unreachable_states = []
                        # Calculate expected immediate reward for current state from all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            # print(s_dash)
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
                        # Compute  expected future reward for current state over all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            if s_dash in unreachable_states:
                                continue

                            if s == s_dash:
                                total += game_map.gamma * (p + excess_p) * self.values[x - 1][y - 1][d]
                            else:
                                total += game_map.gamma * p * self.values[x_dash - 1][y_dash - 1][d_dash]
                    # Determine case for deterministic actions separately
                    else:
                        x_dash, y_dash, d_dash = self.apply_move_deterministic(s, action)
                        # Expected reward + discounted future reward
                        total += game_map.move_cost + game_map.gamma * self.values[x_dash - 1][y_dash - 1][d_dash]
                        # print(f"Action : {action}")
                        # print(f"Next state: {(game_map_clone.player_x, game_map_clone.player_y,
                        #                       game_map_clone.player_heading)}")

                    self.values[x - 1][y - 1][d] = total

            # Policy improvement
            new_policy = [[state[:] for state in states] for states in self.policy]
            for s in self.states:
                # print(f"State: {s}")
                x, y, d = s
                # Skips states that are goal state, blocked states or game-over states
                if x == self.game_map.flag_x and y == self.game_map.flag_y or self.game_map.cell_is_blocked(y, x):
                    continue

                game_map = self.game_map.make_clone()
                game_map.player_x = x
                game_map.player_y = y
                game_map.player_heading = d
                transition_p = self.transition_ps[x - 1][y - 1][d]
                max_action_value = -1e6
                max_action_move = None
                for action in LaserTankMap.MOVES:
                    total = 0

                    if action == LaserTankMap.MOVE_FORWARD:
                        expected_reward = 0
                        excess_p = 0
                        unreachable_states = []
                        # Calculate expected immediate reward for current state from all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            # print(s_dash)
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
                        # Compute  expected future reward for current state over all possible next states
                        for s_dash in transition_p:
                            x_dash, y_dash, d_dash = s_dash
                            p = transition_p[s_dash]
                            if s_dash in unreachable_states:
                                continue

                            if s == s_dash:
                                total += game_map.gamma * (p + excess_p) * self.values[x - 1][y - 1][d]
                            else:
                                total += game_map.gamma * p * self.values[x_dash - 1][y_dash - 1][d_dash]
                    # Determine case for deterministic actions separately
                    else:
                        x_dash, y_dash, d_dash = self.apply_move_deterministic(s, action)
                        # Expected reward + discounted future reward
                        total += game_map.move_cost + game_map.gamma * self.values[x_dash - 1][y_dash - 1][d_dash]
                        # print(f"Action : {action}")
                        # print(f"Next state: {(game_map_clone.player_x, game_map_clone.player_y,
                        #                       game_map_clone.player_heading)}")

                    if total > max_action_value:
                        max_action_value = total
                        max_action_move = action
                    # print(total)

                new_policy[x - 1][y - 1][d] = max_action_move

            if new_policy == self.policy:
                break

            self.policy = new_policy


        # Computed values and policy
        print(self.values)
        print(self.policy)
        print(f"Number of iterations: {num_runs}")
        print(f"Elapsed runtime: {time.time() - start_time}")

    def get_offline_value(self, state):
        """
        Get the value of this state.
        :param state: a LaserTankMap instance
        :return: V(s) [a floating point number]
        """
        return self.values[state.player_x - 1][state.player_y - 1][state.player_heading]

    def get_offline_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """
        return self.policy[state.player_x - 1][state.player_y - 1][state.player_heading]

    def get_mcts_policy(self, state):
        """
        Choose an action to be performed using online MCTS.
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """

        #
        # TODO
        # Write your Monte-Carlo Tree Search implementation here.
        #
        # Each time this method is called, you are allowed up to [state.time_limit] seconds of compute time - make sure
        # you stop searching before this time limit is reached.
        #

        return self.ucb_search(state)

    def ucb_search(self, state):
        self.root_node = Node(state)
        for t in range(100):
            last_node = self.tree_policy(self.root_node)
            reward = self.default_policy(last_node.state)
            # reward = self.default_policy(self.root_node)
            self.backup(last_node, reward)
        best_action = self.best_child(self.root_node).action
        print(best_action)
        return best_action

    @staticmethod
    def terminal_state(node):
        x = node.state.player_x
        y = node.state.player_y
        if x == node.state.flag_x and y == node.state.flag_y:
            return True
        elif node.state.cell_is_game_over(y, x):
            return True

        return False

    # SELECTION (INCL. EXPANSION)
    def tree_policy(self, node):
        while node == self.root_node or node.children:
            # Node not fully expanded
            if len(node.children) < 4:
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    # EXPANSION
    def expand(self, node):
        x = node.state.player_x
        y = node.state.player_y
        d = node.state.player_heading
        # All possible next states if action is MOVE_FORWARD
        transition_p = self.transition_ps[x - 1][y - 1][d]

        children = []
        action = node.untried_actions.pop()
        if action == LaserTankMap.MOVE_FORWARD:
            for s_dash in transition_p:
                # TODO
                x_dash, y_dash, d_dash = s_dash

                # Blocked cells --> return to original state
                if node.state.cell_is_blocked(y_dash, x_dash):
                    if node.state not in [other_nodes.state for other_nodes in children]:
                        child = Node(node.state)
                        child.action = action
                        child.parent = node
                        children.append(child)
                    else:
                        continue
                # Move to another state
                else:
                    cloned_state = node.state.make_clone()
                    cloned_state.player_x = x_dash
                    cloned_state.player_y = y_dash
                    cloned_state.player_heading = d_dash
                    child = Node(cloned_state)
                    child.action = action
                    child.parent = node
                    children.append(child)
        # MOVE_LEFT, MOVE_RIGHT, SHOOT_LASER
        else:
            cloned_state = node.state.make_clone()
            cloned_state.apply_move(action)
            child = Node(cloned_state)
            child.action = action
            child.parent = node
            children.append(child)

        node.children.append(children)
        return children[np.random.randint(0, len(children))]

    @staticmethod
    def best_child(node):
        max_ucb_value = -1e6
        max_ucb_child = None
        print(node.children)
        for children in node.children:
            for child in children:
                ucb_value = child.avg_reward + np.sqrt(2) + np.sqrt(np.log(node.n)/child.n)
                print(ucb_value)
                if ucb_value > max_ucb_value:
                    max_ucb_value = ucb_value
                    max_ucb_child = child

        # time.sleep(10)
        return max_ucb_child

    @staticmethod
    def default_policy(state):
        cloned_state = state.make_clone()
        last_reward = None
        total_reward = 0
        while last_reward != cloned_state.game_over_cost and last_reward != cloned_state.goal_reward:
            action = LaserTankMap.MOVES[np.random.randint(0, 4)]
            last_reward = cloned_state.apply_move(action)
            total_reward += last_reward

        return total_reward

    # def default_policy(self, node):
    #     cloned_state = node.state.make_clone()
    #     total_reward = 0
    #     while node.children:
    #         action = self.best_child(node).action
    #         total_reward += cloned_state.apply_move(action)
    #
    #     final_action =  LaserTankMap.MOVES[np.random.randint(0, 4)]
    #     total_reward += cloned_state.apply_move(action)
    #
    #     return total_reward

    @staticmethod
    def backup(node, reward):
        while node is not None:
            node.avg_reward = (node.avg_reward * node.n + reward)/(node.n + 1)
            node.n += 1
            node = node.parent


class Node:
    def __init__(self, state):
        self.state = state
        self.action = None
        self.parent = None
        self.children = []
        self.untried_actions = [action for action in LaserTankMap.MOVES]
        self.avg_reward = 0
        self.n = 1

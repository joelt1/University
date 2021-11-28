import sys
from problem_spec import ProblemSpec
import angle
from robot_config import *
from tester import *
# from visualiser import Visualiser
import math
import heapq
import numpy as np
import time


def manhattan_dist(p, q):
    """
    A heuristic for the A* search algorithm. Finds the sum of the
    horizontal and vertical distances between any two points on a grid.

    Parameters:
        p (tuple<float>): first point's x- and y- coordinates.
        p (tuple<float>): second point's x- and y- coordinates.

    Returns:
        (float): Sum of horizontal and vertical distances between the two
            points.
    """
    p_x, p_y = p
    q_x, q_y = q
    return abs(p_x - q_x) + abs(p_y - q_y)


class GraphNode:
    """
    Class representing a node in the state graph.
    """
    def __init__(self, spec, config):
        """
        Create a new graph node object for the given config.

        Neighbors are added by appending to self.neighbors after creating each
        new GraphNode.

        Parameters:
            spec (ProblemSpec): A ProblemSpec object detailing the provided
                problem (test case).
            config (RobotConfig): RobotConfig object to be stored in this node.
        """
        self.spec = spec
        self.config = config
        self.neighbors = []
        self.cost = 0

    def __eq__(self, other):
        """
        Override the equality operator to check when 2 states are equal
        (self == obj). Equality based on checking config equality using
        test_config_equality.

        Parameters:
            other (Node): Another node in the search tree.
        Returns:
            (bool): True if two states are equal, false otherwise.
        """
        return test_config_equality(self.config, other.config, self.spec)

    def __hash__(self):
        """
        Override hash function to be able to store previously visited nodes in
        the search tree in a dictionary instead of a list.
        """
        return hash(tuple(self.config.points))

    def __lt__(self, obj):
        """
        Override the less than function for the priority queue used in the A*
        search algorithm.

        Parameters:
            obj (Node): A node in the search tree.

        Returns:
            (bool): True if a node's cost is less than another node, false
                otherwise. Note that for A*, costs are the sum of a path from
                the initial state to a node (UCS) and the estimated cost from
                the end of the path to the goal (GBFS).
        """
        return self.cost < obj.cost

    def get_successors(self):
        """
        Retrieves all neighbour nodes of a given node.

        Returns:
            (list<GraphNode>): List of all neighbour nodes of the given node.
        """
        return self.neighbors

    @staticmethod
    def add_connection(n1, n2):
        """
        Creates a neighbor connection between the 2 given nodes (GraphNode
        objects).

        Parameters:
            n1 (GraphNode): A GraphNode object
            n2 (GraphNode): A different GraphNode object
        """
        n1.neighbors.append(n2)
        n2.neighbors.append(n1)


def find_graph_path(spec, start_node, end_node):
    """
    This method performs an A* search of the state graph and returns a list of
    configs which form a path through the state graph between the initial and
    the goal. Note that this path does not satisfy the primitive step
    requirement, this will be handled through interpolation between all the
    configs in the returned list.

    Parameters:
        spec (ProblemSpec): A ProblemSpec object detailing the provided problem
            (test case).
        init_node (GraphNode): The initial node for the initial configuration
        end_ndoe (GraphNode): The end node for the end configuration

    Returns:
        (list<RobotConfig>): List of robot configurations from the start node
            to the end node.
    """
    # Search the graph
    container = [start_node]
    heapq.heapify(container)
    # Here, each key is a graph node, each value is the list of configs visited
    # on the path to the graph node.
    visited = {start_node: [start_node.config]}

    while container:
        node = heapq.heappop(container)

        # Found path from start to end
        if test_config_equality(node.config, end_node.config, spec):
            return visited[node]

        successors = node.get_successors()
        for suc in successors:
            if suc not in visited:
                suc.cost += node.cost
                heapq.heappush(container, suc)
                visited[suc] = visited[node] + [suc.config]

    return None


def generate_sample(config, spec, min_lengths, max_lengths):
    """
    Uses uniform random sampling to generate a random number for each dimension
    of the C-space, uses them to create a robot configuration and tests if this
    config is in collision.

    Parameters:
        config (RobotConfig): A robot configuration, used here for retrieving
            intial details e.g. intial x and y, etc.
        spec (ProblemSpec): A ProblemSpec object detailing the provided problem
            (test case).
        min_lengths (list<float>): The minimum possible lengths for each arm.
        max_lengths (list<float>): The maximum possible lengths for each arm.

    Returns:
        sample_config (RobotConfig): A valid robot configuration sample.
    """
    lengths = []
    angles = []
    # Randomly generate all robot arm lengths and angles
    for i in range(len(config.lengths)):
        lengths.append(np.random.uniform(min_lengths[i], max_lengths[i]))
        angles.append(Angle(radians=np.random.uniform(0, 2*np.pi)))

    # Make robot configuration accordingly depending on whether end effector 1
    # is grappled or not.
    if config.ee1_grappled:
        (eex, eey) = config.get_ee1()
        sample_config = make_robot_config_from_ee1(eex, eey, angles, lengths,
            config.ee1_grappled, config.ee2_grappled)
    else:
        (eex, eey) = config.get_ee2()
        sample_config = make_robot_config_from_ee2(eex, eey, angles, lengths,
            config.ee1_grappled, config.ee2_grappled)

    # Full collision check
    if individual_config_collision(sample_config, spec, debug=False):
        return None

    return sample_config


def generate_bridge_sample(config, end_point, spec, min_lengths, max_lengths):
    """
    Similar to generate_sample but only generates the first n-1 links randomly.
    Attempts to make a bridge (between two different grapple points) by forcing
    the last link to become attached to the second grapple point. In doing
    this, we find the angle and length required for this last link.

    Parameters:
        config (RobotConfig): A robot configuration, used here for retrieving
            intial details e.g. intial x and y, etc.
        end_point (tuple<float>): The next grapple point's location.
        spec (ProblemSpec): A ProblemSpec object detailing the provided problem
            (test case).
        min_lengths (list<float>): The minimum possible lengths for each arm.
        max_lengths (list<float>): The maximum possible lengths for each arm.

    Returns:
        sample_config (RobotConfig): A valid robot configuration sample.
    """
    lengths = []
    angles = []
    points = []
    eex, eey = config.get_ee1() if config.ee1_grappled else config.get_ee2()
    points.append((eex, eey))
    net_angle = Angle(radians=0)

    # Randomly generate n-1 links
    for i in range(len(config.lengths) - 1):
        x, y = points[i]
        new_length = np.random.uniform(min_lengths[i], max_lengths[i])
        new_angle = Angle(radians=np.random.uniform(0, 2*np.pi))
        x_new = x + (new_length * math.cos((net_angle + new_angle).in_radians()))
        y_new = y + (new_length * math.sin((net_angle + new_angle).in_radians()))

        lengths.append(new_length)
        angles.append(new_angle)
        points.append((x_new, y_new))
        net_angle += new_angle

    x, y = points[-1]
    end_x, end_y = end_point
    # Store delta distances between last joint and next grapple point
    delta_x = end_x - x
    delta_y = end_y - y

    # Find length between last joint and next grapple point
    lengths.append(np.sqrt(delta_x**2 + delta_y**2))
    # Use this as a (brute-force) way to verify correct angle is calculated.
    # Please see Report Question 3 for more detail.
    angle_fixes = [np.pi - math.atan(delta_y/delta_x) - net_angle.in_radians(),
                   -(np.pi - math.atan(delta_y/delta_x) - net_angle.in_radians()),
                   np.pi + math.atan(delta_y/delta_x) - net_angle.in_radians(),
                   -(np.pi + math.atan(delta_y/delta_x) - net_angle.in_radians()),

                   math.atan(delta_y/delta_x) - net_angle.in_radians(),
                   math.atan(delta_y/delta_x) + net_angle.in_radians(),
                   -math.atan(delta_y/delta_x) - net_angle.in_radians(),
                   -math.atan(delta_y/delta_x) + net_angle.in_radians()]

    # Verify correct angle is used
    for angle in angle_fixes:
        angles.append(Angle(radians=angle))
        if config.ee1_grappled:
            sample_config = make_robot_config_from_ee1(eex, eey, angles[:],
                lengths[:], config.ee1_grappled, config.ee2_grappled)
        else:
            sample_config = make_robot_config_from_ee2(eex, eey, angles[:],
                lengths[:], config.ee1_grappled, config.ee2_grappled)

        # Need to make sure calculated angle together with length will form the
        # correct.
        # last link between the last joint and next grapple point.
        if sample_config.points[-1] == end_point and not \
                individual_config_collision(sample_config, spec, debug=False):
            return sample_config

        # Remove appended calculated angle and try again
        angles.pop()

    return None


def interpolate_path(config_1, config_2, spec):
    """
    Outputs a list of robot configurations where every 2 consecutive
    configurations are <= 1 primitive step apart

    Parameters:
        config_1 (RobotConfig): Starting configuration
        config_2 (RobotCofnig): Ending configuration
        spec (ProblemSpec): A ProblemSpec object detailing the provided problem
            (test case).

    Returns:
        (list<RobotConfig>): The interpolated path as a list of robot
            configurations between the start and end configurations.
    """
    configs = [config_1]
    # Keep interpolating until last config in configs matches the second config
    while test_config_equality(configs[-1], config_2, spec) is not True:
        # eex, eey remains fixed depending on which ee is grappled
        (eex, eey) = config_1.get_ee1() if config_1.ee1_grappled else \
            config_1.get_ee2()

        # First interpolate angles
        new_config_angles = configs[-1].ee1_angles[:] if config_1.ee1_grappled \
            else configs[-1].ee2_angles[:]
        config_2_angles = config_2.ee1_angles if config_1.ee1_grappled else \
            config_2.ee2_angles
        for i in range(len(new_config_angles)):
            # Reduce step size for fine-tuning the interpolation once close/near
            # enough
            if abs(new_config_angles[i].in_radians() - config_2_angles[i].in_radians()) < 0.001:
                if new_config_angles[i] > config_2_angles[i]:
                    new_config_angles[i] -= spec.TOLERANCE/3
                elif new_config_angles[i] < config_2_angles[i]:
                    new_config_angles[i] += spec.TOLERANCE/3
            else:
                if new_config_angles[i] > config_2_angles[i]:
                    new_config_angles[i] -= 0.001
                elif new_config_angles[i] < config_2_angles[i]:
                    new_config_angles[i] += 0.001

        # Then interpolate lengths
        new_config_lengths = configs[-1].lengths[:]
        config_2_lengths = config_2.lengths
        for i in range(len(new_config_lengths)):
            # Reduce step size for fine-tuning the interpolation once close/near
            # enough
            if abs(new_config_lengths[i] - config_2_lengths[i]) < 0.001:
                if new_config_lengths[i] > config_2_lengths[i]:
                    new_config_lengths[i] -= spec.TOLERANCE/3
                elif new_config_lengths[i] < config_2_lengths[i]:
                    new_config_lengths[i] += spec.TOLERANCE/3
            else:
                if new_config_lengths[i] > config_2_lengths[i]:
                    new_config_lengths[i] -= 0.001
                elif new_config_lengths[i] < config_2_lengths[i]:
                    new_config_lengths[i] += 0.001

        if config_1.ee1_grappled:
            configs.append(make_robot_config_from_ee1(eex, eey,
                new_config_angles[:], new_config_lengths[:],
                config_1.ee1_grappled, config_1.ee2_grappled))
        else:
            configs.append(make_robot_config_from_ee2(eex, eey,
                new_config_angles[:], new_config_lengths[:],
                config_1.ee1_grappled, config_1.ee2_grappled))

    return configs


def individual_config_collision(config, spec, debug=False):
    """
    Checks whether a single robot configuration satisfies a collision test.
    Multiple functions from tester.py are used here. Checks include -
    environment bounds, angle constraints, length constraints, grapple point
    constraints, collision with another link in the arm and obstacle collision.

    Parameters:
        config (RobotConfig): A robot configuration.
        spec (ProblemSpec): A ProblemSpec object detailing the provided problem
            (test case).
        debug (bool): Flag that states the failed test when one of the above
            tests fail.

    Returns:
        (bool): True if passes the collision check, false otherwise.
    """
    if not test_environment_bounds(config):
        if debug:
            print("Fail collision check - test_environment_bounds")
        return True
    elif not test_angle_constraints(config, spec):
        if debug:
            print("Fail collision check - test_angle_constraints")
        return True
    elif not test_length_constraints(config, spec):
        if debug:
            print("Fail collision check - test_length_constraints")
        return True
    elif not test_grapple_point_constraint(config, spec):
        if debug:
            print("Fail collision check - test_grapple_point_constraint")
        return True
    elif not test_self_collision(config, spec):
        if debug:
            print("Fail collision check - test_self_collision")
        return True
    elif not test_obstacle_collision(config, spec, spec.obstacles):
        if debug:
            print("Fail collision check - test_obstacle_collision")
        return True

    return False


def path_collision(configs, spec, debug=False):
    """
    Collision tests an entire (interpolated) path of robot configurations by
        collision testing each
    robot configuration individually. Uses recursion to check midpoints for
        (expected) faster
    performance.

    Parameters:
        configs (list<RobotConfig>): A list of (interpolated) robot
            configurations.
        spec (ProblemSpec): A ProblemSpec object detailing the provided problem
            (test case).
        debug (bool): Flag that states the failed test when one of the above
            tests fail.

    Returns:
        (bool): True if passes the collision check, false otherwise.
    """
    # Two base cases
    if len(configs) == 0:
        return False

    i = len(configs)//2
    if individual_config_collision(configs[i], spec, debug=debug):
        return True

    # Recursive case
    return path_collision(configs[:i - 1], spec, debug) or \
        path_collision(configs[i + 1:], spec, debug)


def pass_distance_check(config_1, config_2):
    """
    A distance check to avoid performing path interpolation and checking path
    collision if it is expected that the interpolated path will be too long
    suggesting that there is higher likelihood of collision somehwere. The
    distance is based on the differences between two configs' angles.

    Parameters:
        config_1 (RobotConfig): Starting configuration.
        config_2 (RobotCofnig): Ending configuration.
    """
    # Use np.array for vector subtraction below
    if config_1.ee1_grappled:
        config_1_angles = np.array([angle.in_radians() for angle in
            config_1.ee1_angles])
        config_2_angles = np.array([angle.in_radians() for angle in
            config_2.ee1_angles])
    else:
        config_1_angles = np.array([angle.in_radians() for angle in
            config_1.ee2_angles])
        config_2_angles = np.array([angle.in_radians() for angle in
            config_2.ee2_angles])

    # Add up the absolute differences between the angles.
    distance = sum(abs(config_1_angles - config_2_angles))
    min_prim_steps = distance/0.001
    return min_prim_steps <= 3000


def find_path(start_node, end_node, spec, min_lengths, max_lengths):
    """
    Finds and outputs a valid path consisting of interpolated robot
    configurations between the starting node's configuration and the ending
    node's configuration.

    Parameters:
        start_node (GraphNode): The starting node in the search graph.
        end_node (GraphNode): The ending node in the search graph.
        spec (ProblemSpec): A ProblemSpec object detailing the provided problem
            (test case).
        min_lengths (list<float>): The minimum possible lengths for each arm.
        max_lengths (list<float>): The maximum possible lengths for each arm.

    Returns:
        steps (list<RobotConfig>): A valid path consisting of interpolated robot
            configurations between the starting node's configuration and the
            ending node's configuration.
    """
    nodes = [start_node, end_node]
    # Use this for retrieving the solution, explained in more detail below.
    interpolation_order_dict = {start_node.config: [], end_node.config: []}
    steps = []

    # Time the runtime
    start_time = time.time()
    while True:
        # Generate 10 new samples and if valid, try to add them to the search
        # graph.
        new_nodes = []
        for n in range(10):
            sample_config = generate_sample(start_node.config, spec,
                min_lengths, max_lengths)

            if sample_config is not None:
                new_node = GraphNode(spec, sample_config)

                # Check if new node and existing nodes can be neighbours through
                # checking for a valid path after path interpolation between the
                # two nodes' configurations.
                for existing_node in nodes:
                    existing_config = existing_node.config

                    if pass_distance_check(existing_config, sample_config):
                        path_configs = interpolate_path(existing_config,
                            sample_config, spec)

                        if path_configs and not path_collision(path_configs,
                                spec, debug=False):
                            GraphNode.add_connection(new_node, existing_node)

                            interpolation_order_dict[sample_config] = []
                            interpolation_order_dict[existing_config].append(sample_config)

                            # High probability a randomly sampled config will
                            # not already exist, so add to list of nodes.
                            if new_node not in new_nodes:
                                # Calculate manhattan distance for each of the
                                # joint positions between the new node and end
                                # node to be used in A* search in the search graph.
                                new_node.cost = sum([manhattan_dist(sample_config.points[i], end_node.config.points[i])
                                                     for i in range(len(sample_config.points))])
                                new_nodes.append(new_node)

        nodes += new_nodes

        print(f"Number of nodes in search graph for current bridge: {len(nodes)}")
        solution_configs = find_graph_path(spec, start_node, end_node)
        if solution_configs is not None:
            print("Bridge found!")
            print("START: " + str(start_node.config))
            print("END: " + str(end_node.config))

            for i in range(len(solution_configs) - 1):
                # Since need to know the original order in which path
                # interpolation was conducted, use the dictionary to verify the
                # order. This is done because swapping the configs and calling
                # interpolate_path will not yield the same configs (likely to
                # suffer path collision).
                if solution_configs[i + 1] not in interpolation_order_dict[solution_configs[i]]:
                    # Reverse the interpolated path if the next config was not
                    # in the list of configs for the current config - one way to
                    # guarantee the correct solution.
                    path_configs = interpolate_path(solution_configs[i + 1],
                        solution_configs[i], spec)[::-1]
                else:
                    path_configs = interpolate_path(solution_configs[i],
                        solution_configs[i + 1], spec)

                steps += path_configs
            break

    end_time = time.time()
    print(f"Elapsed runtime: {end_time - start_time}")
    # Visualiser(spec, steps)
    return steps


def main(arglist):
    """
    Given the initial and goal configurations of the Canadarm robotic arm, as
    well as a map of the environment, this finds a valid path from the initial
    to the goal configurations. A valid path satisfies the path collision test.
    For cases with multiple grapple points, a valid path is generated from the
    initial configuration to a grapple point as a bridge, then from the current
    grapple point to the next, etc. until the final grapple point to the end
    configuration.

    Parameters:
        arglist (list<str>): List of arguments provided during program runtime.
    """
    input_file = arglist[0]
    output_file = arglist[1]

    spec = ProblemSpec(input_file)
    min_lengths = spec.min_lengths
    max_lengths = spec.max_lengths

    init_node = GraphNode(spec, spec.initial)
    goal_node = GraphNode(spec, spec.goal)
    steps = []
    bridge_points = spec.grapple_points

    # Check single vs. multiple grapple points
    if len(bridge_points) == 1:
        steps += find_path(init_node, goal_node, spec, min_lengths, max_lengths)
    else:
        previous_node = init_node
        for i in range(len(bridge_points)):
            start_node = previous_node
            end_node = None
            if i == len(bridge_points) - 1:
                end_node = goal_node
            # Need to generate a valid end node configuration (where the
            # opposite end effector is positioned over another grapple point).
            else:
                while end_node is None:
                    end_node = generate_bridge_sample(start_node.config,
                        bridge_points[i + 1], spec, min_lengths, max_lengths)
                end_node = GraphNode(spec, end_node)

            # Visualiser(spec, [end_node.config]*10)
            steps += find_path(start_node, end_node, spec, min_lengths, max_lengths)

            # Need to make a new robot configuration where the angles and
            # lengths are correct and the grappled end effector's status is
            # swapped with the non-grappled end effector. Essentially an
            # orientation change.
            end_config = end_node.config
            if end_config.ee1_grappled:
                eex, eey = end_config.get_ee2()
                previous_config = make_robot_config_from_ee2(eex, eey,
                    end_config.ee2_angles, end_config.lengths,
                    ee1_grappled=False, ee2_grappled=True)
            else:
                eex, eey = end_node.config.get_ee1()
                previous_config = make_robot_config_from_ee1(eex, eey,
                    end_config.ee1_angles, end_config.lengths,
                    ee1_grappled=True, ee2_grappled=False)

            previous_node = GraphNode(spec, previous_config)

    if len(arglist) > 1:
        write_robot_config_list_to_file(output_file, steps)

    # v = Visualiser(spec, steps)


if __name__ == '__main__':
    main(sys.argv[1:])

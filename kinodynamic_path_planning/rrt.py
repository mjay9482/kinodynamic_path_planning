import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.children = []
        self.parent = None

class RRTAlgorithm:
    def __init__(self, start, goal, num_iterations, grid, step_size):
        self.start_node = Node(start[0], start[1])
        self.goal_node = Node(goal[0], goal[1])
        self.random_tree = self.start_node  # tree root
        self.iterations = num_iterations
        self.grid = grid
        self.rho = step_size
        self.waypoints = []

    def unit_vector(self, start_node, end_point):
        vec = np.array([end_point[0] - start_node.x, end_point[1] - start_node.y])
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def sample_point(self):
        x = random.randint(0, self.grid.shape[1] - 1)
        y = random.randint(0, self.grid.shape[0] - 1)
        return np.array([x, y])

    def steer_to_point(self, start_node, end_point):
        direction = self.unit_vector(start_node, end_point)
        new_x = start_node.x + self.rho * direction[0]
        new_y = start_node.y + self.rho * direction[1]
        # Clamp to grid boundaries
        new_x = min(max(new_x, 0), self.grid.shape[1] - 1)
        new_y = min(max(new_y, 0), self.grid.shape[0] - 1)
        return np.array([new_x, new_y])

    def is_in_obstacle(self, start_node, end_point):
        """
        Check along the path from start_node to end_point for collision.
        Assumes grid cells with value 0 are obstacles.
        """
        direction = self.unit_vector(start_node, end_point)
        steps = int(self.rho)
        for i in range(steps):
            test_x = int(round(start_node.x + i * direction[0]))
            test_y = int(round(start_node.y + i * direction[1]))
            # Check if out of bounds
            if (test_x < 0 or test_x >= self.grid.shape[1] or 
                test_y < 0 or test_y >= self.grid.shape[0]):
                return True
            # Check if cell is an obstacle
            if self.grid[test_y, test_x] == 0:
                return True
        return False

    # def distance(self, node, point):
    #     return np.sqrt((node.x - point[0])**2 + (node.y - point[1])**2)
    def distance(self, node, point):
        # If 'point' has attribute 'x', assume it's a Node.
        if hasattr(point, 'x'):
            px = point.x
            py = point.y
        else:
            # Otherwise, assume it's indexable.
            px = point[0]
            py = point[1]
        return np.sqrt((node.x - px)**2 + (node.y - py)**2)

    def goal_found(self, point):
        """Return True if point is within step distance of the goal."""
        return self.distance(self.goal_node, point) <= self.rho

    def find_nearest(self, current_node, point, best_node=None, best_dist=None):
        """Recursively traverse the tree to find the nearest node to 'point'."""
        if best_node is None:
            best_node = current_node
            best_dist = self.distance(current_node, point)
        else:
            d = self.distance(current_node, point)
            if d < best_dist:
                best_node = current_node
                best_dist = d
        for child in current_node.children:
            best_node, best_dist = self.find_nearest(child, point, best_node, best_dist)
        return best_node, best_dist

    def plan(self, ax, writer=None):
        """
        Run the RRT algorithm for the specified iterations.
        As soon as a new node is found that is near the goal, stop the iterations,
        connect the goal node through that candidate, and return the final path.
        """
        best_candidate = None
        for i in range(self.iterations):
            # Sample a random point
            rand_point = self.sample_point()
            # Find the nearest node in the tree
            nearest_node, _ = self.find_nearest(self.random_tree, rand_point)
            # Steer toward the random point
            new_point = self.steer_to_point(nearest_node, rand_point)
            # Check for collision along the path
            if not self.is_in_obstacle(nearest_node, new_point):
                # Create a new node and attach it to the tree
                new_node = Node(new_point[0], new_point[1])
                new_node.parent = nearest_node
                nearest_node.children.append(new_node)
                # Visualize the edge
                ax.plot([nearest_node.x, new_node.x],
                        [nearest_node.y, new_node.y],
                        'g--', linewidth=1)
                plt.pause(0.05)
                if writer is not None:
                    writer.grab_frame()
                # If this new node is near the goal, record it and break
                if self.goal_found(new_point):
                    best_candidate = new_node
                    print("Goal reached at iteration:", i)
                    break

        # If a candidate was found, connect the goal through it.
        if best_candidate is not None:
            self.goal_node.parent = best_candidate
            best_candidate.children.append(self.goal_node)
            # Visualize the final connection to the goal
            ax.plot([best_candidate.x, self.goal_node.x],
                    [best_candidate.y, self.goal_node.y],
                    'b--', linewidth=1)
            plt.pause(0.001)
            if writer is not None:
                writer.grab_frame()
            path = self.retrace_path(self.goal_node)
            return path
        else:
            print("Goal not reached!")
            return None
        
    def retrace_path(self, goal):
        """Retrace the path from goal node back to the start."""
        path = []
        current = goal
        while current is not None:
            path.insert(0, np.array([current.x, current.y]))
            current = current.parent
        return path

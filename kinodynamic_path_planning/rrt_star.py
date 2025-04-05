import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.children = []
        self.cost = 0  # Cost from start to this node

class RRTStarAlgorithm:
    def __init__(self, start, goal, num_iterations, grid, step_size):
        self.start_node = Node(start[0], start[1])
        self.goal_node = Node(goal[0], goal[1])
        self.grid = grid
        self.rho = step_size
        self.num_iterations = num_iterations
        self.nodes = [self.start_node]
        self.neigh_radius = self.rho * 2   # Neighborhood radius for rewiring
    
    def sample_point(self):
        # Sample a random point within grid bounds
        x = random.randint(0, self.grid.shape[1] - 1)
        y = random.randint(0, self.grid.shape[0] - 1)
        return np.array([x, y])
    
    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def unit_vector(self, start_node, end_point):
        vec = np.array([end_point[0] - start_node.x, end_point[1] - start_node.y])
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def steer_to_point(self, nearest_node, random_point):
        # Steer from nearest_node towards random_point by step size rho
        u = self.unit_vector(nearest_node, random_point)
        new_x = nearest_node.x + self.rho * u[0]
        new_y = nearest_node.y + self.rho * u[1]
        # Ensure new point is within grid bounds
        new_x = min(max(new_x, 0), self.grid.shape[1] - 1)
        new_y = min(max(new_y, 0), self.grid.shape[0] - 1)
        return np.array([new_x, new_y])
    
    def is_in_obstacle(self, start_node, end_point):
        # Check along the path from start_node to end_point for collision
        u = self.unit_vector(start_node, end_point)
        steps = int(self.rho)
        for i in range(steps):
            test_x = int(round(start_node.x + i * u[0]))
            test_y = int(round(start_node.y + i * u[1]))
            if test_x < 0 or test_x >= self.grid.shape[1] or test_y < 0 or test_y >= self.grid.shape[0]:
                return True
            if self.grid[test_y, test_x] == 0:  
                return True
        return False

    
    def find_nearest(self, random_point):
        # Returns the node in self.nodes that is closest to random_point
        nearest = self.nodes[0]
        min_dist = self.distance(nearest, Node(random_point[0], random_point[1]))
        for node in self.nodes:
            dist = self.distance(node, Node(random_point[0], random_point[1]))
            if dist < min_dist:
                nearest = node
                min_dist = dist
        return nearest
    
    def find_nearby_nodes(self, new_node):
        # Return nodes within a neighborhood of new_node
        nearby = []
        for node in self.nodes:
            if self.distance(node, new_node) < self.neigh_radius:
                nearby.append(node)
        return nearby
    
    def rewire(self, new_node, nearby_nodes, writer = None):
        # Try to rewire each nearby node through new_node if it reduces the cost
        for near_node in nearby_nodes:
            if not self.is_in_obstacle(new_node, np.array([near_node.x, near_node.y])):
                potential_cost = new_node.cost + self.distance(new_node, near_node)
                if potential_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = potential_cost
                    # For visualization, draw the rewired edge in yellow
                    plt.plot([new_node.x, near_node.x], [new_node.y, near_node.y], 'y--')
                    plt.pause(0.05)
                    if writer is not None:
                        writer.grab_frame()
                    
    def plan(self, ax, writer=None):
        for i in range(self.num_iterations):
            random_point = self.sample_point()
            nearest_node = self.find_nearest(random_point)
            new_point = self.steer_to_point(nearest_node, random_point)
            if not self.is_in_obstacle(nearest_node, new_point):
                new_node = Node(new_point[0], new_point[1])
                # Set cost if reached from nearest_node
                new_node.cost = nearest_node.cost + self.distance(nearest_node, new_node)
                new_node.parent = nearest_node
                
                # Look for a better parent among nearby nodes
                nearby_nodes = self.find_nearby_nodes(new_node)
                for near_node in nearby_nodes:
                    if not self.is_in_obstacle(near_node, new_point):
                        temp_cost = near_node.cost + self.distance(near_node, new_node)
                        if temp_cost < new_node.cost:
                            new_node.parent = near_node
                            new_node.cost = temp_cost
                
                self.nodes.append(new_node)
                # Draw the new edge in green
                plt.plot([new_node.parent.x, new_node.x], [new_node.parent.y, new_node.y], 'g--')
                plt.pause(0.05)
                if writer is not None:
                    writer.grab_frame()
                
                # Rewire the nearby nodes with new_node as a potential parent
                self.rewire(new_node, nearby_nodes)
                
                # Check if we can connect to the goal
                if self.distance(new_node, self.goal_node) <= self.rho and not self.is_in_obstacle(new_node, [self.goal_node.x, self.goal_node.y]):
                    self.goal_node.parent = new_node
                    self.goal_node.cost = new_node.cost + self.distance(new_node, self.goal_node)
                    self.nodes.append(self.goal_node)
                    plt.plot([new_node.x, self.goal_node.x], [new_node.y, self.goal_node.y], 'b--')
                    plt.pause(0.05)
                    print("Goal reached at iteration:", i)
                    return self.retrace_path(self.goal_node)
        # If goal was not reached, choose the best node that can connect to the goal
        best_goal_node = None
        best_cost = float('inf')
        for node in self.nodes:
            if self.distance(node, self.goal_node) <= self.rho and not self.is_in_obstacle(node, [self.goal_node.x, self.goal_node.y]):
                cost = node.cost + self.distance(node, self.goal_node)
                if cost < best_cost:
                    best_cost = cost
                    best_goal_node = node
        if best_goal_node is not None:
            self.goal_node.parent = best_goal_node
            self.goal_node.cost = best_goal_node.cost + self.distance(best_goal_node, self.goal_node)
            self.nodes.append(self.goal_node)
            plt.plot([best_goal_node.x, self.goal_node.x], [best_goal_node.y, self.goal_node.y], 'b--')
            plt.pause(0.05)
            if writer is not None:
                writer.grab_frame()
            return self.retrace_path(self.goal_node)
        else:
            print("Goal not reached!")
            return None
    
    def retrace_path(self, goal_node):
        # Retrace from goal back to start
        path = []
        current = goal_node
        while current is not None:
            path.append(np.array([current.x, current.y]))
            current = current.parent
        path.reverse()
        return path

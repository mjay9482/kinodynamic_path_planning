import numpy as np
import cv2
from kinodynamic_path_planning.map_preprocessing import MapPreprocessor
from kinodynamic_path_planning.rrt import RRTAlgorithm
from kinodynamic_path_planning.rrt_star import RRTStarAlgorithm
from kinodynamic_path_planning.ILOS import ILOS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import os
from datetime import datetime

def main():
    
    algorithm_switch = "RRT"  # Switch between RRTStar and RRT  
    image = cv2.imread("../data/raw_images/archipelago.png")
    if image is None:
        raise IOError("Could not load image.")
    preprocessor = MapPreprocessor()
    preprocessor.process(image)
    grid = np.load("../data/processed_images/cspace.npy")

    start = np.array([77.0, 618.0])
    goal = np.array([1360.0, 221.0])
    num_iterations = 2000
    step_size = 50

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray', origin='upper')
    ax.plot(start[0], start[1], 'r*', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'b*', markersize=10, label='Goal')  
    start_circle = Circle((start[0], start[1]), step_size/2, color='r', fill=False)
    goal_circle = Circle((goal[0], goal[1]), step_size/2, color='b', fill=False)
    ax.add_patch(start_circle)
    ax.add_patch(goal_circle)
    plt.legend()

    # Initialize metrics
    path_length = 0
    success_rate = 0
    computation_time = 0
    map_coverage_area = 0
    
    # Record start time
    start_time = time.time()
    
    if algorithm_switch == "RRTStar":
        planner = RRTStarAlgorithm(start, goal, num_iterations, grid, step_size)
    else:
        planner = RRTAlgorithm(start, goal, num_iterations, grid, step_size)
    path = planner.plan(ax)
    plt.close(fig)  

    # Calculate metrics
    computation_time = time.time() - start_time
    
    if path:
        print("Path found. Launching ILOS simulation...")
        
        path_length = 0
        for i in range(1, len(path)):
            path_length += np.linalg.norm(path[i] - path[i-1])
        
        success_rate = 1.0
        
        map_coverage_area = len(path) * (step_size ** 2) / 4
        
        # Log metrics to file
        log_metrics(algorithm_switch, path_length, success_rate, computation_time, map_coverage_area)
        
        ilos = ILOS(path)
        ilos.simulate(start, goal, step_size, image, vessel_start=start)
    else:
        print("No path found.")
        
        # Log failed attempt
        log_metrics(algorithm_switch, 0, 0.0, computation_time, 0)

def log_metrics(algorithm, path_length, success_rate, computation_time, map_coverage_area):
    """Log performance metrics to a text file."""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Use a fixed filename in the logs directory
    filename = os.path.join(logs_dir, "performance_metrics.csv")
    
    # Check if file exists, if not create header
    file_exists = os.path.isfile(filename)
    
    # Open file in append mode
    with open(filename, 'a') as f:
        # Write header if file is new
        if not file_exists:
            f.write("Algorithm,Path Length,Success Rate,Computation Time (s),Map Coverage Area,Timestamp\n")
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write metrics
        f.write(f"{algorithm},{path_length:.2f},{success_rate:.2f},{computation_time:.4f},{map_coverage_area:.2f},{timestamp}\n")
    
    print(f"Metrics logged to {filename}")

if __name__ == "__main__":
    main()

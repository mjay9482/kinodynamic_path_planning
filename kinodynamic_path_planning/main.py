# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from matplotlib.animation import FFMpegWriter
# from map_preprocessing import MapPreprocessor
# from rrt import RRTAlgorithm
# from rrt_star import RRTStarAlgorithm  

# if __name__ == "__main__":
#     algorithm_switch = "RRTStar"  # Change to "RRTStar" as needed

#     image = cv2.imread("../data/raw_images/archipelago.png")
#     if image is None:
#         raise IOError("Could not load the image. Check the path.")
    
#     preprocessor = MapPreprocessor()
#     preprocessor.process(image)
    
#     grid = np.load("../data/processed_images/cspace.npy")
    
#     start = np.array([77.0, 618.0])
#     goal = np.array([1360.0, 221.0])
#     num_iterations = 2000
#     step_size = 30

#     # Visualization setup
#     fig, ax = plt.subplots()
#     ax.imshow(grid, cmap='gray', origin='upper')
#     ax.plot(start[0], start[1], 'ro', label='Start')
#     ax.plot(goal[0], goal[1], 'bo', label='Goal')
#     start_circle = Circle((start[0], start[1]), step_size, color='r', fill=False)
#     goal_circle = Circle((goal[0], goal[1]), step_size, color='b', fill=False)
#     ax.add_patch(start_circle)
#     ax.add_patch(goal_circle)
#     plt.legend()

#     writer = FFMpegWriter(fps=20)

#     # Instantiate selected planner
#     if algorithm_switch == "RRT":
#         planner = RRTAlgorithm(start, goal, num_iterations, grid, step_size)
#         title = "RRT Path Planning"
#         video_path = "../results/RRT.mp4"

#     elif algorithm_switch == "RRTStar":
#         planner = RRTStarAlgorithm(start, goal, num_iterations, grid, step_size)
#         title = "RRT* Path Planning"
#         video_path = "../results/RRT_star.mp4"

#     else:
#         raise ValueError("Invalid algorithm choice. Please select 'RRT' or 'RRTStar'.")
    
#     with writer.saving(fig, video_path, dpi=100):
#         path = planner.plan(ax, writer=writer)


#     path = planner.plan(ax)

#     if path:
#         for i in range(len(path) - 1):
#             ax.plot([path[i][0], path[i+1][0]],
#                     [path[i][1], path[i+1][1]],
#                     'r-', linewidth=2)
#         plt.title(title)
#         plt.show()
#     else:
#         plt.title(title + " failed to find a path")
#         plt.show()



import numpy as np
import cv2
from kinodynamic_path_planning.map_preprocessing import MapPreprocessor
from kinodynamic_path_planning.rrt import RRTAlgorithm
from kinodynamic_path_planning.rrt_star import RRTStarAlgorithm
from kinodynamic_path_planning.ILOS import ILOS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def main():
    
    algorithm_switch = "RRTStar"  # Switch between RRTStar and RRT  
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

    if algorithm_switch == "RRTStar":
        planner = RRTStarAlgorithm(start, goal, num_iterations, grid, step_size)
    else:
        planner = RRTAlgorithm(start, goal, num_iterations, grid, step_size)
    path = planner.plan(ax)
    plt.close(fig)  

    if path:
        print("Path found. Launching ILOS simulation...")
        ilos = ILOS(path)
        ilos.simulate(start, goal, step_size, image, vessel_start=start)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()

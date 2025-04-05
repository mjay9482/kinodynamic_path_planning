import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from kinodynamic_path_planning.map_preprocessing import MapPreprocessor
from kinodynamic_path_planning.rrt import RRTAlgorithm
from kinodynamic_path_planning.rrt_star import RRTStarAlgorithm

st.title("Path Planning with RRT / RRT* in Archipalego")

# Allow user to upload a map image
uploaded_file = st.file_uploader("Upload a map image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="Uploaded Image")
    
    # Preprocess the image using MapPreprocessor
    preprocessor = MapPreprocessor(K=5, attempts=10, max_iter=20, eps=0.9)
    binary_map = preprocessor.process(image)
    st.write("Image processing complete.")
    
    # Sidebar: collect planning parameters
    st.sidebar.header("Planning Parameters")
    start_x = st.sidebar.number_input("Start X", min_value=0, value=77)
    start_y = st.sidebar.number_input("Start Y", min_value=0, value=618)
    goal_x  = st.sidebar.number_input("Goal X", min_value=0, value=1360)
    goal_y  = st.sidebar.number_input("Goal Y", min_value=0, value=221)
    algorithm_choice = st.sidebar.selectbox("Algorithm", ["RRT", "RRTStar"])
    num_iterations = st.sidebar.number_input("Iterations", min_value=100, value=2000)
    step_size = st.sidebar.number_input("Step Size", min_value=1, value=30)
    
    if st.sidebar.button("Plan Path"):
        start = np.array([start_x, start_y])
        goal = np.array([goal_x, goal_y])
        grid = binary_map  
        
        # Setup Matplotlib figure for visualization
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray', origin='upper')
        ax.plot(start[0], start[1], 'ro', label='Start')
        ax.plot(goal[0], goal[1], 'bo', label='Goal')
        start_circle = Circle((start[0], start[1]), step_size, color='c', fill=False)
        goal_circle = Circle((goal[0], goal[1]), step_size, color='b', fill=False)
        ax.add_patch(start_circle)
        ax.add_patch(goal_circle)
        ax.legend()
        
        # Instantiate the selected planner
        if algorithm_choice == "RRT":
            planner = RRTAlgorithm(start, goal, num_iterations, grid, step_size)
            title = "RRT Path Planning"
        else:
            planner = RRTStarAlgorithm(start, goal, num_iterations, grid, step_size)
            title = "RRT* Path Planning"
        
        path = planner.plan(ax)
        
        # Plot the final path if found
        if path:
            for i in range(len(path) - 1):
                ax.plot([path[i][0], path[i+1][0]],
                        [path[i][1], path[i+1][1]],
                        'r-', linewidth=2)
            ax.set_title(title)
        else:
            ax.set_title(title + " failed to find a path")
        
        st.pyplot(fig)

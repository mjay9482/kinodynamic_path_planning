import numpy as np
import cv2
from scipy.spatial import distance
from scipy.optimize import minimize

class GraphSLAM:
    def __init__(self, map_size=(1000, 1000), feature_detector='orb', resolution=0.1):
        self.map_size = map_size
        self.landmarks = []  # List of landmark positions
        self.landmark_descriptors = []  # List of landmark descriptors
        self.robot_poses = []  # List of robot poses [x, y, theta]
        self.measurements = []  # List of measurements to landmarks
        self.odometry = []  # List of odometry measurements
        self.map_image = None  # Store the current map image
        self.last_frame = None  # Store the last processed frame
        self.last_keypoints = None  # Store the last keypoints
        self.last_descriptors = None  # Store the last descriptors
        self.optimization_counter = 0  # Counter for optimization frequency
        self.optimization_frequency = 10  # Optimize every N frames
        
        # Initialize feature detector with fewer features for speed
        if feature_detector == 'orb':
            self.detector = cv2.ORB_create(nfeatures=200, scaleFactor=1.2, nlevels=8, edgeThreshold=15, firstLevel=0, WTA_K=2, patchSize=31, fastThreshold=20)
        else:
            raise ValueError("Unsupported feature detector")
        
        # Add occupancy grid
        self.resolution = resolution
        self.occupancy_grid = np.zeros(map_size, dtype=np.float32)
        self.occupancy_grid.fill(0.5)  # Initialize with uncertainty
        
        # Create a new map from scratch
        self.generated_map = np.ones(map_size, dtype=np.uint8) * 255  # White background (free space)
        self.explored_cells = np.zeros(map_size, dtype=np.uint8)  # Track explored cells
        
        # Add map boundaries and scaling
        self.map_boundaries = {
            'x_min': 0,
            'x_max': map_size[0],
            'y_min': 0,
            'y_max': map_size[1]
        }
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def detect_features(self, frame):
        """Detect features in the current frame."""
        if frame is None:
            return [], None
            
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        scale = 0.5
        small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        # Scale keypoints back to original size
        for kp in keypoints:
            kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
        
        # Store for next frame
        self.last_frame = frame
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors
        
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio=0.75):
        """Match features between two frames using ratio test."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Use FLANN matcher for faster matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:  # FLANN matcher returned 2 matches
                    m, n = m_n
                    if m.distance < ratio * n.distance:
                        good_matches.append(m)
                elif len(m_n) == 1:  # FLANN matcher returned only 1 match
                    good_matches.append(m_n[0])
        except Exception as e:
            print(f"FLANN matcher failed: {e}")
            # Fall back to BF matcher if FLANN fails
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:  # BF matcher returned 2 matches
                    m, n = m_n
                    if m.distance < ratio * n.distance:
                        good_matches.append(m)
                elif len(m_n) == 1:  # BF matcher returned only 1 match
                    good_matches.append(m_n[0])
        
        return good_matches

    def update_map(self, frame, robot_pose):
        """Update the map with new observations."""
        # Always add the robot pose to track its trajectory
        # Convert robot_pose to a list if it's a numpy array
        if isinstance(robot_pose, np.ndarray):
            robot_pose = robot_pose.tolist()
        self.robot_poses.append(robot_pose)
        
        # Update the generated map with the robot's position
        self.update_generated_map(robot_pose)
        
        # Detect features in the current frame
        keypoints, descriptors = self.detect_features(frame)
        
        if len(keypoints) == 0:
            return
            
        if len(self.landmarks) == 0:
            # First frame - initialize landmarks
            self.landmarks = [kp.pt for kp in keypoints]
            self.landmark_descriptors = descriptors
            print(f"Initialized SLAM with {len(self.landmarks)} landmarks")
        else:
            # Match with existing landmarks
            matches = self.match_features(descriptors, self.landmark_descriptors)
            
            # Update existing landmarks and add new ones
            for match in matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                
                # Update measurement
                measurement = {
                    'landmark_idx': train_idx,
                    'position': keypoints[query_idx].pt,
                    'robot_pose': robot_pose
                }
                self.measurements.append(measurement)
            
            # Add new landmarks
            unmatched = set(range(len(keypoints))) - set(m.queryIdx for m in matches)
            for idx in unmatched:
                self.landmarks.append(keypoints[idx].pt)
                if self.landmark_descriptors is None:
                    self.landmark_descriptors = descriptors[idx].reshape(1, -1)
                else:
                    self.landmark_descriptors = np.vstack([self.landmark_descriptors, descriptors[idx]])
            
            print(f"Updated SLAM map: {len(self.landmarks)} landmarks, {len(self.measurements)} measurements")
            
            # Increment optimization counter
            self.optimization_counter += 1
            
            # Optimize map periodically
            if self.optimization_counter >= self.optimization_frequency:
                self.optimize_map()
                self.optimization_counter = 0

    def update_generated_map(self, robot_pose):
        """Update the generated map based on robot position."""
        # Convert robot pose to map coordinates
        x, y = int(robot_pose[0]), int(robot_pose[1])
        
        # Update map boundaries based on robot position
        self.map_boundaries['x_min'] = min(self.map_boundaries['x_min'], x - 100)
        self.map_boundaries['x_max'] = max(self.map_boundaries['x_max'], x + 100)
        self.map_boundaries['y_min'] = min(self.map_boundaries['y_min'], y - 100)
        self.map_boundaries['y_max'] = max(self.map_boundaries['y_max'], y + 100)
        
        # Calculate scale factor to fit the map in the visualization area
        map_width = self.map_boundaries['x_max'] - self.map_boundaries['x_min']
        map_height = self.map_boundaries['y_max'] - self.map_boundaries['y_min']
        
        if map_width > 0 and map_height > 0:
            # Add some padding
            self.scale_factor = min(
                self.map_size[0] / (map_width + 200),
                self.map_size[1] / (map_height + 200)
            )
            
            # Calculate offset to center the map
            self.offset_x = (self.map_size[0] - map_width * self.scale_factor) / 2
            self.offset_y = (self.map_size[1] - map_height * self.scale_factor) / 2
        
        # Mark the robot's position as explored
        if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
            self.explored_cells[y, x] = 255
            
            # Mark surrounding cells as explored (free space)
            radius = 20  # cells
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if 0 <= x+i < self.map_size[0] and 0 <= y+j < self.map_size[1]:
                        # Mark as explored
                        self.explored_cells[y+j, x+i] = 255
                        
                        # Mark as free space in the generated map
                        self.generated_map[y+j, x+i] = 255  # White (free space)
                        
                        # Add some noise to make it look more natural
                        if np.random.random() < 0.1:  # 10% chance
                            self.generated_map[y+j, x+i] = 240  # Slightly darker (still free space)
        
        # Add some "obstacles" based on landmarks
        for landmark in self.landmarks:
            lx, ly = int(landmark[0]), int(landmark[1])
            if 0 <= lx < self.map_size[0] and 0 <= ly < self.map_size[1]:
                # Mark landmark as explored
                self.explored_cells[ly, lx] = 255
                
                # Add a small obstacle around the landmark
                obstacle_radius = 5
                for i in range(-obstacle_radius, obstacle_radius+1):
                    for j in range(-obstacle_radius, obstacle_radius+1):
                        if 0 <= lx+i < self.map_size[0] and 0 <= ly+j < self.map_size[1]:
                            # Mark as obstacle in the generated map
                            self.generated_map[ly+j, lx+i] = 0  # Black (obstacle)

    def optimize_map(self):
        """Optimize the map using graph optimization."""
        if len(self.measurements) < 2:
            return
        
        # Limit the number of poses and measurements for optimization
        max_poses = 50
        max_measurements = 200
        
        # Use only the most recent poses and measurements
        recent_poses = self.robot_poses[-max_poses:] if len(self.robot_poses) > max_poses else self.robot_poses
        recent_measurements = self.measurements[-max_measurements:] if len(self.measurements) > max_measurements else self.measurements
        
        # Simple optimization using scipy
        def objective(params):
            error = 0
            num_landmarks = len(self.landmarks)
            landmark_params = params[:num_landmarks*2].reshape(-1, 2)
            pose_params = params[num_landmarks*2:].reshape(-1, 3)
            
            # Odometry constraints
            for i in range(len(self.odometry)-1):
                pred_pose = pose_params[i+1]
                meas_pose = self.odometry[i+1]
                error += np.sum((pred_pose - meas_pose)**2)
            
            # Measurement constraints
            for meas in recent_measurements:
                landmark_idx = meas['landmark_idx']
                robot_pose = meas['robot_pose']
                measured_pos = np.array(meas['position'])
                
                # Transform landmark to robot frame
                dx = landmark_params[landmark_idx][0] - robot_pose[0]
                dy = landmark_params[landmark_idx][1] - robot_pose[1]
                predicted_pos = np.array([
                    dx * np.cos(robot_pose[2]) + dy * np.sin(robot_pose[2]),
                    -dx * np.sin(robot_pose[2]) + dy * np.cos(robot_pose[2])
                ])
                
                error += np.sum((predicted_pos - measured_pos)**2)
            
            return error
        
        # Initial guess
        x0 = np.concatenate([
            np.array(self.landmarks).flatten(),
            np.array(recent_poses).flatten()
        ])
        
        # Optimize with fewer iterations
        result = minimize(objective, x0, method='BFGS', options={'maxiter': 50})
        
        # Update landmarks and poses
        num_landmarks = len(self.landmarks)
        optimized_landmarks = result.x[:num_landmarks*2].reshape(-1, 2)
        
        # Convert optimized landmarks back to a list of lists
        self.landmarks = optimized_landmarks.tolist()
        
        # Update only the recent poses
        optimized_poses = result.x[num_landmarks*2:].reshape(-1, 3).tolist()
        if len(self.robot_poses) > max_poses:
            self.robot_poses = self.robot_poses[:-max_poses] + optimized_poses
        else:
            self.robot_poses = optimized_poses

    def get_map(self):
        """Return the current map state."""
        return {
            'landmarks': self.landmarks,
            'robot_poses': self.robot_poses,
            'measurements': self.measurements
        }

    def visualize_map(self, frame):
        """Visualize the current map state."""
        # Create a visualization frame based on the generated map
        vis_frame = cv2.cvtColor(self.generated_map, cv2.COLOR_GRAY2BGR)
        
        # Limit the number of landmarks and poses to visualize for performance
        max_visualize = 100
        landmarks_to_show = self.landmarks[:max_visualize] if len(self.landmarks) > max_visualize else self.landmarks
        poses_to_show = self.robot_poses[-max_visualize:] if len(self.robot_poses) > max_visualize else self.robot_poses
        
        # Draw landmarks with scaling
        for i, landmark in enumerate(landmarks_to_show):
            # Apply scaling and offset
            x = int(landmark[0] * self.scale_factor + self.offset_x)
            y = int(landmark[1] * self.scale_factor + self.offset_y)
            
            if 0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]:
                cv2.circle(vis_frame, (x, y), 5, (0, 255, 0), -1)
                # Add landmark ID only for a subset
                if i % 5 == 0:  # Show ID for every 5th landmark
                    cv2.putText(vis_frame, str(i), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw robot poses with scaling
        for i, pose in enumerate(poses_to_show):
            x, y, theta = pose
            # Apply scaling and offset
            x = int(x * self.scale_factor + self.offset_x)
            y = int(y * self.scale_factor + self.offset_y)
            
            if 0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]:
                # Draw robot position
                cv2.circle(vis_frame, (x, y), 7, (0, 0, 255), -1)
                
                # Draw robot orientation
                end_x = int(x + 30 * np.cos(theta) * self.scale_factor)
                end_y = int(y + 30 * np.sin(theta) * self.scale_factor)
                if 0 <= end_x < vis_frame.shape[1] and 0 <= end_y < vis_frame.shape[0]:
                    cv2.line(vis_frame, (x, y), (end_x, end_y), (0, 0, 255), 2)
                
                # Draw pose ID only for a subset
                if i % 5 == 0:  # Show ID for every 5th pose
                    cv2.putText(vis_frame, str(i), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw connections between landmarks and robot poses (limited)
        max_connections = 50
        connections_to_show = self.measurements[-max_connections:] if len(self.measurements) > max_connections else self.measurements
        
        for meas in connections_to_show:
            landmark_idx = meas['landmark_idx']
            robot_pose = meas['robot_pose']
            if landmark_idx < len(self.landmarks):
                landmark = self.landmarks[landmark_idx]
                # Apply scaling and offset
                x1 = int(landmark[0] * self.scale_factor + self.offset_x)
                y1 = int(landmark[1] * self.scale_factor + self.offset_y)
                x2 = int(robot_pose[0] * self.scale_factor + self.offset_x)
                y2 = int(robot_pose[1] * self.scale_factor + self.offset_y)
                
                if (0 <= x1 < vis_frame.shape[1] and 0 <= y1 < vis_frame.shape[0] and
                    0 <= x2 < vis_frame.shape[1] and 0 <= y2 < vis_frame.shape[0]):
                    cv2.line(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        # Add text with statistics
        cv2.putText(vis_frame, f"Landmarks: {len(self.landmarks)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Poses: {len(self.robot_poses)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Measurements: {len(self.measurements)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add map boundaries information
        cv2.putText(vis_frame, f"Map: {self.map_boundaries['x_min']:.0f},{self.map_boundaries['y_min']:.0f} to {self.map_boundaries['x_max']:.0f},{self.map_boundaries['y_max']:.0f}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.map_image = vis_frame
        return vis_frame

    def get_map_image(self):
        """Return the current map image."""
        if self.map_image is None:
            # Create a default map if none exists
            self.map_image = cv2.cvtColor(self.generated_map, cv2.COLOR_GRAY2BGR)
        return self.map_image

    def update_occupancy_grid(self, robot_pose):
        # Convert robot pose to grid coordinates
        x, y = int(robot_pose[0] / self.resolution), int(robot_pose[1] / self.resolution)
        
        # Mark cells around robot as free space
        radius = 10  # cells
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if 0 <= x+i < self.map_size[0] and 0 <= y+j < self.map_size[1]:
                    # Update with log-odds
                    self.occupancy_grid[y+j, x+i] += 0.1  # Increase probability of free space 
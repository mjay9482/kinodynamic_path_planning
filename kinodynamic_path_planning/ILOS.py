import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
from .slam.graph_slam import GraphSLAM

class ILOS:
    def __init__(self, waypoints, speed=30, vessel_length=20, DELTA=20, kig=0.5, kappa=0.5, Kp=1.0, Kd=0.5):
        self.waypoints = waypoints
        self.speed = speed
        self.vessel_length = vessel_length
        self.DELTA = DELTA
        self.kig = kig
        self.kappa = kappa
        self.Kp = Kp
        self.Kd = Kd
        self.slam = GraphSLAM()  # Initialize SLAM system
        self.current_frame = None
        self.map_updated = False

    def ilos_compute_guidance_params(self, state, waypoints, latest_tracked):
        x_curr, y_curr, psi, yp_int = state
        if latest_tracked < 1:
            latest_tracked = 1
        wpi = waypoints[latest_tracked][0] - waypoints[latest_tracked - 1][0]
        wpf = waypoints[latest_tracked][1] - waypoints[latest_tracked - 1][1]
        theta = np.arctan2(wpf, wpi)
        R_np = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
        diff = np.array([[x_curr - waypoints[latest_tracked][0]],
                         [y_curr - waypoints[latest_tracked][1]]])
        xerr, yerr = (R_np.T @ diff).flatten()
        kpg = 1.0 / self.DELTA
        psi_des = theta - np.arctan(kpg * yerr + self.kig * yp_int)
        ypd_int = (self.DELTA * yerr) / (self.DELTA**2 + (yerr + self.kappa * yp_int)**2)
        return psi_des, ypd_int, xerr, yerr

    def kinematic_ode(self, t, state, rudder_command):
        x, y, psi, yp_int = state
        dxdt = self.speed * np.cos(psi)
        dydt = self.speed * np.sin(psi)
        dpsidt = rudder_command
        dyp_int_dt = 0
        return [dxdt, dydt, dpsidt, dyp_int_dt]

    def simulate(self, start, goal, step_size, map_array, vessel_start=None, dt=0.2, total_time=400):
        if vessel_start is None:
            vessel_start = self.waypoints[0]
        if len(self.waypoints) >= 2:
            dx = self.waypoints[1][0] - self.waypoints[0][0]
            dy = self.waypoints[1][1] - self.waypoints[0][1]
            psi0 = np.arctan2(dy, dx)
        else:
            psi0 = 0
        state = np.array([vessel_start[0], vessel_start[1], psi0, 0.0])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original map view
        image_rgb = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)
        ax1.imshow(image_rgb, origin='upper')
        waypoints_arr = np.array(self.waypoints)
        ax1.plot(start[0], start[1], 'y*', label='start', markersize=10)
        ax1.plot(goal[0], goal[1], 'y*', label='goal', markersize=10)
        ax1.add_patch(Circle((goal[0],goal[1]), step_size/2, color='y', fill=False))
        ax1.plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 'r--', linewidth=2, label='Planned Path')
        vessel_line, = ax1.plot([], [], 'k', linewidth=3, label='Vessel')
        ax1.legend()
        ax1.set_title("Vessel Trajectory")
        
        # SLAM map view
        ax2.set_title("SLAM Map")
        # Initialize with a copy of the original map
        slam_map_img = ax2.imshow(image_rgb.copy(), origin='upper')
        ax2.legend()
        
        dt_sim = dt
        num_steps = int(total_time / dt_sim)
        current_target_idx = 1
        prev_error = 0
        anim = None
        
        # Initialize SLAM with the first frame
        self.update_frame(map_array)
        print("Initialized SLAM with the map")
        
        # Counter for SLAM updates
        slam_update_counter = 0
        slam_update_frequency = 5  # Update SLAM every N frames

        def pid_control(psi_des, psi, dt):
            nonlocal prev_error
            error = psi_des - psi
            error = (error + np.pi) % (2 * np.pi) - np.pi
            derivative = (error - prev_error) / dt
            rudder_command = self.Kp * error + self.Kd * derivative
            prev_error = error
            return rudder_command

        def update(frame):
            nonlocal state, current_target_idx, anim, map_array, slam_update_counter
            
            # Update SLAM periodically
            slam_update_counter += 1
            if slam_update_counter >= slam_update_frequency:
                # Convert state to a list for SLAM
                robot_pose = [float(state[0]), float(state[1]), float(state[2])]
                
                # Create a synthetic frame for SLAM based on the current state
                # This is a simple visualization frame that shows the robot's view
                synthetic_frame = map_array.copy()
                
                # Draw a circle around the robot's position to simulate its view
                cv2.circle(synthetic_frame, (int(state[0]), int(state[1])), 50, (0, 255, 0), 2)
                
                # Draw a line in the direction the robot is facing
                end_x = int(state[0] + 50 * np.cos(state[2]))
                end_y = int(state[1] + 50 * np.sin(state[2]))
                cv2.line(synthetic_frame, (int(state[0]), int(state[1])), (end_x, end_y), (0, 0, 255), 2)
                
                # Update the SLAM system with the synthetic frame
                self.update_frame(synthetic_frame)
                self.slam.update_map(synthetic_frame, robot_pose)
                
                # Update SLAM visualization
                slam_vis = self.slam.visualize_map(map_array.copy())
                slam_map_img.set_array(cv2.cvtColor(slam_vis, cv2.COLOR_BGR2RGB))
                
                slam_update_counter = 0
            
            # Original ILOS update
            psi_des, ypd_int, xerr, yerr = self.ilos_compute_guidance_params(state, self.waypoints, current_target_idx)
            rudder_command = pid_control(psi_des, state[2], dt_sim)
            sol = solve_ivp(lambda t, s: self.kinematic_ode(t, s, rudder_command),
                           [0, dt_sim], state, method='RK23')
            state[:] = sol.y[:, -1]
            
            distance_to_target = np.linalg.norm(state[:2] - np.array(self.waypoints[current_target_idx]))
            if distance_to_target < self.DELTA * 0.5 and current_target_idx < len(self.waypoints) - 1:
                current_target_idx += 1
            if current_target_idx == len(self.waypoints) - 1:
                final_dist = np.linalg.norm(state[:2] - np.array(self.waypoints[-1]))
                if final_dist < self.DELTA * 0.5:
                    if anim is not None:
                        anim.event_source.stop()
            
            direction = np.array([np.cos(state[2]), np.sin(state[2])])
            vessel_end = state[:2] + self.vessel_length * direction
            vessel_line.set_data([state[0], vessel_end[0]], [state[1], vessel_end[1]])
            
            return vessel_line, slam_map_img
        
        anim = animation.FuncAnimation(fig, update, frames=num_steps,
                                     interval=dt_sim * 1000, blit=True, repeat=False)
        plt.show()
        return anim

    def update_frame(self, frame):
        """Update the current frame for SLAM processing."""
        self.current_frame = frame

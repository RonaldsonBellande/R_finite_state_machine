from grid import *
from particle import Particle
from utils import *
from setting import *
from itertools import product as prd
import numpy as np
import bisect
import math


# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- noisy odometry measurements, a pair of robot pose, i.e. last time
                step pose and current time step pose

        Returns: the list of particle representing belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    alpha1 = 0.001
    alpha2 = 0.001
    alpha3 = 0.005
    alpha4 = 0.005

    #get the pose of the robot
    previous_pose, current_pose = odom

    #The pose x, y, and angle of the previous and current pose
    previous_pose_x, previous_pose_y, previous_pose_angle = previous_pose
    current_pose_x, current_pose_y, current_pose_angle = current_pose

    #Calculate the two pose relative to each other in position and angle
    x_position, y_position, angle_position = current_pose_x - previous_pose_x, current_pose_y - previous_pose_y, proj_angle_deg(current_pose_angle - previous_pose_angle)

    rotation1 = proj_angle_deg(math.degrees(math.atan2(y_position, x_position)) - previous_pose_angle)
    trans = math.sqrt(x_position ** 2 + y_position ** 2)
    rotation2 = proj_angle_deg(angle_position - rotation1)

    #add gaussian noise 
    rotation1 = proj_angle_deg(add_gaussian_noise(rotation1, alpha1))
    trans = add_gaussian_noise(trans, ODOM_TRANS_SIGMA)
    rotation2 = proj_angle_deg(add_gaussian_noise(rotation2, alpha2))
    
    for particle in particles:
        #Get particles position
        particle.move(rotation1, trans, rotation2)
        
    return particles


class measurement_util(object):
    def __init__(self):
        self.constant_transformation = 2 * (MARKER_TRANS_SIGMA **2)
        self.constant_rotation = 2 * (MARKER_ROT_SIGMA **2)
        self.minimum_number_particle = int(PARTICLE_COUNT / 100)
        self.max_angle = 45
        self.probability = 1.0
        self.coefficient_constant_transformation = 0
        self.coefficient_rotation = (self.max_angle ** 2) / self.constant_rotation
    
    
    # If particle is in bound
    def particle_in_bound(self, particle, grid):
        return particle.x >= 0 and particle.x < grid.width and particle.y >= 0 and particle.y < grid.height

    # If Particle is being used
    def particle_in_used(self, particle, grid):
        return (particle.x, particle.y) in grid.occupied
    
    # Calculate probability
    def probability_calculator(self, particle_markers, robot_markers):
        diffrence = int(abs(len(robot_markers) - len(particle_markers)))
        pairs = min(len(robot_markers), len(particle_markers))
        
        for _ in range(0, pairs):
            self.best_particle_measurement, self.best_robot_measurement, self.best_robot_noise_measurement = None, None, None
            for robot_measurement in robot_markers:
                robot_noise_measurement = add_marker_measurement_noise(robot_measurement, MARKER_TRANS_SIGMA, MARKER_ROT_SIGMA)
                
                for particle_measurement in particle_markers:
                    if self.best_particle_measurement == None or grid_distance(particle_measurement[0], particle_measurement[1], robot_noise_measurement[0], robot_noise_measurement[1]) <  grid_distance(self.best_particle_measurement[0], self.best_particle_measurement[1], self.best_robot_noise_measurement[0], self.best_robot_noise_measurement[1]):
                        
                        self.best_particle_measurement, self.best_robot_measurement, self.best_robot_noise_measurement = particle_measurement, robot_measurement, robot_noise_measurement
                        
            particle_markers.remove(self.best_particle_measurement)
            robot_markers.remove(self.best_robot_measurement)
            
            # Calculate probability
            distance, angle = grid_distance(self.best_particle_measurement[0], self.best_particle_measurement[1], self.best_robot_noise_measurement[0], self.best_robot_noise_measurement[1]), diff_heading_deg(self.best_particle_measurement[2], self.best_robot_noise_measurement[2])
            
            coefficient_transformation, coefficient_rotation = (distance **2) / self.constant_transformation, (angle **2) / self.constant_rotation
            coefficient_sum = coefficient_transformation + coefficient_rotation
            self.coefficient_constant_transformation = max(self.coefficient_constant_transformation, coefficient_transformation)
            self.probability *= math.exp(-coefficient_sum)
        
        total_coefficient = self.coefficient_constant_transformation + self.coefficient_rotation
        for _ in range(diffrence):
            self.probability *= math.exp(-total_coefficient)
            
        return self.probability
    
    
    # Resampling
    def resampling(self, particles, grid, particle_weights, number_particle_replacement):
        particle_weights.sort(key=lambda x: x[1])
        for particle, weight in particle_weights:
            if weight == 0:
                number_particle_replacement += 1
                
        number_particle_replacement = max(number_particle_replacement, self.minimum_number_particle)
        particle_weights = particle_weights[number_particle_replacement:]
        total_probability = sum(weight for particle, weight in particle_weights)
        
        particles = [particle for particle, weight in particle_weights]
        weights = [weight / total_probability for particle, weight in particle_weights]
        measured_particles = Particle.create_random(number_particle_replacement, grid)
        
        resampled = np.random.choice(particles, size=len(particles), replace = True, p=weights)
        
        for par in resampled:
            measured_particles.append(Particle(add_gaussian_noise(par.x, ODOM_TRANS_SIGMA), add_gaussian_noise(par.y, ODOM_TRANS_SIGMA), add_gaussian_noise(par.h, ODOM_HEAD_SIGMA)))
            
        return measured_particles
            
        
        
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- a list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
        grid -- grid world map containing the marker information. 
                see grid.py and CozGrid for definition

        Returns: the list of particle representing belief p(x_{t} | u_{t})
                after measurement update
    """
    particle_weights = []
    number_particle_replacement = 0
    
    if len(measured_marker_list) == 0: 
        return particles
    #Getting the weights
    for particle in particles:
        utilities = measurement_util()
        if not utilities.particle_in_bound(particle, grid) or utilities.particle_in_used(particle, grid):
            particle_weights.append((particle, 0))
            continue

        probability = utilities.probability_calculator(particle.read_markers(grid), measured_marker_list[:])
        particle_weights.append((particle, probability))
        
    # Resampling
    measured_particles = utilities.resampling(particle, grid, particle_weights, number_particle_replacement)
    return measured_particles
    
    

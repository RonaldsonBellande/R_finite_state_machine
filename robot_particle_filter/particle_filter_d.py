from numpy.lib.function_base import angle, average
from scipy.stats.stats import _shape_with_dropped_axis
from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np
from math import *
import scipy.stats
import sys

# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """Particle filter motion update

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

    (xbar, ybar, thbar), (xbar2, ybar2, thbar2) = odom
    

    def sample(bsquared):
        return np.random.normal(0, abs(bsquared))
        pass
    
    def limit_angle(thnew):
        while thnew > pi:
            thnew -= pi * 2
        while thnew < -pi:
            thnew += pi * 2
        return thnew
    
    thbar = radians(thbar)
    thbar2 = radians(thbar2)

    def update_particle(xt):

        # insight! xt is the hypothesized robot position
        (x,y,th) = xt.xyh
        
        # convert degrees to radians
        th = radians(th)

        # recover relative motion parameters from the odometry readings
        delta_rot1 = atan2(ybar2 - ybar, xbar2 - xbar) - thbar
        delta_trans = sqrt((xbar2 - xbar) ** 2 + (ybar2 - ybar) ** 2)
        delta_rot2 = thbar2 - thbar - delta_rot1
        
        delta_hat_rot1 = delta_rot1 - sample(alpha1 * delta_rot1 + alpha2 * delta_trans)
        delta_hat_trans = delta_trans - sample(alpha3 * delta_trans + alpha4 * (delta_rot1 + delta_rot2))
        delta_hat_rot2 = delta_rot2 - sample(alpha1 * delta_rot2 + alpha2 * delta_trans)

        thnew = limit_angle(th + delta_hat_rot1 + delta_hat_rot2)

        xp,yp,thp = (
            x + delta_hat_trans * cos(th + delta_hat_rot1),
            y + delta_hat_trans * sin(th + delta_hat_rot1),
            
            # convert radians to degrees I guess
            degrees(thnew)
        )
    
        return xp,yp,thp

    new_particles = list(map(update_particle, iter(particles)))

    return new_particles

def npdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid: CozGrid):
    """Particle filter measurement update

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
    
    q = []
    
    if not any(measured_marker_list):
        return [Particle(p[0],p[1],p[2]) for p in particles]
    
    def add_noise(p):
        x,y,h = p
        return (x+np.random.normal(0,0.1),y+np.random.normal(0,0.1),h+np.random.normal(0,.1))
        # return x,y,h
    
    def constrain(degrees):
        if degrees > 180:
            return degrees - 360
        if degrees < -180:
            return degrees + 360
        return degrees
    
    rnorm = scipy.stats.norm(0,.5)
    hnorm = scipy.stats.norm(0,7)
    
    def get_range_error(sim,msr):
        simx, simy, _ = sim
        msrx, msry, _ = msr
        predicted_range = sqrt(simx**2+simy**2)
        actual_range = sqrt(msrx**2+msry**2)
        diff_range = predicted_range - actual_range
        return diff_range
    
    def get_angle_error(sim,msr):
        _, __, simh = sim
        _, __, msrh = msr
        predicted_angle = simh
        actual_angle = msrh
        diff_angle = constrain(actual_angle - predicted_angle)
        return diff_angle
        
    range_errors_main = []
    angle_errors_main = []
        
    for p in particles:
        
        x,y,h = p
        particle = Particle(x,y,h)
        
        sim_markers = particle.read_markers(grid)
        
        if not any(sim_markers):
            q.append(0)
            continue
        
        range_errors = [get_range_error(sm,mm) for sm, mm in zip(sim_markers, measured_marker_list)]
        angle_errors = [get_angle_error(sm,mm) for sm, mm in zip(sim_markers, measured_marker_list)]
        
        # probs = [rnorm.pdf(rerr)*hnorm.pdf(aerr) for rerr,aerr in zip(range_errors,angle_errors)]
        
        nre = np.linalg.norm(range_errors)
        nae = np.linalg.norm(angle_errors)
        
        q.append(1/(nre*nae**1.1))
        
        # for sm,mm in zip(sim_markers, measured_marker_list):
                        
        #     simx, simy, simh = sm
        #     msrx, msry, msrh = mm
                    
        #     # msrd = sqrt((msrx)**2+(msry)**2) # measured distance to landmark, from robot
        #     # simd = sqrt((simx)**2+(simy)**2) # simulated distance to landmark, from particle
            
        #     predicted_range = sqrt(simx**2+simy**2)
        #     actual_range = sqrt(msrx**2+msry**2)
        #     diff_range = predicted_range - actual_range
            
        #     dist = sqrt((msrx-simx)**2+(msry-simy)**2)
            
        #     dprob = rnorm.pdf(dist)
        #     hprob = hnorm.pdf(constrain(simh-msrh))
        #     p = dprob * hprob
            
        #     q.append(p)
    
    # qh = hnorm.pdf(angle_errors_main)
    # qr = rnorm.pdf(range_errors_main)
    # q = qh * qr
    
    if sum(q) > 0:
        weights = np.array(q)/sum(q)
        indexes = np.arange(0, len(particles))
        resampled = np.random.choice(indexes, size=(len(particles)),p=weights)
        new_particles = [particles[i] for i in resampled]
    else:
        new_particles = particles
    
    new_particles = map(add_noise, new_particles)
    
    # x=1
    
    # def prob(a,b):
    #     return scipy.stats.norm(0,b).pdf(a)
    
    # def get_likelihood(p):
        
    #     x,y,h = p
    #     particle = Particle(x,y,heading=h)
        
    #     # get the landmarks the particle should see, given a robot's FOV
    #     simulated_markers_list = particle.read_markers(grid)
        
        
        
    #     q = 1/len(particles)
        
    #     for (sm, mm) in zip(simulated_markers_list, measured_marker_list):
            
    #         simx, simy, simh = sm
    #         msrx, msry, msrh = mm
            
            
            
    #         # r_it = sqrt((simx-x)**2+(simy-y)**2)
            
    #         # distance between what I would see and what I do see
    #         r_hat = sqrt((simx-msrx)**2+(simy-msry)**2)
    #         # phi_it = atan2(simy - y, simx - x) 
    #         phi_hat = atan2(msry - simy, msrx - simx) 
            
    #         # if phi_it > pi:
    #         #     phi_it -= pi
    #         # if phi_it < -pi:
    #         #     phi_it += pi
    #         if phi_hat > pi:
    #             phi_hat -= pi
    #         if phi_hat < -pi:
    #             phi_hat += pis
            
    #         sigr = 2
            
    #         q2 = scipy.stats.norm(0,sigr).pdf(r_hat) * scipy.stats.norm(sigr).pdf(phi_hat)
    #         if (q == 0):
    #             q = q2
    #         else:
    #             q *= q2
                
    #     return q
    
    # weights = np.array(list(map(get_likelihood, particles)))
    # weights /= sum(weights)
    # resample = np.random.choice(np.arange(0,len(particles)), p=weights, size=(len(particles)))
    return list(map(lambda p : Particle(p[0],p[1],p[2]), new_particles))


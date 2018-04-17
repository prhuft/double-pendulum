""""
 Double Compound Pendulum Simulation. 
	
Preston Huft, Spring 2018. 

Numerical simulation of compound double pendulum, solved iteratively with
the leap-frog method. For now, both pendula have the same mass and arm 
length. Note that there aren't any mass terms in the equations, as all 
mass terms cancel out under this condition. 

To-Do List: 
- Replace leap-frog method with Runge-Kutta (4th order).
- Generalize acceleration equations to not assume the same mass and arm 
length for both pendula. 
- Become bored enough to turn this into a double "springdulum" simulation. 
- Add subplots of angular position, velocity, and acceleration
- figure out how to force the plot to be square (i.e. not distorted)
- use a state variable to keep track of theta, omega, alpha?
"""

## Libraries 

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math as m
import time as t
from math import cos
from math import sin

## Global Constants

g = 9.8 # [m/s]
DEBUG = 0 # set to 1 if we're debugging

## Methods

def alpha1_update(theta1,theta2,omega1,omega2,alpha2,m1,m2,l1,l2):
	""" Returns alpha1, the angular acceleration of the first arm. """
	
	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a2 = alpha2
	
	I1 = (1./3)*m1*l1**2
	
	a1 = (1./8)*m2*l1*l2*(o2*(o1*(sin(o1)*cos(o2)+cos(o2)*sin(o2))+o2*(cos(o1)*sin(o2)+sin(o1)*cos(o2)))
	+a2*(sin(o2)*sin(o1)-cos(o1)*cos(o2)))/(m1*((l1/2.)**2)+I1)
	
	# a1 = (15/2)*((1/8)*(cos(t1)*sin(t2)*o1*o2 - sin(o1)*sin(o2)*o1*o2 
	# - cos(o1)*cos(o2)*a2 - sin(o2)*sin(o1)*a2) - (g/l)*sin(o2))
	
	# if DEBUG:
		# print('o1*o2, a2 = ', o1*o2, a2)
		# print('alpha1 update: ', a1)
		# print()
		
	return a1
	
def alpha2_update(theta1,theta2,omega1,omega2,alpha1,m1,m2,l1,l2):
	""" Returns alpha2, the angular acceleration of the first arm. """
	
	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a1 = alpha1
	
	I2 = (1./3)*m2*l2**2
	
	a2 = (1./8)*m1*l2*l1*(o1*(o2*(sin(o2)*cos(o1)+cos(o1)*sin(o1))+o1*(cos(o2)*sin(o1)+sin(o2)*cos(o1)))
	+a1*(sin(o1)*sin(o2)-cos(o2)*cos(o1)))/(m2*((l2/2.)**2)+I2)
	
	# a2 = (15/2)*((1/8)*(cos(t2)*sin(t1)*o1*o2 - sin(o1)*sin(o2)*o1*o2
	# - cos(o1)*cos(o2)*a1 - sin(o2)*sin(o1)*a1) - (g/l)*sin(o1))
	
	# if DEBUG:
		# print('alpha2 update: ', a2)
		# print()
	return a2
	
# #begin region: Methods for leap-frog method of ODE solving 
# def LF_theta_update(theta,omega,tau):
	# """ Returns theta_(n+2), given theta_n and omega_(n+1) and tau."""
	# # if DEBUG:
		# # print('theta update: ',theta + 2*tau*omega)
		# # print()
	# return theta + 2*tau*omega
	
# def LF_omega_update(omega,alpha,tau):
	# """ Returns omega_(n+1), the angular speed of an arm, given omega_(n-1) and
	# alpha_n. and tau."""
	# # if DEBUG:
		# # print('omega update: ',omega + 2*tau*alpha)
		# # print()
	# return omega + 2*tau*alpha
	
# def LF_data(theta1,theta2,omega1,omega2,alpha1,alpha2,dt,steps):
	# """" Computes the positions of the double pendulum system at each
	# timestep, for steps number of iterations, dt [s] apart."""
	
	# # Angular position
	# t1 = theta1 
	# t2 = theta2
	# t1_old = t1
	# t2_old = t2 
	
	# # Angular velocity 
	# o1 = omega1 
	# o2 = omega2
	
	# # Angular acceleration 
	# a1 = alpha1
	# a2 = alpha2
	
	# # Timestep
	# tau = dt
	
	# xData1, yData1 = toXY(t1)[0], toXY(t1)[1]
	# xData2, yData2 = toXY(t2)[0]+xData1, toXY(t2)[1]+yData1
	
	# # Leap-frog method gets odd positions and even velocities, so we iterate 
	# # twice as many steps to get steps number of positions.
	# for i in range(1,2*steps+1):
		# try:
			# if i % 2 != 0:
				# o1 = LF_omega_update(o1,a1,tau)
				# o2 = LF_omega_update(o2,a2,tau)
			# else: 
				# temp_t1 = LF_theta_update(t1,o1,tau)
				# t2 = LF_theta_update(t2,o2,tau)
				
				# temp_a1 = LF_alpha1_update(t1,t2,o1,o2,a2)
				# temp_a2 = LF_alpha2_update(t1,t2,o1,o2,a1)
				
				# a1 = temp_a1
				# a2 = temp_a2
				# t1 = temp_t1
				
				# xData1, yData1 = toXY(t1)[0],toXY(t1)[1]
				# xData2, yData2 = toXY(t2)[0]+xData1,toXY(t2)[1]+yData1
				
				# data1 = xData1,yData1 # endpoint of arm 1
				# data2 = xData2,yData2 # endpoint of arm 2
				
				# # return the data
				# yield data1,data2 
				
			# if DEBUG:
				# print('Iter ',i,': t1,t2= ',t1,t2)
				# print('o1,o2= ',o1,o2,' a1,a2= ',a1,a2)
				# print()
			# # if i % 10 == 0:
				# # yield data1,data2
		# except ValueError:		
			# print('value error at iteration ',i)
			# break
# #end region: Methods for leap-frog method of ODE solving 

#begin region: Methods for Runge-Kutta method of ODE solving
def RK_omega1_update(theta1,theta2,omega1,omega2,alpha2,tau,m1,m2,l1,l2):

	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a2 = alpha2
	
	h = tau

	def k(do):
		return h*alpha1_update(t1,t2,o1+do,o2,a2,m1,m2,l1,l2)
	k1 = k(0)
	k2 = k(k1/2)
	k3 = k(k2/2)
	k4 = k(k3)
		
	return o1+(k1+2*k2+2*k3+k4)/6
	
def RK_omega2_update(theta1,theta2,omega1,omega2,alpha1,tau,m1,m2,l1,l2):

	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a1 = alpha1
	
	dt = tau

	def k(do):
		return dt*alpha2_update(t1,t2,o1+do,o2,a1,m1,m2,l1,l2)
	k1 = k(0)
	k2 = k(k1/2)
	k3 = k(k2/2)
	k4 = k(k3)
		
	return o2+(k1+2*k2+2*k3+k4)/6
	
def RK_theta1_update(theta1,theta2,omega1,omega2,alpha2,tau,m1,m2,l1,l2):

	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a2 = alpha2
	
	dt = tau

	def k(dt):
		return dt*RK_omega1_update(t1+dt,t2,o1,o2,a2,dt,m1,m2,l1,l2)
	k1 = k(0)
	k2 = k(k1/2)
	k3 = k(k2/2)
	k4 = k(k3)
		
	return o1+(k1+2*k2+2*k3+k4)/6
	
def RK_theta2_update(theta1,theta2,omega1,omega2,alpha1,tau,m1,m2,l1,l2):

	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a1 = alpha1
	
	dt = tau

	def k(dt):
		return dt*RK_omega2_update(t1,t2+dt,o1,o2,a1,dt,m1,m2,l1,l2)
	k1 = k(0)
	k2 = k(k1/2)
	k3 = k(k2/2)
	k4 = k(k3)
		
	return o1+(k1+2*k2+2*k3+k4)/6

def RK_data(m1,m2,l1,l2,theta1,theta2,omega1,omega2,tau,steps):
	"""" Computes the positions of the double pendulum system at each
	timestep, for steps number of iterations, dt [s] apart."""
	
	# Angular position, n = 0
	t1 = theta1 
	t2 = theta2
	
	# Angular velocity, n = 0 
	o1 = omega1 
	o2 = omega2
	
	# Angular acceleration, n = 0
	a1 = alpha_init(t1,l1)#-3*g*sin(t1)/(2*l)
	a2 = alpha_init(t2,l2)#-3*g*sin(t2)/(2*l) 
	
	if DEBUG:
				print('Iter 0',': t1,t2= ',t1,t2)
				print('o1,o2= ',o1,o2,' a1,a2= ',a1,a2)
				print()
	
	# Timestep
	dt = tau
	
	# The initial endpoints of each arm, in Cartesian coordinates
	xData1, yData1 = toXY(t1,l1)[0], toXY(t1,l1)[1]
	xData2, yData2 = toXY(t2,l2)[0]+xData1, toXY(t2,l2)[1]+yData1
	
	# Angular velocity, n = 1 (Taylor expansion "classical kinematics" approx)
	# o1 = o1 + .5*a1*dt**2 
	# o2 = o2 + .5*a2*dt**2
	
	# Forward feed the RK4 method for m = 0 to m = steps
	for i in range(0,steps): 
		try:
			
			# Angular positions, n = m + 1 
			t1 = RK_theta1_update(t1,t2,o1,o2,a2,dt,m1,m2,l1,l2)
			t2 = RK_theta1_update(t1,t2,o1,o2,a1,dt,m1,m2,l1,l2)
			
			# Update our position state
			xData1, yData1 = toXY(t1,l1)[0],toXY(t1,l1)[1]
			xData2, yData2 = toXY(t2,l2)[0]+xData1,toXY(t2,l2)[1]+yData1
			
			data1 = xData1,yData1 # endpoint of arm 1
			data2 = xData2,yData2 # endpoint of arm 2
			
			if DEBUG:
				print('Iter ',i,': t1,t2= ',t1,t2)
				print('o1,o2= ',o1,o2,' a1,a2= ',a1,a2)
				print()
			
			# Angular acceleration, n = m + 2 
			a1 = alpha1_update(t1,t2,o1,o2,a2,m1,m2,l1,l2)
			a2 = alpha2_update(t1,t2,o1,o2,a1,m1,m2,l1,l2)
			
			# Angular velocity, n = m + 2
			o1 = RK_omega1_update(t1,t2,o1,o2,a2,dt,m1,m2,l1,l2)
			o2 = RK_omega1_update(t1,t2,o1,o2,a1,dt,m1,m2,l1,l2)
			
			# Return the data
			yield data1,data2 
				
		except ValueError:		
			print('value error at iteration ',i)
			break
	
#end region: Methods for Runge-Kutta method of ODE solving

def alpha_init(theta,length):
	""" Returns the initial angular acceleration due only to gravitational 
	torque exerted on the center of mass of a uniform arm of length length
	about one end, held at angle theta from the vertical."""
	return -6*g*sin(theta)/length
	
def toXY(theta,length):
	""" Returns the (x,y) coordinate of the end of a single pendulum arm."""
	return length*sin(theta), -length*cos(theta)
	
def init():
	""" Initialize the plot. """
	l = 0
	if (len1 > len2):
		l = len1
	else:
		l = len2
	
	ax.set_ylim(-2*l*1.1,2*l*1.1)
	ax.set_xlim(-2*l*1.1,2*l*1.1)

	# line.set_data([],[])
	pen_line.set_data([],[])
	trail1_line.set_data([],[])
	trail2_line.set_data([],[])
	# return line
	return pen_line,trail1_line,trail2_line

# This seems like a really bad place to initialize variables

# Create the figure and axes objects
fig, ax = plt.subplots()
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Initialize the 2DLines, which are lists of tuples (hence the comma)
pen_line, = ax.plot([],[],color='white',lw=2)
trail1_line, = ax.plot([],[],color='orange',lw=1)
trail2_line, = ax.plot([],[],color='purple',lw=1)

# Initialize the data which will fill the lines
pen_xdata, pen_ydata = [],[]
trail1_xdata, trail1_ydata = [],[]
trail2_xdata, trail2_ydata = [],[]

def run(data):
	# update the data
	x1, y1 = data[0]
	x2, y2 = data[1]

	pen_xdata = [[0,x1,x2]]
	pen_ydata = [[0,y1,y2]]
	pen_line.set_data(pen_xdata,pen_ydata)
	
	trail1_xdata.append(x1)
	trail1_ydata.append(y1)
	trail1_line.set_data(trail1_xdata,trail1_ydata)
	
	trail2_xdata.append(x2)
	trail2_ydata.append(y2)
	trail2_line.set_data(trail2_xdata,trail2_ydata)
	
	return pen_line,trail1_line,trail2_line

## The main code

# Simulation parameters
mass1 = 1 # [kg]
mass2 = 2 # [kg]
len1 = 1 # [m]
len2 = 1 # [m]
theta1_0 = m.pi/4 # [rad] from vertical
theta2_0 = m.pi/6 # ditto
omega1_0 = 0 # [rad/s] 
omega2_0 = 0 # ditto
alpha1_0 = alpha_init(theta1_0,len1) # [rad/s^2]
alpha2_0 = alpha_init(theta2_0,len2) # ditto
dt = 0.5 # [s]
iters = 1000 # times to update the systems

# data_gen = LF_data(theta1_0,theta2_0,omega1_0,omega2_0,alpha1_0,alpha2_0,dt,iters)
data_gen = RK_data(mass1,mass2,len1,len2,theta1_0,theta2_0,omega1_0,omega2_0,
dt,iters)

# This is what iterates through run(data) and plots the result.
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=dt*1000,
                              repeat=False, init_func=init)
plt.show()
""""
Double Compound Pendulum Simulation. 
	
Preston Huft, Spring 2018. 

Numerical simulation of compound double pendulum, solved iteratively with
the leap-frog method. For now, both pendula have the same mass and arm 
length. Note that there aren't any mass terms in the equations, as all 
mass terms cancel out under this condition. 

To-Do List: 
- Become bored enough to turn this into a double "springdulum" simulation. 
- Add subplots of angular position, velocity, and acceleration
- figure out how to force the plot to be square (i.e. not distorted)
- calculate frames prior to animation
"""

## Libraries 

from rk4_two_bodies import rk4_update as rk4
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from random import random as rnd
import math as m
import time as t
from math import cos
from math import sin

## Global Constants

g = 9.8 # [m/s]
TRIPPY = 0
DEBUG = 0 # set to 1 if we're debugging
TIMING = 0 # " " " " " trying to measure update time

## Methods

#begin region: Methods for Runge-Kutta method of ODE solving
def derivs(state,tau,params):

	# The state of the pendulum
	t1,t2,o1,o2,a1,a2 = state
	
	# The masses and arm lengths
	m1,m2,l1,l2 = params
	
	dt = tau
	
	# Moments of inertia for each arm
	I1 = (1./3)*m1*l1**2
	I2 = (1./3)*m2*l2**2
	
	a1 = (((1./8)*m2*l1*l2*(o2*(o1*(sin(o1)*cos(o2)+cos(o2)*sin(o2))+
	o2*(cos(o1)*sin(o2)+sin(o1)*cos(o2)))+a2*(sin(o2)*sin(o1)-
	cos(o1)*cos(o2)))-m1*g*l1*sin(t1)/2.-(m2/4.)*a2*l2**2)/(m1*((l1/2.)**2)+
	I1+(1./4)*m2*l1**2))
		
	a2 = (((1./8)*m1*l2*l1*(o1*(o2*(sin(o2)*cos(o1)+cos(o1)*sin(o1))+
	o1*(cos(o2)*sin(o1)+sin(o2)*cos(o1)))+a1*(sin(o1)*sin(o2)-
	cos(o2)*cos(o1)))-m2*g*l2*sin(t2)/2.-(m1/4.)*a1*l1**2)/(m2*((l2/2.)**2)+
	I2+(1./4)*m2*l1**2))
	
	return o1,o2,a1,a2
		
# make mass, etc. into a params array
def get_data(params,state,tau,steps,num_update):
	"""" Computes the positions of the double pendulum system at each
	timestep, for steps number of iterations, dt [s] apart. num_update
	is the numerical method function to be used."""
	
	# Get pendulum parameters
	m1,m2,l1,l2 = params
	
	# The initial state
	t1,t2,o1,o2,a1,a2 = state
	
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
			t1,t2,o1,o2,a1,a2 = num_update([t1,t2,o1,o2,a1,a2],dt,params,derivs)
			
			# Update our position state data
			xData1, yData1 = toXY(t1,l1)[0],toXY(t1,l1)[1]
			xData2, yData2 = toXY(t2,l2)[0]+xData1,toXY(t2,l2)[1]+yData1
			
			data1 = xData1,yData1 # endpoint of arm 1
			data2 = xData2,yData2 # endpoint of arm 2
			
			if DEBUG:
				print('Iter ',i,': t1,t2= ',t1,t2)
				print('o1,o2= ',o1,o2,' a1,a2= ',a1,a2)
				print()
			
			# Return the data
			yield data1,data2 
				
		except ValueError:		
			print('value error at iteration ',i)
			break
	
#end region: Methods for Runge-Kutta method of ODE solving

def alpha_init(theta,length):
	""" Returns the initial angular acceleration due only to gravitational 
	torque exerted on the center of mass of a uniform arm of length 'length'
	about one end, held at angle theta from the vertical."""
	return -6*g*sin(theta)/length
	
def toXY(theta,length):
	""" Returns the (x,y) coordinate of the end of a single pendulum arm."""
	return length*sin(theta), -length*cos(theta)
	
def init():
	""" Initialize the plot. """
	l = 0
	if (l1 > l2):
		l = l1
	else:
		l = l2
	
	ax.set_ylim(-2*l*1.1,2*l*1.1)
	ax.set_xlim(-2*l*1.1,2*l*1.1)

	pen_line.set_data([],[])
	trail1_line.set_data([],[])
	trail2_line.set_data([],[])
	# return line
	return pen_line,trail1_line,trail2_line

def run(data):
	# update the data
	x1, y1 = data[0]
	x2, y2 = data[1]

	# Random colored lines, updated each iteration
	# if (TRIPPY):
		# # pen_line, = ax.plot([],[],color=(rnd(),rnd(),rnd()),lw=1)
		# trail1_line, = ax.plot([],[],color=(rnd(),rnd(),rnd()),lw=1)
		# trail2_line, = ax.plot([],[],color=(rnd(),rnd(),rnd()),lw=1)
	
	pen_xdata = [[0,x1,x2]]
	pen_ydata = [[0,y1,y2]]
	pen_line.set_data(pen_xdata,pen_ydata)
	
	trail1_xdata.append(x1)
	trail1_ydata.append(y1)
	trail1_line.set_data(trail1_xdata,trail1_ydata)
	
	trail2_xdata.append(x2)
	trail2_ydata.append(y2)
	trail2_line.set_data(trail2_xdata,trail2_ydata)
	
	if TIMING:
		t_arr.append(t.clock())
		print(t.clock())
	# t.sleep(.5)
	
	return pen_line,trail1_line,trail2_line

## The main code

# Create the figure and axes objects
fig, ax = plt.subplots()
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Initialize the 2DLines, which are lists of tuples (hence the comma)
pen_line, = ax.plot([],[],color='white',lw=2)
trail1_line, = ax.plot([],[],color='yellow',lw=1)
trail2_line, = ax.plot([],[],color='magenta',lw=1)

# Initialize the data which will fill the lines
pen_xdata, pen_ydata = [],[]
trail1_xdata, trail1_ydata = [],[]
trail2_xdata, trail2_ydata = [],[]

run_dt = 0.045 # [s] ~ time each update takes, for which we can correct

# Simulation parameters
m1 = 1 #1 # [kg]
m2 = 1 # [kg]
l1 = 1 # [m]
l2 = 1# [m]
t1_0 = m.pi/2 # [rad] from vertical
t2_0 = m.pi/2 # ditto
o1_0 = 0 # [rad/s] 
o2_0 = 0 # ditto
a1_0 = alpha_init(t1_0,l1) # [rad/s^2]
a2_0 = alpha_init(t2_0,l2) # ditto

# Pendulum attributes
params = [m1,m2,l1,l2]

# Initial state
state_0 = [t1_0,t2_0,o1_0,o2_0,a1_0,a2_0]

dt = 0.01 # [s]
iters = 10000 # times to update the systems

data_gen = get_data(params,state_0,dt,iters,rk4)

# timekeeping
t_arr = [] 

# This is what iterates through run(data) and plots the result.
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=dt*1000,
							  repeat=False, init_func=init)
							  
plt.show()

if TIMING:
	dt_arr = []
	last_t = 0
	for t in t_arr:
		dt_arr.append(t-last_t)
		last_t = t
		
# ani.save('dbl_pendul.mp4', writer=writer)
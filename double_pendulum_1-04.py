""""
Double Compound Pendulum Simulation, v1.04
	
Preston Huft, Spring 2018. 

Numerical simulation of compound double pendulum, solved iteratively with
the Runge-Kutta (4th order) method. 

Version notes: First version to utilize rk4.py (rather than rk4_two_bodies.py)
which is a generalized RK4 update method. Format of state variable changed to 
accomadate this update. 

To-Do List: 
- Become bored enough to turn this into a double "springdulum" simulation. 
- Add subplots of angular position, velocity, and acceleration
"""

## Libraries 

from rk4 import rk4_update as rk4
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

def derivs(state,tau,params):
	"""Returns a list of the first and second derivatives for each pendulum,
	given the current state,timestep tau, and system params."""
	# The state of the pendulum
	t1,t2 = state[0]
	o1,o2 = state[1]
	a1,a2 = state[2]
	
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
	
	return [o1,o2],[a1,a2]
		
def get_data(state,tau,steps,params,num_update):
	""" Returns a list of the states of the double pendulum system at each
	timestep, for steps number of iterations, dt [s] apart. num_update is the
	numerical method function to be used. """
	
	# Get pendulum parameters
	m1,m2,l1,l2 = params
	
	# The initial state
	t1,t2 = state[0]
	o1,o2 = state[1]
	a1,a2 = state[2]
	
	if DEBUG:
		print('Iter 0',': t1,t2= ',t1,t2)
		print('o1,o2= ',o1,o2,' a1,a2= ',a1,a2)
		print()
	
	# Timestep
	dt = tau
	
	# Initialize the endpoints of each arm, in Cartesian coordinates
	xData1,yData1 = [toXY(t1,l1)[0]],[toXY(t1,l1)[1]]
	xData2,yData2 = [toXY(t2,l2)[0]+xData1[0]],[toXY(t2,l2)[1]+yData1[0]]
	
	# Forward feed the solver method for i = 0 to i = steps
	for i in range(0,steps): 
		try:
			# Update each variable
			new_state = num_update([[t1,t2],[o1,o2],[a1,a2]],dt,params,derivs)
			t1,t2 = new_state[0]
			o1,o2 = new_state[1]
			a1,a2 = new_state[2]
			
			xData1 += [toXY(t1,l1)[0]]
			yData1 += [toXY(t1,l1)[1]]
			xData2 += [toXY(t2,l2)[0]+xData1[i+1]]
			yData2 += [toXY(t2,l2)[1]+yData1[i+1]]
			
			if DEBUG:
				print('Iter ',i,': t1,t2= ',t1,t2)
				print('o1,o2= ',o1,o2,' a1,a2= ',a1,a2)
				print()
			
		except ValueError:		
			print('value error at iteration ',i)
			break
			
		data1 = xData1,yData1 # endpoints of arm 1
		data2 = xData2,yData2 # endpoints of arm 2
			
	return data1,data2
	
def alpha_init(theta,length):
	""" Returns the initial angular acceleration due only to gravitational 
	torque exerted on the center of mass of a uniform arm of length 'length'
	about one end, held at angle theta from the vertical."""
	return -6*g*sin(theta)/length
	
def toXY(theta,length):
	""" Returns the (x,y) coordinate of the end of a single pendulum arm."""
	return length*sin(theta), -length*cos(theta)

## SYSTEM INITIALIZATION
# run_dt = 0.045 # [s] ~ time each update takes, for which we can correct

# Simulation parameters
m1 = 1 #1 # [kg]
m2 = 1 # [kg]
l1 = 1 # [m]
l2 = 1 # [m]
t1_0 = m.pi/2 # [rad] from vertical
t2_0 = m.pi/2 # ditto
o1_0 = 0 # [rad/s] 
o2_0 = 0 # ditto
a1_0 = alpha_init(t1_0,l1) # [rad/s^2]
a2_0 = alpha_init(t2_0,l2) # ditto

# Pendulum attributes
params = [m1,m2,l1,l2]

# Initial state, with pendula state variables grouped by derivative order
state_0 = [[t1_0,t2_0],[o1_0,o2_0],[a1_0,a2_0]]

dt = 0.01 # [s]
iters = 1#10000 # times to update the systems

data = get_data(state_0,dt,iters,params,rk4)

## SIMULATION SETUP

# Initialize the figure
data1,data2 = data
x1pts,y1pts = data1
x2pts,y2pts = data2

iters = len(x1pts) # this could have used any of the point arrays.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('black')
ax.set_aspect(aspect='equal')
fig.patch.set_facecolor('black')

# Initialize the lines to be plotted
pen_line, = ax.plot([],[],color='white',lw=3)
trail1_line, = ax.plot([],[],color='yellow',lw=1)
trail2_line, = ax.plot([],[],color='magenta',lw=1)

def init():
	""" Set the axes limits. """
	l = 0
	if (params[2] > params[3]):
		l = params[2]
	else:
		l = params[3]

	ax.set_ylim(-2*l*1.1,2*l*1.1)
	ax.set_xlim(-2*l*1.1,2*l*1.1)
	return (pen_line,
			trail1_line,
			trail2_line,)
	
def update(i):
	""" Uses values established previously as globals."""
	# Set the line describing the pendula arms
	pen_line.set_data([0,x1pts[i],x2pts[i]],[0,y1pts[i],y2pts[i]])

	# Set the trail lines from the arm endpoints
	j = i + 1;
	trail1_line.set_data(x1pts[:j],y1pts[:j])
	trail2_line.set_data(x2pts[:j],y2pts[:j])
	
	return (pen_line,
			trail1_line,
			trail2_line,)

anim = animation.FuncAnimation(fig, update, frames=range(0,iters+1), 
	init_func=init, blit=True, interval=1000*dt, repeat=True)
plt.show()


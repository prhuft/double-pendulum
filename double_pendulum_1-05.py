""""
Double Compound Pendulum Simulation, v1.05
	
Preston Huft, Spring 2018. 

Numerical simulation of compound double pendulum, solved iteratively with
the Runge-Kutta (4th order) method. 

Version notes: First version to attempt multiple- double pendula plotting, i.e.
a simulation showing the time evolution of several overlayed double pendula with
differing initial conditions.

To-Do List: 
- Become bored enough to turn this into a double "springdulum" simulation. 
- Add subplots of angular position, velocity, and acceleration
- Generalize get_initial_states for any type of system; i.e. pass in a function
	to handle system-specific things, such as getting the initial a1,a2. 
"""

## LIBRARIES

from rk4 import rk4_update as rk4
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from random import random as rnd
import math as m
import time as t
from math import cos
from math import sin

## GLOBAL CONSTANTS

g = 9.8 # [m/s]
TRIPPY = 0
DEBUG = 0 # set to 1 if we're debugging
TIMING = 0 # " " " " " trying to measure update time

## METHODS

def derivs(state,tau,params):
	"""Returns a list of the first and second derivatives for both arms in the
	double pendulum system, given the current state,timestep tau, and system 
	params."""
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
		
# def get_data(state,tau,steps,params,num_update):
def get_data(states,tau,steps,params,num_update):
	""" Returns a list of the states of the double pendulum systems at each
	timestep, for steps number of iterations, dt [s] apart. num_update is the
	numerical method function to be used. """
	
	# Get pendulum parameters; assume same params for each system
	m1,m2,l1,l2 = params 
	
	# The arrays of endpoints of arm1 and arm2 for each system. I.e., 
	# arm1_data = [[[x1_1,y1_1],[x1_2,y1_2],...,[x1_n,y1_n]],
	#			   [[x2_1,y2_1],[x2_2,y2_2],...,[x2_n,y2_n]],...,
	#			   [[xm_1,y2_1],[xm_2,y2_2],...,[xm_n,ym_n]]]
	# where [xi_j,yi_j] is the endpoint coordinate of arm1 in the ith 
	# double pendulum system at step j. Likewise for arm2.
	arm1_data,arm2_data = [],[] 
	
	# Generate the states at each timestep over the specified number of
	# steps for each double pendulum system
	for state in states:
	
		# The initial state of the nth double pendulum
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
				
			arm1_data += xData1,yData1 # endpoints of arm 1
			arm2_data += xData2,yData2 # endpoints of arm 2
			
	return arm1_data,arm2_data
	
def get_initial_states(params,state_template,dstate,sys_size):
	""" Returns a list of length 'sys_size', the elements of which are the 
		initial states for each double pendulum system. That is, 
		states = [[state1_0],[state2_0],...,[statem_0
		where statei_0 is the initial state of the ith double pendulum, and 
		m = sys_size. 
		dstate = the initial difference between adjacent systems
		state_template = the state of one system; each other state is built
		by adding adding dstate scaled by an integer in (1,sys_size).
		sys_size = the number of systems to generate."""
	l1,l2 = params[2:]
	
	state = state_template
	states_0 = [state_template]
	for i in range(0,sys_size):
		state += dstate # add dstate each iteration
		# overwrite [a1,a2] which depend on the initial angle
		state[2] = [alpha_init(state[0][0],l1),alpha_init(state[0][1],l2)]
		states_0.append(state)
	return states_0
	
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
m2 = .5 # [kg]
l1 = 1 # [m]
l2 = 1 # [m]

# Pendulum attributes -- assume each double pendulum system has the same params
params = [m1,m2,l1,l2]

# The state variables for one double pendulum
t1_0 = m.pi/2 # [rad] from vertical
t2_0 = m.pi/2 # ditto
o1_0 = 0 # [rad/s] 
o2_0 = 0 # ditto
a1_0 = alpha_init(t1_0,l1) # [rad/s^2]
a2_0 = alpha_init(t2_0,l2) # ditto

# The difference in initial variables between "adjacent" systems
dt1 = m.pi/360 # half a degree to radians
dt2 = m.pi/360 # same
do1 = 0
do2 = 0
da1 = 0 # a1_0 has to be calculated for each system in get_initial_states()
da2 = 0 # same. just leave as 0 for now

# Initial variables of one double_pendulum, grouped by derivative order. 
state1 = [[t1_0,t2_0],[o1_0,o2_0],[a1_0,a2_0]]

# The initial state difference between "adjacent" systems
delta_state = [[dt1,dt2],[do1,do2],[da1,da2]]

# The number of different systems we want to generate
total = 2

# Generate the initial state for each double pendulum system
states_0 = get_initial_states(params,state1,delta_state,total)

dt = 0.01 # [s]
iters = 10000 # times to update the systems

# Generate the data
data = get_data(states_0,dt,iters,params,rk4)

## SIMULATION SETUP

# Initialize the figure
x1pts,y1pts = [],[]
x2pts,y2pts = [],[]
arm1data,arm2data = data # positions for all pendula for all timesteps
xpts,ypts = [],[]
for armdata in data: # iterates twice
	for data_n in armdata: # iterate through the data for all n systems
		xpts.append(data_n[0])
		ypts.append(data_n[1])
x1pts,x2pts = xpts[:int(len(xpts)/2)],xpts[int(len(xpts)/2+1):]
y1pts,y2pts = ypts[:int(len(xpts)/2)],xpts[int(len(xpts)/2+1):]

iters = len(x1pts) # this could have used any of the point arrays.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('black')
ax.set_aspect(aspect='equal')
fig.patch.set_facecolor('black')

# Initialize the lines to be plotted 
pen_line = []
trail1_line = []
trail2_line = []
for i in range(0,total):
	pen_line.append(ax.plot([],[],color='white',lw=3))
	trail1_line.append(ax.plot([],[]))#,color='yellow',lw=1)
	trail2_line.append(ax.plot([],[]))#,color='magenta',lw=1)

# pen_line, = ax.plot([],[],color='white',lw=3)
# trail1_line, = ax.plot([],[],color='yellow',lw=1)
# trail2_line, = ax.plot([],[],color='magenta',lw=1)

def init():
	""" Set the axes limits. """
	l = 0
	if (params[2] > params[3]):
		l = params[2]
	else:
		l = params[3]

	ax.set_ylim(-2*l*1.1,2*l*1.1)
	ax.set_xlim(-2*l*1.1,2*l*1.1)
	# return (pen_line,
			# trail1_line,
			# trail2_line,)
	return pen_line + trail1_line + trail2_line

	
def update(i):
	""" Uses values established previously as globals."""
	j = i + 1;
	# Set the lines to plot
	for k in range(0,total):
		# The line describing the kth double pendulum
		pen_line[k].set_data([0,x1pts[k][i],x2pts[k][i]],
							 [0,y1pts[k][i],y2pts[k][i]])
		# The lines from the arm endpoints
		trail1_line[k].set_data(x1pts[k][:j],y1pts[k][:j])
		trail2_line[k].set_data(x2pts[k][:j],y2pts[k][:j])
	# return (pen_line,
			# trail1_line,
			# trail2_line,)
	return pen_line + trail1_line + trail2_line

anim = animation.FuncAnimation(fig, update, frames=range(0,iters+1), 
	init_func=init, blit=True, interval=1000*dt, repeat=True)
plt.show()


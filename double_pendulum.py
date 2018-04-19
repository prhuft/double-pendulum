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
from random import random as rnd
import math as m
import time as t
from math import cos
from math import sin

## Global Constants

g = 9.8 # [m/s]
TRIPPY = 1
DEBUG = 0 # set to 1 if we're debugging

## Methods

#begin region: Methods for Runge-Kutta method of ODE solving
def derivs(theta1,theta2,omega1,omega2,alpha1,alpha2,tau,params):
	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a1 = alpha1
	a2 = alpha2
	
	m1,m2,l1,l2 = params
	
	dt = tau
	
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
	
def RK4_update(r1,r2,v1,v2,a1,a2,t,p,derivatives):
	"""Return the next r1,r2,v1,v2,a1,a2 given a function that returns values
	of v1,v2,a1,a2. t is the step size, and p is an array of additional 
	parameters."""
		
	def k(dh): 
		h = t
		# return kr1(h),kr2(h),kv1(h),kv2(h)
		return (h*derivatives(r1+dh[0],r2,v1,v2,a1,a2,t,p)[0],
		h*derivatives(r1,r2+dh[1],v1,v2,a1,a2,t,p)[1],
		h*derivatives(r1,r2,v1+dh[2],v2,a1,a2,t,p)[2],
		h*derivatives(r1,r2,v1,v2+dh[3],a1,a2,t,p)[3])
		
	# each k_i here is a list of k(h_i) evaluated for each state variable
	k1 = k([0,0,0,0]) 
	k2 = k([x/2. for x in k1])
	k3 = k([x/2. for x in k2])
	k4 = k([x for x in k3])

	# current state
	s = [r1,r2,v1,v2]
	
	# new state
	snew = []
		
	for i in range(0,len(s)):
		snew.append(s[i]+(k1[i]+2*(k2[i]+k3[i])+k4[i])/6.)
		
	r1,r2,v1,v2 = snew
		
	d = derivatives(r1,r2,v1,v2,a1,a2,t,p)
	
	snew.append(d[-2]) # get new a1
	snew.append(d[-1]) # get new a2

	return snew
		
# make mass, etc. into a params array
def RK_data(m1,m2,l1,l2,theta1,theta2,omega1,omega2,tau,steps):
	"""" Computes the positions of the double pendulum system at each
	timestep, for steps number of iterations, dt [s] apart."""
	
	# Params; eventually just pass this in
	params = [m1,m2,l1,l2]
	
	# Angular position, n = 0
	t1 = theta1 
	t2 = theta2
	
	# Angular velocity, n = 0 
	o1 = omega1 
	o2 = omega2
	
	# Angular acceleration, n = 0
	a1 = alpha_init(t1,l1)
	a2 = alpha_init(t2,l2)
	
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
			
			# Update each variable
			t1,t2,o1,o2,a1,a2 = RK4_update(t1,t2,o1,o2,a1,a2,dt,params,derivs)
			
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
	torque exerted on the center of mass of a uniform arm of length length
	about one end, held at angle theta from the vertical."""
	return 6*g*sin(theta)/length
	
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

def run(data):
	# update the data
	x1, y1 = data[0]
	x2, y2 = data[1]

	# Random colored lines, updated each iteration\
	if (TRIPPY):
		# pen_line, = ax.plot([],[],color=(rnd(),rnd(),rnd()),lw=1)
		trail1_line, = ax.plot([],[],color=(rnd(),rnd(),rnd()),lw=1)
		trail2_line, = ax.plot([],[],color=(rnd(),rnd(),rnd()),lw=1)
	
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

# Create the figure and axes objects
fig, ax = plt.subplots()
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Create the figure and its subplots
# fig = plt.figure()
# ax_p = fig.add_subplot(1,)
# ax_t = fig.add_subplot()
# ax_o = fig.add_subplot()
# ax_a = fig.add_subplot()


# Initialize the 2DLines, which are lists of tuples (hence the comma)
pen_line, = ax.plot([],[],color='white',lw=2)
trail1_line, = ax.plot([],[],color='orange',lw=1)
trail2_line, = ax.plot([],[],color='purple',lw=1)

# Initialize the data which will fill the lines
pen_xdata, pen_ydata = [],[]
trail1_xdata, trail1_ydata = [],[]
trail2_xdata, trail2_ydata = [],[]

# Simulation parameters
mass1 = 1 # [kg]
mass2 = 1 # [kg]
len1 = 1 # [m]
len2 = 1 # [m]
theta1_0 = m.pi/4 # [rad] from vertical
theta2_0 = m.pi/3 # ditto
omega1_0 = 0 # [rad/s] 
omega2_0 = 0 # ditto
alpha1_0 = alpha_init(theta1_0,len1) # [rad/s^2]
alpha2_0 = alpha_init(theta2_0,len2) # ditto
dt = 0.05 # [s]
iters = 10000 # times to update the systems

# data_gen = LF_data(theta1_0,theta2_0,omega1_0,omega2_0,alpha1_0,alpha2_0,dt,iters)
data_gen = RK_data(mass1,mass2,len1,len2,theta1_0,theta2_0,omega1_0,omega2_0,
dt,iters)

# This is what iterates through run(data) and plots the result.
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=dt*1000,
                              repeat=False, init_func=init)
plt.show()

# ani.save('dbl_pendul.mp4', writer=writer)
""" Double Compound Pendulum Simulation. 
	
	Preston Huft, Spring 2018. 
	
	Numerical simulation of compound double pendulum, solved iteratively with
	the leap-frog method. For now, both pendula have the same mass and arm 
	length. Note that there aren't any mass terms in the equations, as all 
	mass terms cancel out under this condition. 
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
l = 0.1 # [m]
DEBUG = 0 # set to 1 if we're debugging

## Methods

#begin region: Methods for leap-frog method of ODE solving 
def LF_theta_update(theta,omega,tau):
	""" Returns theta_(n+2), given theta_n and omega_(n+1) and tau."""
	# if DEBUG:
		# print('theta update: ',theta + 2*tau*omega)
		# print()
	return theta + 2*tau*omega
	
def LF_omega_update(omega,alpha,tau):
	""" Returns omega_(n+1), the angular speed of an arm, given omega_(n-1) and
	alpha_n. and tau."""
	# if DEBUG:
		# print('omega update: ',omega + 2*tau*alpha)
		# print()
	return omega + 2*tau*alpha
	
def LF_alpha1_update(theta1,theta2,omega1,omega2,alpha2):
	""" Returns alpha1, the angular acceleration of the first arm. """
	
	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a2 = alpha2
	
	a1 = (15/2)*((1/8)*(cos(t1)*sin(t2)*o1*o2 - sin(o1)*sin(o2)*o1*o2 
	- cos(o1)*cos(o2)*a2 - sin(o2)*sin(o1)*a2) - (g/l)*sin(o2))
	
	# if DEBUG:
		# print('o1*o2, a2 = ', o1*o2, a2)
		# print('alpha1 update: ', a1)
		# print()
		
	return a1
	
def LF_alpha2_update(theta1,theta2,omega1,omega2,alpha1):
	""" Returns alpha2, the angular acceleration of the first arm. """
	
	t1 = theta1
	t2 = theta2
	o1 = omega1
	o2 = omega2
	a1 = alpha1
	
	a2 = (15/2)*((1/8)*(cos(t2)*sin(t1)*o1*o2 - sin(o1)*sin(o2)*o1*o2
	- cos(o1)*cos(o2)*a1 - sin(o2)*sin(o1)*a1) - (g/l)*sin(o1))
	# if DEBUG:
		# print('alpha2 update: ', a2)
		# print()
	return a2
#end region: Methods for leap-frog method of ODE solving 
	
def toXY(theta):
	""" Returns the (x,y) coordinate of the end of a single pendulum arm."""
	return l*sin(theta), -l*cos(theta)

def LF_data(theta1,theta2,omega1,omega2,alpha1,alpha2,dt,steps):
	"""" Computes the positions of the double pendulum system at each
	timestep, for steps number of iterations, dt [s] apart."""
	
	# Angular position
	t1 = theta1 
	t2 = theta2
	t1_old = t1
	t2_old = t2 
	
	# Angular velocity 
	o1 = omega1 
	o2 = omega2
	
	# Angular acceleration 
	a1 = alpha1
	a2 = alpha2
	
	# Timestep
	tau = dt
	
	xData1, yData1 = toXY(t1)[0], toXY(t1)[1]
	xData2, yData2 = toXY(t2)[0]+xData1, toXY(t2)[1]+yData1
	
	# Leap-frog method gets odd positions and even velocities, so we iterate 
	# twice as many steps to get steps number of positions.
	for i in range(1,2*steps+1):
		try:
			if i % 2 != 0:
				o1 = LF_omega_update(o1,a1,tau)
				o2 = LF_omega_update(o2,a2,tau)
			else: 
				temp_t1 = LF_theta_update(t1,o1,tau)
				t2 = LF_theta_update(t2,o2,tau)
				
				temp_a1 = LF_alpha1_update(t1,t2,o1,o2,a2)
				temp_a2 = LF_alpha2_update(t1,t2,o1,o2,a1)
				
				a1 = temp_a1
				a2 = temp_a2
				t1 = temp_t1
				
				xData1, yData1 = toXY(t1)[0],toXY(t1)[1]
				xData2, yData2 = toXY(t2)[0]+xData1,toXY(t2)[1]+yData1
				
				data1 = xData1,yData1 # endpoint of arm 1
				data2 = xData2,yData2 # endpoint of arm 2
				
				# return the data
				yield data1,data2 
				
			if DEBUG:
				print('Iter ',i,': t1,t2= ',t1,t2)
				print('o1,o2= ',o1,o2,' a1,a2= ',a1,a2)
				print()
			# if i % 10 == 0:
				# yield data1,data2
		except ValueError:		
			print('value error at iteration ',i)
			break
	
def init():
	""" Initialize the plot. """
	ax.set_ylim(-2*l*1.1,2*l*1.1)
	ax.set_xlim(-2*l*1.1, 2*l*1.1)

	# line.set_data([],[])
	pen_line.set_data([],[])
	trail1_line.set_data([],[])
	trail2_line.set_data([],[])
	# return line
	return pen_line,trail1_line,trail2_line

fig, ax = plt.subplots()
pen_line, = ax.plot([],[],color='black',lw=2)
trail1_line, = ax.plot([],[],color='orange',lw=1)
trail2_line, = ax.plot([],[],color='purple',lw=1)
#ax.grid()
#xdata, ydata = [],[]
pen_xdata, pen_ydata = [],[]
trail1_xdata, trail1_ydata = [],[]
trail2_xdata, trail2_ydata = [],[]

def run(data):
	# update the data
	x1, y1 = data[0]
	x2, y2 = data[1]
	
	# xdata,ydata are the three points defining our pendula's position. 
	# xdata = [[0,x1,x2]]
	# ydata = [[0,y1,y2]]
	
	pen_xdata = [[0,x1,x2]]
	pen_ydata = [[0,y1,y2]]
	pen_line.set_data(pen_xdata,pen_ydata)
	
	trail1_xdata.append(x1)
	trail1_ydata.append(y1)
	trail1_line.set_data(trail1_xdata,trail1_ydata)
	
	trail2_xdata.append(x2)
	trail2_ydata.append(y2)
	trail2_line.set_data(trail2_xdata,trail2_ydata)
	
	# set the line to the pendula arms position
	# line.set_data(xdata,ydata)
	# return line
	return pen_line,trail1_line,trail2_line

## The main code

# Simulation parameters
theta1_0 = m.pi/3 # [rad] from vertical
theta2_0 = m.pi/6 # ditto
omega1_0 = 0 # [rad/s] 
omega2_0 = 0 # ditto
alpha1_0 = -3*g*sin(theta1_0)/l # [rad/s^2]
alpha2_0 = -3*g*sin(theta2_0)/l # ditto
tau = 0.0005 # [s]
iters = 1000 # times to update the systems

data_gen = LF_data(theta1_0,theta2_0,omega1_0,omega2_0,alpha1_0,alpha2_0,tau,iters)

# This is what iterates through run(data) and plots the result.
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=1,
                              repeat=False, init_func=init)
plt.show()
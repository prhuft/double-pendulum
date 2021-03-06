""" 
Runge-Kutta Numerical ODE Solving Method, tweaked for a two-body system.

Preston Huft, Spring 2018.

The method below can be called from another python file to generate the next 
iteration of parameters after a given step 'h', for a system with state variables
contained in 'state' and other system parameters contained in 'p', where 
'derivatives' is a method which returns an array of derivatives as described.

To-do: generalize for single or multiple-body (>2) systems. This could be achieved
with overload prototypes. 
"""
DEBUG = 0

def rk4_update(state,h,p,derivatives):
	"""Return the next r1,r2,v1,v2,a1,a2 given a function that returns values
	of v1,v2,a1,a2. t is the step size, and p is an array of additional 
	parameters which are specific to the system and may be needed by the
	derivatives function."""
	
	r1,r2,v1,v2,a1,a2 = state
		
	def k(dh): 
		# print the modified states:
		if DEBUG:
			# print('temp,0,0',[r1+dh[0],r2,v1,v2,a1,a2])
			# print('temp,0,1',[r1,r2+dh[1],v1,v2,a1,a2])
			# print('temp,1,0',[r1,r2,v1+dh[2],v2,a1,a2])
			# print('temp,1,1',[r1,r2,v1,v2+dh[3],a1,a2])
		
		return (h*derivatives([r1+dh[0],r2,v1,v2,a1,a2],h,p)[0],
		h*derivatives([r1,r2+dh[1],v1,v2,a1,a2],h,p)[1],
		h*derivatives([r1,r2,v1+dh[2],v2,a1,a2],h,p)[2],
		h*derivatives([r1,r2,v1,v2+dh[3],a1,a2],h,p)[3])
		
	# each k_i here is a list of k(h_i) evaluated for each state variable
	k1 = k([0,0,0,0]) 
	k2 = k([x/2. for x in k1])
	k3 = k([x/2. for x in k2])
	k4 = k([x for x in k3])
	
	if (DEBUG):
		print('dh1: ',[0,0,0,0])
		print('k1: ',k1)
		print('dh2: ',[(1/2.)*x for x in k1])
		print('k2: ',k2)
		print('dh3: ',[(1/2.)*x for x in k2])
		print('k3: ',k3)
		print('dh4: ',[x for x in k3])
		print('k4: ',k4)
	
	# new state
	snew = []
	
	# Update t1,t2,o1,o2
	for i in range(0,4):
		snew.append(state[i]+(k1[i]+2*(k2[i]+k3[i])+k4[i])/6.)
		
	d = derivatives(state,h,p)
	
	snew.append(d[-2]) # get new a1
	snew.append(d[-1]) # get new a2

	return snew
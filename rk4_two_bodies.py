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


def rk4_update(state,h,p,derivatives):
	"""Return the next r1,r2,v1,v2,a1,a2 given a function that returns values
	of v1,v2,a1,a2. t is the step size, and p is an array of additional 
	parameters which are specific to the system and may be needed by the
	derivatives function."""
	
	r1,r2,v1,v2,a1,a2 = state
		
	def k(dh): 
		# return kr1(h),kr2(h),kv1(h),kv2(h)
		return (h*derivatives([r1+dh[0],r2,v1,v2,a1,a2],h,p)[0],
		h*derivatives([r1,r2+dh[1],v1,v2,a1,a2],h,p)[1],
		h*derivatives([r1,r2,v1+dh[2],v2,a1,a2],h,p)[2],
		h*derivatives([r1,r2,v1,v2+dh[3],a1,a2],h,p)[3])
		
	# each k_i here is a list of k(h_i) evaluated for each state variable
	k1 = k([0,0,0,0]) 
	k2 = k([x/2. for x in k1])
	k3 = k([x/2. for x in k2])
	k4 = k([x for x in k3])
	
	# new state
	snew = []
	
	# Update t1,t2,o1,o2
	for i in range(0,4):
		snew.append(state[i]+(k1[i]+2*(k2[i]+k3[i])+k4[i])/6.)
	
	# not sure if this is needed
	# r1,r2,v1,v2 = snew
		
	d = derivatives(state,h,p)
	
	snew.append(d[-2]) # get new a1
	snew.append(d[-1]) # get new a2

	return snew
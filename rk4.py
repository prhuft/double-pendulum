""" 
Runge-Kutta Numerical ODE Solving Method

Preston Huft, Spring 2018.

The method below can be called from another python file to generate the next 
iteration of parameters after a given step 'h', for a system with state variables
contained in 'state' and other system parameters contained in 'p', where 
'derivatives' is a method which returns an array of derivatives as described.
"""


def rk4_update(state,h,params,derivatives):
	""" state = [[f1^(0),f2^(0),...,fn^(0)],[f1^(1),f2^(1),...fn^(1)],
				...,[f1^(m),f2^(m),...fn^(m)]]
				
				where fi^(j) denotes the jth derivative of the generalized
				position of the ith object in the system.
		h = step size to be used in derivative calculations. 
		params = a list of system parameters which required by the 
			derivatives method; see the specific derivatives function for more
			information.
		derivatives = derivatives(state,h,params)
			a method which returns a list of the m derivatives of 
			the generalized position of the n bodies in the system, i.e.:
			[[f1^(1),f2^(1),...fn^(1)], ...,[f1^(m),f2^(m),...fn^(m)]]
			
		This method returns the updated state of the system, in the exact form 
		of the 'state' list which is passed in. 
	"""
	
	def k_mat(dh_mat):
		""" The output is of the form: 
			k_mat = [[f1^(0)(q+dh),f2^(0)(q+sh),...,fn^(0)(q+dh)],[f1^(1)(q+dh),
					f2^(1)(q+dh),...fn^(1)(q+dh)],...,[f1^(m-1)(q+dh),f2^(m-1)(q+dh),
					...fn^(m-1)(q+h)]]
					
			where fi^(j)(q+h) is the jth derivative of the generalized position
			of the ith object, evaluated at q+h."""
			
		k_mat = []
		for m in range(0,len(state[:-1])): # iter over orders of derivs up to m-1, inclusive
			k_list = []
			for n in range(0,len(state[m])): # iter through the object indices
				temp_s = list(state) # copy the state list
				temp_s[m][n] = state[m][n]+dh_mat[m][n] # add dh to the ith f^(j)
				k_list.append(h*derivatives(temp_s,h,params)[m][n])
			k_mat.append(list(k_list)) # append a copy of l; this is the sth k list
		return k_mat
				
	def dh_mat(k_mat,c):
		""" Returns the matrix of steps dh, given the previous matrix of k vals. 
			The output is in the format of input expected by k_mat(), i.e.:
			[[h1^1,h2^1,...,hn^1],[h1^2,h2^2,...,hn^2],...,[h1^m,h2^m,...,hn^m]]
			
			where hi^j = dh is the step in fi^(j)(q+dh)."""
		dh_mat = []
		for k_list in k_mat:
			dh_list = []
			for k in k_list:
				dh_list.append(k*c)
			dh_mat.append(list(dh_list))
		return dh_mat
	
	# Create the initial list of zeros to pass into k()
	dh_mat_0 = []
	for i in range(0,len(state[:-1])): # up to to m-1, inclusive
		dh_mat_0.append([0]*len(state[i]))
		
	# Generate the k matrices, where k(dh) def: f(q_i+dh) for systems which do 
	# not depend explicitly on the parameter (e.g. time) with respect to which 
	# we are differentiating.
	k1 = k_mat(dh_mat_0)
	k2 = k_mat(dh_mat(k1,1/2.))
	k3 = k_mat(dh_mat(k2,1/2.))
	k4 = k_mat(k3) # passing in k3 is equivalent to passing in dh_mat(k3,1)
	
	new_state = []
	for m in range(0,len(state[:-1])): # iter over deriv orders up to m-1, inclusive
		new_derivs = []
		for n in range(0,len(state[0])): # iter over coords up to q_n, inclusive
			new_derivs.append(state[m][n]+(k1[m][n]+2*(k2[m][n]+k3[m][n])+
								k4[m][n])/6.)
		new_state.append(list(new_derivs))
		
	# Get the mth derivatives with the analytical method passed in
	new_state.append(derivatives(state,h,params)[-1])
		
	return new_state
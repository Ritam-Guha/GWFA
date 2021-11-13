from initialization import initialize
from get_Function import get_function_details
import numpy as np
import copy

def evaluate(solutions, func):
	# function to evaluate a group of candidate solutions
	temp_solutions = copy.deepcopy(solutions)
	if len(solutions.shape) == 1:
		temp_solutions = temp_solutions.reshape(1, -1)

	func_val = np.zeros(temp_solutions.shape[0])
	for i, solution in enumerate(temp_solutions):
		func_val[i] = func(solution)

	return func_val

def rank(solutions, func_val):
	# function to rank the solutions
	num_solutions = solutions.shape[0]
	sorted_arg = np.argsort(func_val)

	return sorted_arg

def find_neighbor(solution, num_neighbors, dim, lb, ub):	
	# function to find neighbots for a particular solution
	neighbors = np.zeros((num_neighbors,dim))

	for neighbor_no in range(num_neighbors):
		chng_cnt = np.random.randint(dim)
		chng_pos = gen_Intergers(dim, chng_cnt)
		neighbors[neighbor_no] = solution.copy()

		for pos in chng_pos:
			cur_lb = lb[pos].copy()
			cur_ub = ub[pos].copy()
			neighbors[neighbor_no, pos] = np.random.rand(1) * (cur_ub - cur_lb) + cur_lb

	return neighbors

def loc_search(solution, func_val, dim, lb, ub, func):
	# function to implement local search for the DAs	
	num_neighbors = 5
	neighbors = find_neighbor(solution, num_neighbors, dim, lb, ub).copy()

	for neighbor_no in range(num_neighbors):
		cur_func_val = func(neighbors[neighbor_no])
		if(cur_func_val < func_val):
			func_val = cur_func_val.copy()
			solution = np.copy(neighbors[neighbor_no])

	return solution, func_val

def normalize(solution, lb, ub):
	# function to normalize a solution
	Flag4lb = solution<lb
	Flag4ub = solution>ub 
	solution = solution * (~(Flag4ub + Flag4lb)) + (ub * Flag4ub) + (lb * Flag4lb)
	return solution

def gen_Intergers(dim, count):
	# helper function to generate a permutation of integers
	temp = np.random.random(dim)
	pos = np.argsort(temp)[0:count]
	return pos

def GWFA(function_name, num_water_sources=1000, num_iteration=50, k=0.65, alpha=0.7, percent_DA=20, beta=0.5, gamma=0.5):
	# some preprocessing after getting the function to be optimized
	f, lb, ub, dim = get_function_details(function_name)
	ub = np.asarray(ub)
	lb = np.asarray(lb)
	boundary_no = np.shape(ub)[0]
	if boundary_no == 1:
		ub = np.repeat(ub,dim)
		lb = np.repeat(lb,dim)
	
	# initialize the water sources and calculate their objective values
	water_sources = initialize(num_water_sources, dim, lb, ub)
	func_val = evaluate(water_sources, f)

	# parametric values
	num_DA = int((percent_DA/100)*num_water_sources)
	num_RA = num_water_sources - num_DA
	DA = np.zeros((num_DA, dim))
	RA = np.zeros((num_RA, dim))
	
	# some useful parameters
	vel = np.zeros((num_water_sources, dim), dtype=float)
	fit_DA = np.zeros((1, num_DA),dtype=float)
	best_till_now_val = float('inf')
	best_till_now_water_source = np.zeros((1,dim))

	for iteration_no in range(num_iteration):	
		# iterations starts
		# rank the water sources in increasing order of objective values
		sorted_idx = rank(water_sources, func_val)	
		water_sources = water_sources[sorted_idx, :]
		func_val = func_val[sorted_idx]
		vel = vel[sorted_idx, :]

		# update the best solution till now
		if(func_val[0] < best_till_now_val):
			best_till_now_val = copy.deepcopy(func_val[0])
			best_till_now_water_source = water_sources[0, :]
		
		# print('Iteration ',iteration_no+1,':',best_till_now_val)

		# divide the solutions into RAs and DAs
		DA = water_sources[0:num_DA, :]
		RA = water_sources[num_DA:, :]
		fit_DA = func_val[0:num_DA]
		fit_RA = func_val[num_DA:]
		max_dif = abs(func_val[0] - func_val[num_water_sources-1])

		for cur_RA_idx in range(1, num_RA):			
			# process the RAs

			# first select an LA and a DA for the current RA
			cur_LA = sum(RA[0:cur_RA_idx])/num_RA
			cur_fit_LA = f(cur_LA)
			cur_DA_num = np.random.randint(num_DA)
			cur_DA = DA[cur_DA_num, :]
			cur_fit_DA = fit_DA[cur_DA_num]

			if(max_dif!=0):							
				# calculate L according to eqn. 10-11
				l_LA = max_dif / abs(cur_fit_LA - fit_RA[cur_RA_idx])
				l_DA = max_dif / abs(cur_fit_DA - fit_RA[cur_RA_idx])

				# calculate delH according to eqn. 8-9
				delH_LA = cur_LA - RA[cur_RA_idx, :]
				delH_DA = cur_DA - RA[cur_RA_idx, :]

				# calculate hydraulic gradients according to eqn. 12-13
				i_LA = delH_LA / l_LA
				i_DA = delH_DA / l_DA

				i = alpha*i_DA + (1-alpha)*i_LA

				# select the dimensions to be updated
				minchng_vel = 1 - ((iteration_no+1)/num_iteration)
				control_factor = minchng_vel * dim
				chng_cnt = np.random.randint(low=int(control_factor), high=dim)
				chng_pos = gen_Intergers(dim, chng_cnt)
				
				psi = chng_cnt/dim	

				# update the RAs
				if psi!=0:
					vel[num_DA + cur_RA_idx] = beta*vel[num_DA + cur_RA_idx] + gamma*(k*i)/psi
					RA[cur_RA_idx, chng_pos] = RA[cur_RA_idx, chng_pos] + vel[num_DA + cur_RA_idx][chng_pos]
					RA[cur_RA_idx] = normalize(RA[cur_RA_idx], lb, ub)
			else:
				RA[cur_RA_idx] = initialize(1, dim, lb, ub)

			fit_RA[cur_RA_idx] = f(RA[cur_RA_idx])


		for cur_DA_idx in range(num_DA):
			# process the DAs
			if(max_dif!=0):
				# perform the local search
				DA[cur_DA_idx], fit_DA[cur_DA_idx] = loc_search(DA[cur_DA_idx, :], fit_DA[cur_DA_idx], dim, lb, ub, f)
			else:
				DA[cur_DA_idx] = initialize(1, dim, lb, ub)
				fit_DA[cur_DA_idx] = f(DA[cur_DA_idx])

		# combine the updated DAs and RAs
		water_sources = np.append(DA, RA).reshape(num_water_sources, dim)
		func_val = np.append(fit_DA, fit_RA)
					
	sorted_idx = rank(water_sources,f)		
	water_sources = water_sources[sorted_idx, :]
	func_val = func_val[sorted_idx]
	vel = vel[sorted_idx]

	if(func_val[0]<best_till_now_val):
			best_till_now_val=func_val[0]
			best_till_now_water_source=water_sources[0, :]

	return best_till_now_water_source, best_till_now_val
	

			
if __name__ == "__main__":
	np.random.seed(0)
	GWFA(5, 100, "F1", 1)




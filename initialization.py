import numpy as np


def initialize(num_agents,dim,lb,ub):
	# function to initialize the solutions
	init_population=np.zeros((num_agents,dim),dtype=float)

	for cur_dim in range(dim):
		cur_ub=ub[cur_dim]
		cur_lb=lb[cur_dim]
		for agent_no in range(num_agents):
			init_population[agent_no,cur_dim]=np.random.rand(1)*(cur_ub-cur_lb)+cur_lb	
	return init_population

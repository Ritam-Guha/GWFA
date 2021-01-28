from initialization import initialize
from get_Function import get_function_details
import numpy as np


def rank(solutions,func):
	num_solutions=np.shape(solutions)[0]
	func_val=np.zeros((1,num_solutions),dtype=float)
	for solution_no in range(num_solutions):
		func_val[0,solution_no]=func(solutions[solution_no])
	sorted_arg=np.argsort(func_val)[0].copy()	
	solutions=solutions[sorted_arg].copy()
	func_val=func_val[0,sorted_arg]
	return solutions, func_val

def find_neighbor(solution,num_neighbors,dim,lb,ub):	
	neighbors=np.zeros((num_neighbors,dim))
	for neighbor_no in range(num_neighbors):
		chng_cnt=np.random.randint(dim)
		chng_pos=gen_Intergers(dim,chng_cnt)
		neighbors[neighbor_no]=solution.copy()
		for pos in chng_pos:
			cur_lb=lb[pos].copy()
			cur_ub=ub[pos].copy()
			neighbors[neighbor_no,pos]=np.random.rand(1)*(cur_ub-cur_lb)+cur_lb
	return neighbors

def loc_search(solution,dim,lb,ub,func):	
	num_neighbors=5
	neighbors=find_neighbor(solution,num_neighbors,dim,lb,ub).copy()
	func_val=func(solution).copy()	
	for neighbor_no in range(num_neighbors):
		cur_func_val=func(neighbors[neighbor_no]).copy()
		if(cur_func_val<func_val):
			func_val=cur_func_val.copy()
			solution=np.copy(neighbors[neighbor_no])
	# print('after:',func_val)
	return solution

def normalize(solution,lb,ub):
	Flag4lb=solution<lb
	Flag4ub=solution>ub 
	solution=solution*(~(Flag4ub+Flag4lb)) + ub*Flag4ub + lb*Flag4lb
	# solution=np.minimum(solution,ub)
	# solution=np.maximum(solution,lb)
	return solution

def gen_Intergers(dim,count):
	temp=np.random.random((1,dim))
	pos=np.argsort(temp)[0][0:count]
	return pos


def GWFA(num_water_sources,num_iteration,function_name,num_runs, alfa_val, k_val):
    f,lb,ub,dim=get_function_details(function_name)
    ub=np.asarray(ub)
    lb=np.asarray(lb)
    boundary_no=np.shape(ub)[0]
    if boundary_no==1:
        ub=np.repeat(ub,dim)
        lb=np.repeat(lb,dim)
        
    water_sources=initialize(num_water_sources,dim,lb,ub)
    
    
    k=k_val
    alpha=alfa_val
    num_best=int(0.2*num_water_sources)
    best_val_in_each_itr = np.zeros(num_iteration) # to store the best values in each iteration
    vel=np.zeros((num_water_sources,dim),dtype=float)
    fitGbest=np.zeros((1,num_best),dtype=float)
    avgFitGbest=0
    best_till_now_val=float('inf')
    best_till_now_water_source=np.zeros((1,dim))

    for iteration_no in range(num_iteration):
        water_sources, func_val=rank(water_sources,f)		
		
        if(func_val[0]<best_till_now_val):
            best_till_now_val=func_val[0].copy()
            best_till_now_water_source=water_sources[0].copy()
        
        best_val_in_each_itr[iteration_no] = best_till_now_val
        gbest=water_sources[0:num_best].copy()
        fitGbest=func_val[0:num_best].copy()
        control_factor=1-((iteration_no+1)/num_iteration)
        max_dif=abs(fitGbest[0]-func_val[num_water_sources-1])
		
        for water_source_no in range(num_best,num_water_sources):		
            if water_source_no>=num_best:		
                if water_source_no==num_best:
                    lbest=water_sources[water_source_no].copy()
				
                else:
                    lbest=sum(water_sources[num_best:water_source_no])/num_water_sources
				
                fitLbest=f(lbest)
                cur_num=np.random.randint(num_best)
				
                if(max_dif!=0):									
                    l_lbest=max(min(abs(fitLbest-func_val[water_source_no])/max_dif,1),0.001)			
                    l_gbest=max(abs(fitGbest[cur_num]-func_val[water_source_no])/max_dif,0.001)			
					
                    delH_lbest=(lbest-water_sources[water_source_no])					
                    delH_gbest=(gbest[cur_num]-water_sources[water_source_no])					
					
                    i_lbest=delH_lbest/l_lbest
                    i_gbest=delH_gbest/l_gbest
					
                    i= alpha*delH_gbest + (1-alpha)*delH_lbest

                    minchng_vel=1-((iteration_no+1)/num_iteration)					
                    chng_cnt=np.random.randint(low=int(minchng_vel*dim), high=dim)					
                    chng_pos=gen_Intergers(dim,chng_cnt)
                    psi=chng_cnt/dim	
					
                    if psi!=0:
                        vel[water_source_no]=0.5*vel[water_source_no]+0.5*(k*i)/psi
                        water_sources[water_source_no,chng_pos]=water_sources[water_source_no,chng_pos]+vel[water_source_no][chng_pos]
                        water_sources[water_source_no]=normalize(water_sources[water_source_no],lb,ub)
				
                else:
                    water_sources[water_source_no]=initialize(1,dim,lb,ub).copy()				
		
        for water_source_no in range(num_best):
            if max_dif!=0:
				# if control_factor>np.random.random(1):
				# 	water_sources[water_source_no]=((ub+lb)-water_sources[water_source_no])
				# else:
				# print('prev:',f(water_sources[water_source_no]))
                water_sources[water_source_no]=loc_search(water_sources[water_source_no],dim,lb,ub,f).copy()
			
            else:
                water_sources[water_source_no]=initialize(1,dim,lb,ub).copy()				
				# print('next:',f(water_sources[water_source_no]))
        print('Iteration ',iteration_no+1,':',best_till_now_val)	
	
    water_sources, func_val=rank(water_sources,f)		
	
    if(func_val[0]<best_till_now_val):
        best_till_now_val=func_val[0].copy()
        best_till_now_water_source=water_sources[0].copy()

    return best_till_now_water_source, best_till_now_val, best_val_in_each_itr

			





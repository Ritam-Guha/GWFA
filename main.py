from GWFA import GWFA
import numpy as np
from get_Function import get_function_details

num_runs = 2
num_water_sources = 50
num_iteration = 1000

function_name = 'F1'
f,lb,ub,dim = get_function_details(function_name)
best_val = np.zeros(num_runs)
best_water_source = np.zeros((num_runs, dim),dtype=float)

for run_no in range(num_runs):
	best_water_source[run_no, :], best_val[run_no] = GWFA(function_name, num_water_sources, num_iteration)
	print('Run ', run_no+1)
	print("Best value:", best_val[run_no])

print("Min Value: ", min(best_val))
print("Average Value: ", np.average(best_val))
print("Std Deviation: ", np.std(best_val))

from GWFA import GWFA
import numpy as np
from get_Function import get_function_details

num_runs=1
num_water_sources=50
num_iteration=1000
``
for fno in range (1,2):
    function_name='F' + str(fno)
    f,lb,ub,dim=get_function_details(function_name)
    best_val=np.zeros((num_runs,1))
    best_water_source=np.zeros((num_runs,dim),dtype=float)
    
    #print("Begining execution of Funtion number: ", fno)
    alfa_val = 0.65
    k_val = 0.7
    
    for run_no in range(num_runs):
        best_water_source[run_no],best_val[run_no,0], best_val_arr = GWFA(num_water_sources,num_iteration,function_name,num_runs, alfa_val, k_val)
    	# print('Run ',run_no+1)\
        print("Best value:", best_val[run_no,0])
    	# print("Best water source:", best_water_source[run_no])
    
    print("Min Value: ", min(best_val[:,0]))
    print("Avg Value: ", np.average(best_val[:,0]))
    print("Std Deviation: ", np.std(best_val[:,0]))
    #print(best_val_arr)
    #with open(r'convergence_test.txt', 'a') as f:
        # f.write(function_name+ " : ")
        #f.write(",".join(map(str, best_val_arr)))
        #f.write("\n")
    

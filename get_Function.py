import numpy as np
import math

def get_function_details(function_name):
	# function which returns the function definition 
	# based on the name
	if function_name=='F1':
		fobj = F1
		lb=[-100]
		ub=[100]
		dim=30
    
	if function_name=='F2':
		fobj = F2
		lb=[-100]
		ub=[100]
		dim=30

	if function_name=='F3':
		fobj = F3
		lb=[-100]
		ub=[100]
		dim=30       

	if function_name=='F4':
		fobj = F4
		lb=[-100]
		ub=[100]
		dim=30
        

	if function_name=='F5':
		fobj = F5
		lb=[-30]
		ub=[30]
		dim=30  
        

	if function_name=='F6':
		fobj = F6
		lb=[-100]
		ub=[100]
		dim=30 
        
	if function_name=='F7':
		fobj = F7
		lb=[-1.28]
		ub=[1.28]
		dim=30
        
	if function_name=='F8':
		fobj = F8
		lb=[-500]
		ub=[500]
		dim=30        

	if function_name=='F9':
		fobj = F9
		lb=[-5.12]
		ub=[5.12]
		dim=30        

	if function_name=='F10':
		fobj = F10
		lb=[-32]
		ub=[32]
		dim=30


	if function_name=='F11':
		fobj = F11
		lb=[-600]
		ub=[600]
		dim=30

	if function_name=='F12':
		fobj = F12
		lb=[-50]
		ub=[50]
		dim=30

	if function_name=='F13':
		fobj = F13
		lb=[-50]
		ub=[50]
		dim=30

	if function_name=='F14':
		fobj = F14
		lb=[-65.536]
		ub=[65.536]
		dim=2

	if function_name=='F15':
		fobj = F15
		lb=[-5]
		ub=[5]
		dim=4
        
	if function_name=='F16':
		fobj = F16
		lb=[-5]
		ub=[5]
		dim=2

	if function_name=='F17':
		fobj = F17
		lb=[-5,0]
		ub=[10,15]
		dim=2

	if function_name=='F18':
		fobj = F18
		lb=[-2]
		ub=[2]
		dim=2

	if function_name=='F19':
		fobj = F19
		lb=[0]
		ub=[1]
		dim=3
        
	if function_name=='F20':
		fobj = F20
		lb=[0]
		ub=[1]
		dim=6

	if function_name=='F21':
		fobj = F21
		lb=[0]
		ub=[10]
		dim=4

	if function_name=='F22':
		fobj = F22
		lb=[0]
		ub=[10]
		dim=4

	if function_name=='F23':
		fobj = F23
		lb=[0]
		ub=[10]
		dim=4
        
    #Tension/Compression Spring Design Problem    
	if function_name=='F24':
		fobj = F24
		lb=[0.05, 0.25, 2.00]
		ub=[2.00, 1.30, 15.00]
		dim=3
    #Gear Train Design Problem    
	if function_name=='F25':
		fobj = F25
		lb=[12]
		ub=[60]
		dim=4
    #Welded Beam Design Problem    
	if function_name=='F26':
		fobj = F26
		lb=[0.1, 0.1, 0.1, 0.1]
		ub=[2.00, 10.00, 10.00, 2.00]
		dim=4
    #Pressure Design Vessel Problem    
	if function_name=='F27':
		fobj = F27
		lb=[0.00, 0.00, 10.00, 10.00]
		ub=[100.00, 100.00, 200.00, 200.00]
		dim=4
    #Closed Coil Helical Spring Design Problem    
	if function_name=='F28':
		fobj = F28
		lb=[0.508, 1.270, 15.00]
		ub=[1.016, 7.620, 25.00]
		dim=3

        
	return fobj, lb, ub, dim


# function definition begins
        
def F1(x):
	return(sum(x ** 2))

def F2(x):
    return (sum(abs(x) + np.prod(abs(x))))

def F3(x):
    l = len(x)
    val = 0
    for i in range(l):
        val = val + sum(x[0:i+1]) * sum(x[0:i+1])
    return val

def F4(x):
	return(max(abs(x)))
    
def F5(x):
    #return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    dim = len(x)
    x = np.array(x)
    return np.sum(100*(x[1:dim]-(x[0:dim-1]**2))**2+(x[0:dim-1]-1)**2)
    
def F6(x):
    y = np.array(x)
    return sum((np.floor(y+.5) ** 2))
    
def F7(x): 
    dim = len(x)
    y = np.array(x)
    return sum(np.arange(1, dim+1)*(y**4))+np.random.uniform()
    
def F8(x):  
    y = np.array(x)
    return sum(-y * (np.sin(np.sqrt(abs(y)))))
    
def F9(x):
    dim = len(x)
    y = np.array(x)
    return sum(y**2 - 10 * np.cos(2 * math.pi * y)) + 10 * dim
    
def F10(x):
    dim = len(x)
    y = np.array(x)
    return -20* np.exp(-0.2*np.sqrt(sum(y**2)/dim)) - np.exp(sum(np.cos(2 * math.pi * y))/dim) + 20 + np.exp(1)

def F11(x):
    dim = len(x)
    y = np.array(x)
    return sum(y**2)/4000 - np.prod(np.cos(y / (np.sqrt(np.arange(1,dim+1))))) + 1
    
def F12(x):
    
    dim = len(x)
    y = np.array(x)
    val1 = ((math.sin(math.pi * (1+(y[0]+1)/4))) ** 2)
    val2 = ((y[dim-1]+1)/4) ** 2
    val3 = np.sum(Ufun(y, 10, 100, 4))
    val4 = np.sum((((y[0:dim-1]+1)/4) ** 2) * (1+10 * ((np.sin(math.pi * (1+(y[1:dim]+1)/4)))) ** 2))
    val = (math.pi/dim) * (10 * val1 + val4 + val2) + val3
    return val
 

def F13(x):
    dim = len(x)
    y = np.array(x)    
    val1 = np.sum(Ufun(x,5,100,4))
    val2 = np.sum((y[0:dim-1]-1) **2 * (1 + (np.sin(3 * math.pi * y[1:dim])) ** 2))
    val = 0.1*((math.sin(3*math.pi*y[0])) ** 2 + val2 + ((y[dim-1]-1) ** 2)*(1+(math.sin(2*math.pi*y[dim-1]))**2)) + val1
    return val

def F14(x):
    y = np.array(x)
    aS=[[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],[-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]];          
    aS = np.array(aS)
    bS = np.empty(25)
    for j in range(0,25):
        bS[j] = np.sum((np.transpose(y) - aS[:,[j]]) ** 6)         

    return (1/500 + np.sum(1/(np.arange(1,26)+bS))) ** (-1)

def F15(x):
    y = np.array(x)
    aK=[.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246]
    aK = np.array(aK)
    bK=[.25,.5,1,2,4,6,8,10,12,14,16]
    bK = np.array(bK)
    bK=1/bK
    
    return np.sum((aK-((y[0]*(bK ** 2 + y[1] * bK))/ (bK ** 2+y[2] * bK + y[3]))) ** 2)
    
def F16(x):
    y = np.array(x)
    
    return 4*(y[0]**2) - 2.1*(y[0]**4) + (y[0] ** 6)/3 + y[0]*y[1] - 4*(y[1] ** 2) + 4 * (y[1] ** 4)
    
def F17(x):
    x = np.array(x)
    return (x[1]-(x[0] ** 2)*5.1/(4*(math.pi ** 2))+5/math.pi*x[0]-6)**2+10*(1-1/(8*math.pi))*math.cos(x[0])+10;

def F18(x):
    x = np.array(x)

    return (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*(x[0]**2)-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*(x[0]**2)+48*x[1]-36*x[0]*x[1]+27*(x[1]**2)));    

def F19(x):
    x = np.array(x)

    aH=[[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]]
    cH=[1,1.2,3,3.2]
    pH=[[.3689,.117,.2673],[.4699,.4387,.747],[.1091,.8732,.5547],[.03815,.5743 ,.8828]]
    aH = np.array(aH)
    cH = np.array(cH)
    pH = np.array(pH)
    
    val = 0
    
    for i in range(0,4):
        val=val-cH[i]*np.exp(-(np.sum(aH[i,:]*((x-pH[i,:])**2))));

    return val

def F20(x):
    x = np.array(x)   
    aH=[[10,3,17,3.5,1.7,8],[.05,10,17,.1,8,14],[3,3.5,1.7,10,17,8],[17,8,.05,10,.1,14]]
    cH=[1,1.2,3,3.2]
    pH=[[.1312,.1696,.5569,.0124,.8283,.5886],[.2329,.4135,.8307,.3736,.1004,.9991],[.2348,.1415,.3522,.2883,.3047,.6650],[.4047,.8828,.8732,.5743,.1091,.0381]]
    
    aH = np.array(aH)
    cH = np.array(cH)
    pH = np.array(pH)
    
    val = 0
    for i in range(0,4):
        val=val-cH[i]*np.exp(-(np.sum(aH[i,:]*((x-pH[i,:])**2))));
    
    return val
    
    
def F21(x):
    x = np.array(x)
    
    aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5]
    aSH = np.array(aSH)
    cSH = np.array(cSH)
    val = 0
    for i in range(0,5):
        val=val-(np.sum((x-aSH[i,:]) * np.transpose(x-aSH[i,:])) + cSH[i]) ** (-1);
    return val
    

def F22(x):
    x = np.array(x)
    
    aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5]
    aSH = np.array(aSH)
    cSH = np.array(cSH)
    val = 0
    for i in range(0,7):
        val=val-(np.sum((x-aSH[i,:]) * np.transpose(x-aSH[i,:])) + cSH[i]) ** (-1);
    return val    
    
def F23(x):
    x = np.array(x)
    
    aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5]
    aSH = np.array(aSH)
    cSH = np.array(cSH)
    val = 0
    for i in range(0,10):
        val=val-(np.sum((x-aSH[i,:]) * np.transpose(x-aSH[i,:])) + cSH[i]) ** (-1);
    return val   

  
def Ufun(x, a, k, m):
    y = np.array(x)
    return k*((y-a)**m)*(x>a) + k*((-y-a)**m)*(x<(-a))
    
def F24(x):
    x = np.array(x)
    return (x[2] + 2)*x[1]*(x[0]**2)
    	
def F25(x):
    x = np.array(x)
    return ((1/6.931) - ((x[2]*x[1]) / (x[0]*x[3]))) ** 2

def F26(x):
    x = np.array(x)
    return 1.1047*(x[0]**2)*x[1] + 0.04811*x[2]*x[3]*(14.0 + x[1])    
    
def F27(x):
    x = np.array(x)
    val1 = 0.6224*x[0]*x[2]*x[3]
    val2 = 1.7781 * x[1] * (x[2]**2)
    val3 = 3.1661 * (x[0]**2) * x[3]
    val4 = 19.84 * (x[0] **2) * x[2]
    return val1+val2+val3+val4

def F28(x):
    x = np.array(x)
    return ((math.pi**2)/4) * (x[2] + 2) * x[1] * (x[0]**2)    
    
    
    
  
    
    
    
    
    
    
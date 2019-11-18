import numpy as np

def parameters(x_max): 
  
    # Iteration parameters:

    sig=0.25
    d=0.1
    r0=100
    T=0.1
    dx=0.1 # Default dx = 0.1
    sig_ksi=1.5*x_max # Deafult between 0.3 to 0.4 of x_max
    rate=r0/dx*2*sig*np.sqrt(2*np.pi) 
    t_size = round(rate*T) # t_size = 1253
    x_size = round((2*x_max)/dx)
    N_features = x_size*2*t_size

    return int(x_size), int(t_size), N_features
import numpy as np
from iteration import iteration

def data_generator(N_vecs,x_max,prior,n_sigs,D): # N_vecs should be something like 1.2*vec_size
    
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
    
    while True:

      # Data set parameters:
      
      data = np.zeros((N_vecs,2,int(x_size), int(t_size)))
      d_vec = d*((np.random.randint(2, size = N_vecs))*2 - 1)
      lables = ((d_vec*10)+1)*0.5
      count = 0

      # Creating the data set

      while count < N_vecs:
      
          vec_temp , n = iteration(d_vec[count], rate, D, T, sig_ksi, t_size, x_size, x_max, sig, prior, n_sigs)
          if n==1:
              data[count,:,:,:] = vec_temp
              count += 1   

      yield (data , lables.astype(int))
import numpy as np
from Iteration_poly import iteration

def data_generator(N_vecs,x_max,prior,n_sigs,D): 

    # Iteration parameters:

    sig=0.25
    d=0.1
    r0=100
    T=0.1
    dx=0.1 # Default dx = 0.1
    # x_max=2.4*np.sqrt(D*T)+3 # Defined by main script
    sig_ksi=1.5*x_max # Deafult between 0.3 to 0.4 of x_max
    rate=r0/dx*2*sig*np.sqrt(2*np.pi) 
    t_size = round(rate*T) # t_size = 1253
    x_size = round((2*x_max)/dx)
    const = D/((sig**2)*rate)

    # Data set parameters:

    #vec_size = x_size*t_size*2 # Length of each data vector, cuurently 4*10^4
    data = np.zeros((int(N_vecs), int(x_size*2)))
    d_vec = d*((np.random.randint(2, size = N_vecs))*2 - 1); # Creating the discrimination vector
    fasle_count = 0
    vec_count = 0
    i = 0
    false_count = 0

    # Creating the data set

    while vec_count < N_vecs:
        result = iteration(d_vec[i], rate, D, T, sig_ksi, t_size, x_size, x_max, sig, prior, n_sigs)
        vec_temp , n = result
        if n==1:
            data[i,:] = vec_temp
            vec_count += 1
            i += 1
        else:
            false_count += 1

    percent = false_count/(false_count+vec_count)*100    

    # sanity check: print(data[4,:],np.size(data),np.count_nonzero(data))

    lables = d_vec*10
    return data , lables , x_size, t_size, percent
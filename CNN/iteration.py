import numpy as np

def iteration(d, rate, D, T, sig_ksi, t_size, x_size, x_max, sig, prior, n_sigs):

    S = np.zeros((2,int(x_size), int(t_size)),  dtype=np.int_)
    S1 = S[0,:,:]
    S2 = S[1,:,:]
    
    # Setting initial prior:
    
    if prior == 'delta':
        X0 = 0
    elif prior == 'uniform':
        X0 = np.random.uniform(-x_max+n_sigs*sig, x_max-n_sigs*sig)
    # X0 = np.random.normal(0, sig_ksi) # For gaussian distribution
    t_count = 0

    while t_count < T:
            
            # Advancing the stimulus position by a random walk distance in a dt
            
            t = np.random.exponential(1/rate)
            t_count= t_count + t
            sig_heat = np.sqrt(2*D*t)
            X0 = np.random.normal(X0,sig_heat)
            
            # Randomly picking a population an registering the time and position of the spike
            
            pop = np.random.randint(1,3)
            ind_t = int(round(t_count/T*t_size))

            if ind_t == 0:
                ind_t = 1
            elif ind_t>t_size:
                break
            
            if pop == 1:
                X1 = np.random.normal(X0+(d/2),sig)
                ind_x = int(round((X1/x_max)*(x_size/2) + x_size/2))
                # If stimulus is out of limits over:
                if ind_x > x_size :
                    break
                    # ind_x = x_size
                    # disp('Over maximum') 
                # If stimulus is out of limits under:
                elif ind_x <= 0:
                    break
                    # ind_x = 1
                    # disp('Under minimum')
                S1[ind_x-1,ind_t-1] = 1
                
            else:
                X2 = np.random.normal(X0-(d/2),sig)
                ind_x = int(round((X2/x_max)*(x_size/2) + x_size/2))
                if ind_x > x_size:
                    break
                    # ind_x = x_size
                    # disp('Over maximum') 
                elif ind_x <= 0:
                    break
                    # ind_x = 1
                    # disp('Under minimum')
                S2[ind_x-1,ind_t-1] = 1

                
    if t_count >= T:
        count = 1
    else:
        count = 0

    return S , count
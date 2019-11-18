import numpy as np 
import pickle
from scipy.sparse import csr_matrix , vstack
from sklearn.svm import LinearSVC , SVC
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from Data_Generator_poly import data_generator
from sklearn.externals import joblib
import time 

'''
Combinations of parameters:

1 D=100, Prior = delta
2 D=100, Prior = uniform (93%)
'''

# General settings

N_step = 10**3
max_iter = 10**5 
x_max = 11 
N_classifiers = 2

data_check , _, x_size, t_size, _ = data_generator(N_vecs=1,x_max=x_max,prior='delta',n_sigs='_', D=0)
N_features = data_check.size
N_final = 200*N_features                                          
sher = N_final%(N_step)
N_final = int(N_final - sher)

# Creating lists of parameters and arrays for saving

prior_list = ['delta', 'uniform']
D_list = [100,100]

acc_per = np.zeros((N_classifiers+1,2))
weights = np.zeros((N_classifiers,N_features))
print('Entering loop')

# Loop of clasifiers

for i in range(N_classifiers):

    counter = 0
    labels_vec = np.zeros(N_final)
    per_tot = 0

    # First build

    temp_data , lables, _, _ , per_tot = data_generator(N_step , x_max , prior=prior_list[i] , n_sigs=3, D=D_list[i])
    labels_vec[counter:counter+N_step] = lables
    temp_data = csr_matrix(temp_data)
    data = temp_data
    counter += N_step
    n = 1

    # Addition to the data

    while counter < N_final:

        temp_data , lables, _, _, percent = data_generator(N_step , x_max , prior=prior_list[i] , n_sigs=3, D=D_list[i])
        labels_vec[counter:counter+N_step] = lables
        temp_data = csr_matrix(temp_data)
        data = vstack([data,temp_data])
        counter += N_step
        n += 1
        per_tot += percent
        print(counter/N_final*100 , i+1)

    # Scaling and splitting data to test and train

    scale = StandardScaler(copy=False, with_mean=False, with_std=True)
    data = scale.fit_transform(data)
    data_train , data_test, lables_train,  lables_test = train_test_split(data, labels_vec, test_size=0.2) 

    # Creating polynomial moderl without grid-search

    start = time.time()
    pol_clf = SVC(kernel='poly' , degree=2, max_iter = max_iter, cache_size=2000)
    pol_clf.fit(data_train, lables_train)
    end = time.time()
    print(f'Done pol_clf {i+1}, took {end-start} seconds')

    # Saving polynomial classifier to file using joblib

    filename = f'Poly summed on time D=100, {i+1}.joblib'
    joblib.dump(pol_clf, filename)

    # Getting accuracy, weights and false percentage

    acc_per[i,0] = pol_clf.score(data_test,lables_test)
    print(f'D={D_list[i]}, Prior:{prior_list[i]}, Score:{acc_per[i,0]}, bias: {per_tot/n}')
    acc_per[i,1] = per_tot/n

# Writing to CSV 

acc_per[N_classifiers,0] = x_size
acc_per[N_classifiers,1] = t_size

np.savetxt("Accuracy_2_poly_summed_on_time_D=100.csv", acc_per ,fmt='%10.5f', delimiter=",")
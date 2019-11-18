import numpy as np 
import pickle
from scipy.sparse import csr_matrix , vstack
from sklearn.svm import LinearSVC , SVC
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from Data_Generator_d import data_generator
from sklearn.externals import joblib

# General settings

N_step = 10**3
max_iter = 10**4 
x_max = 10             
N_classifiers = 6
d_list = [0.05, 0.1, 0.5, 1, 5, 10]

data_check , _, x_size, t_size, _ = data_generator(N_vecs=1,x_max=x_max,prior='delta',n_sigs='_', D=0, d=0.1)
N_features = data_check.size
N_final = 4*N_features                                          
sher = N_final%(N_step)
N_final = int(N_final - sher)

# Creating lists of parameters and arrays for saving

acc_per = np.zeros((N_classifiers+1,2))
weights = np.zeros((N_classifiers,N_features))
print('Entering loop')

# Loop of clasifiers

for i in range(N_classifiers):

    counter = 0
    labels_vec = np.zeros(N_final)
    per_tot = 0

    # First build

    temp_data , lables, _, _ , per_tot = data_generator(N_step , x_max , prior='delta' , n_sigs=0, D=0, d=d_list[i])
    labels_vec[counter:counter+N_step] = lables
    temp_data = csr_matrix(temp_data)
    data = temp_data
    counter += N_step
    n = 1

    # Addition to the data

    while counter < N_final:

        temp_data , lables, _, _, percent = data_generator(N_step , x_max , prior='delta' , n_sigs=0, D=0, d=d_list[i])
        labels_vec[counter:counter+N_step] = lables
        temp_data = csr_matrix(temp_data)
        data = vstack([data,temp_data])
        counter += N_step
        n += 1
        per_tot += percent
        print(counter/N_final*100 , i)

    # Scaling and splitting data to test and train

    scale = StandardScaler(copy=False, with_mean=False, with_std=True)
    data = scale.fit_transform(data)
    data_train , data_test, lables_train,  lables_test = train_test_split(data, labels_vec, test_size=0.2) 

    # Creating linear model withought grid-search

    print('Starting lin_clf ', i)
    lin_clf = LinearSVC(max_iter = max_iter , dual = False)
    lin_clf.fit(data_train, lables_train)
    print('Done lin_clf', i)

    # Saving linear classifier to file using joblib

    filename = f'Linear model d= {d_list[i]}.joblib'
    joblib.dump(lin_clf, filename)

    # Getting accuracy, weights and false percentage

    acc_per[i,0] = lin_clf.score(data_test,lables_test)
    acc_per[i,1] = per_tot/n
    weights[i,:] = np.ravel(lin_clf.coef_)
    print('Done writing weights, moving to i=',i+1)

# Writing to CSV 

acc_per[N_classifiers,0] = x_size
acc_per[N_classifiers,1] = t_size

np.savetxt("Weights_D=0_Different_d.csv", weights ,fmt='%10.5f', delimiter=",")
np.savetxt("Accuracy_D=0_Different_d.csv", acc_per ,fmt='%10.5f', delimiter=",")


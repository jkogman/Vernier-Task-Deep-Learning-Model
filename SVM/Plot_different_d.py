import numpy as np 
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# Loading results and parameters

weights = np.genfromtxt('Weights_D=0_Different_d.csv',delimiter=',')
acc_per = np.genfromtxt('Accuracy_D=0_Different_d.csv',delimiter=',')

x_max = 10
N_classifiers = 6
d_list = [0.05, 0.1, 0.5, 1, 5, 10]
acc_lin = acc_per[:N_classifiers,0]
bias = acc_per[:N_classifiers,1]
x_size = acc_per[N_classifiers,0]
t_size = acc_per[N_classifiers,1]

weights_up = weights[:,0:int(np.size(weights,axis=1)/2)]
weights_down= weights[:,int(np.size(weights,axis=1)/2):]

'''
weights_up = np.zeros((N_classifiers,int(x_size*t_size)))
weights_down = np.zeros((N_classifiers,int(x_size*t_size)))

for i in range(N_classifiers):
    filename = f'Linear model d= {d_list[i]}.joblib'
    lin_clf = joblib.load(filename)
    weights = np.ra
12.0 5
12.5 5
13.0 5vel(lin_clf.coef_)
    if i==0:
        weights_up = np.zeros((N_classifiers,int(np.size(weights)/2)))
        weights_down = np.zeros((N_classifiers,int(np.size(weights)/2)))
    weights_up[i,:] = weights[0:int(np.size(weights)/2)]
    weights_down[i,:] = weights[int(np.size(weights)/2):]
'''

# Reshaping to space-time matrix and averaging over time for each neuron

weights_up_mat = np.zeros((N_classifiers,int(x_size),int(t_size)))
weights_down_mat = np.zeros((N_classifiers,int(x_size),int(t_size)))
weights_up_avg = np.zeros((N_classifiers,int(x_size)))
weights_down_avg = np.zeros((N_classifiers,int(x_size)))

for i in range(N_classifiers):

    weights_up_mat[i,:,:] = np.reshape(weights_up[i,:],(int(x_size),int(t_size)))
    weights_down_mat[i,:,:] = np.reshape(weights_down[i,:],(int(x_size),int(t_size)))

    weights_up_avg[i,:] = np.mean(weights_up_mat[i,:,:],axis=1)
    weights_down_avg[i,:] = np.mean(weights_down_mat[i,:,:],axis=1)

# Plots

x = np.linspace(-x_max,x_max,np.size(weights_down_avg,axis=1))
n = 0

fig,axs = plt.subplots(2,3)
fig.suptitle("D=0, Delta prior, different d's", fontsize = 18)

for i in range(2):
    for j in range(3):

        axs[i,j].plot(x,weights_up_avg[n,:],'b', x, weights_down_avg[n,:],'r')
        axs[i,j].set_title(f'd = {d_list[n]}, Score: {acc_lin[n]}, Bias:{bias[n]}')
        n+=1

#fig.legend([l_up,l_down],['Up population','Down population'],loc=(0.82,0.94),borderaxespad=0.3)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Position of neuron [arcmin.]" , fontsize = 14)
plt.ylabel("Weight", fontsize = 14)

plt.show()
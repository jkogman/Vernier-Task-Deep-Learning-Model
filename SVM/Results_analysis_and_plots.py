import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Loading results and parameters

weights = np.genfromtxt('Weights_test_maxiter=10^4_noGS.csv',delimiter=',')
acc_per = np.genfromtxt('Accuracy_test_maxiter=10^4_noGS_linear_poly.csv',delimiter=',')

x_max = 11
N_classifiers = 6
acc_lin = acc_per[:N_classifiers,0]
acc_pol = acc_per[:N_classifiers,1]
bias = acc_per[:N_classifiers,2]
per = acc_per[:,2]
x_size = acc_per[N_classifiers,0]
t_size = acc_per[N_classifiers,1]

# Up and down weigWeights_test_maxiter=10^4_noGShts as func of space for each trial (1)

weights_up = weights[:,0:int(np.size(weights,axis=1)/2)]
weights_down= weights[:,int(np.size(weights,axis=1)/2):]

# Reshaping to space-time matrix (3) and averaging over time for each neuron (2)

weights_up_mat = np.zeros((6,int(x_size),int(t_size)))
weights_down_mat = np.zeros((6,int(x_size),int(t_size)))
weights_up_avg = np.zeros((6,int(x_size)))
weights_down_avg = np.zeros((6,int(x_size)))

for i in range(6):

    weights_up_mat[i,:,:] = np.reshape(weights_up[i,:],(int(x_size),int(t_size)))
    weights_down_mat[i,:,:] = np.reshape(weights_down[i,:],(int(x_size),int(t_size)))

    weights_up_avg[i,:] = np.mean(weights_up_mat[i,:,:],axis=1)
    weights_down_avg[i,:] = np.mean(weights_down_mat[i,:,:],axis=1)


# Ploting (1)

x = np.linspace(-x_max,x_max,num = np.size(weights_up,axis=1))

fig , axs = plt.subplots(2,3)
fig.suptitle("Weights for both populations as a funcion of neuron's position at different times", fontsize=18)

l_up = axs[0,0].plot(x,weights_up[0,:],color='b')[0]
l_down = axs[0,0].plot(x,weights_down[0,:],color='r')[0]
axs[0,0].set_title(f'D = 0 , Prior = delta, Bias={bias[0]}% \n Linear:{acc_lin[0]}%,  Poly:{acc_pol[0]}%')

axs[0,1].plot(x,weights_up[1,:],'b', x, weights_down[1,:],'r')
axs[0,1].set_title(f'D = 0 , Prior: uniform (93% of retina), Bias={bias[1]}% \n Linear:{acc_lin[1]}%,  Poly:{acc_pol[1]}%')

axs[0,2].plot(x,weights_up[2,:],'b', x, weights_down[2,:],'r')
axs[0,2].set_title(f'D = 100 , Prior = delta, Bias={bias[2]}% \n Linear:{acc_lin[2]}%,  Poly:{acc_pol[2]}%')

axs[1,0].plot(x,weights_up[3,:],'b', x, weights_down[3,:],'r')
axs[1,0].set_title(f'D = 100 , Prior = uniform (40% of retina), Bias={bias[3]}% \n Linear:{acc_lin[3]}%,  Poly:{acc_pol[3]}%')

axs[1,1].plot(x,weights_up[4,:],'b', x, weights_down[4,:],'r')
axs[1,1].set_title(f'D = 0 , Prior = uniform (68% of retina), Bias={bias[4]}% \n Linear:{acc_lin[4]}%,  Poly:{acc_pol[4]}%')

axs[1,2].plot(x,weights_up[5,:],'b', x, weights_down[5,:],'r')
axs[1,2].set_title(f'D = 0 , Prior = uniform (93% of retina), Bias={bias[5]}% \n Linear:{acc_lin[5]}%,  Poly:{acc_pol[5]}%')

fig.legend([l_up,l_down],['Up population','Down population'],loc=(0.82,0.94),borderaxespad=0.3)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Position of neuron [arcmin.]" , fontsize = 14)
plt.ylabel("Weight", fontsize = 14)

plt.show()

x = np.linspace(-x_max,x_max,num = np.size(weights_up_avg,axis=1))
plt.plot(x,weights_up_avg[0,:],'.',color='b')
plt.plot(x,weights_down_avg[0,:],'.',color='r')[0]
plt.xlim((-1,1))
plt.grid()
plt.show()


# Plotting (2)

x = np.linspace(-x_max,x_max,num = np.size(weights_up_avg,axis=1))

fig , axs = plt.subplots(2,3)
fig.suptitle("Weights for both populations, averaged over time, as a function of the neuron's position", fontsize=18)

l_up = axs[0,0].plot(x,weights_up_avg[0,:],color='b')[0]
l_down = axs[0,0].plot(x,weights_down_avg[0,:],color='r')[0]
axs[0,0].set_title(f'D = 0 , Prior = delta\n Accuracy:{acc_pol[0]}%')

axs[0,1].plot(x,weights_up_avg[1,:],'b', x, weights_down_avg[1,:],'r')
axs[0,1].set_title(f'D = 0 , Prior: uniform (93% of retina)\n Accuracy:{acc_pol[1]}%')

axs[0,2].plot(x,weights_up_avg[2,:],'b', x, weights_down_avg[2,:],'r')
axs[0,2].set_title(f'D = 100 , Prior = delta\n Accuracy:{acc_pol[2]}%')

axs[1,0].plot(x,weights_up_avg[3,:],'b', x, weights_down_avg[3,:],'r')
axs[1,0].set_title(f'D = 100 , Prior = uniform (40% of retina)\n Accuracy:{acc_pol[3]}%')

axs[1,1].plot(x,weights_up_avg[4,:],'b', x, weights_down_avg[4,:],'r')
axs[1,1].set_title(f'D = 0 , Prior = uniform (68% of retina)\n Accuracy:{acc_pol[4]}%')

axs[1,2].plot(x,weights_up_avg[5,:],'b', x, weights_down_avg[5,:],'r')
axs[1,2].set_title(f'D = 0 , Prior = uniform (93% of retina)\n Accuracy:{acc_pol[5]}%')

fig.legend([l_up,l_down],['Up population','Down population'],loc=(0.83,0.94),borderaxespad=0.3)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Position of neuron [arcmin.]" , fontsize = 14)
plt.ylabel("Weight", fontsize = 14)

plt.show()

# Plotting (3.a) - D=0

x_positions = np.array([0 ,np.rint(x_size/8), np.rint(x_size/4) ,np.rint(3*x_size/8), np.rint(x_size/2) ,np.rint(5*x_size/8) ,np.rint(6*x_size/8) ,np.rint(7*x_size/8), int(x_size)])
x_labels = [12 ,9, 6 ,3 ,0 ,-3 ,-6 ,-9 ,-12]
t_positions = [0, np.rint(t_size/5), np.rint(2*t_size/5), np.rint(3*t_size/5), np.rint(4*t_size/5), int(t_size)]
t_labels = [0, 20,40, 60, 80, 100]

fig , axs = plt.subplots(1,4)
fig.suptitle("Weights as a function of position and time - D=0", fontsize=18)

axs[0].imshow(weights_up_mat[0,:,:], cmap = 'seismic', norm=Normalize())
axs[0].set_title('Prior = delta, Up population')
axs[0].set_ylabel("Position of neuron [arcmin.]" , fontsize = 14)

axs[1].imshow(weights_down_mat[0,:,:], cmap = 'seismic', norm=Normalize())
axs[1].set_title('Prior = delta, Down population')

axs[2].imshow(weights_up_mat[1,:,:], cmap = 'seismic', norm=Normalize())
axs[2].set_title('Prior = uniform (93%), Up population')

cb = axs[3].imshow(weights_down_mat[1,:,:], cmap = 'seismic', norm=Normalize())
axs[3].set_title('Prior = uniform (93%), Down population')

plt.figtext(0.45,0.27,"Time [ms]" , fontsize = 14)
plt.setp(axs, xticks=t_positions, xticklabels=t_labels, yticks=x_positions, yticklabels=x_labels)
fig.colorbar(cb, ax=axs, orientation= 'horizontal',aspect=60)

plt.show()


# Plotting (3.b) - Prior = delta

x_positions = np.array([0 ,np.rint(x_size/8), np.rint(x_size/4) ,np.rint(3*x_size/8), np.rint(x_size/2) ,np.rint(5*x_size/8) ,np.rint(6*x_size/8) ,np.rint(7*x_size/8), int(x_size)])
x_labels = [12 ,9, 6 ,3 ,0 ,-3 ,-6 ,-9 ,-12]
t_positions = [0, np.rint(t_size/5), np.rint(2*t_size/5), np.rint(3*t_size/5), np.rint(4*t_size/5), int(t_size)]
t_labels = [0, 20,40, 60, 80, 100]

fig , axs = plt.subplots(1,4)
fig.suptitle("Weights as a function of position and time - Prior=Delta", fontsize=18)

axs[0].imshow(weights_up_mat[0,:,:], cmap = 'seismic', norm=Normalize())
axs[0].set_title('D=0, Up population')
axs[0].set_ylabel("Position of neuron [arcmin.]" , fontsize = 14)

axs[1].imshow(weights_down_mat[0,:,:], cmap = 'seismic', norm=Normalize())
axs[1].set_title('D=0, Down population')

axs[2].imshow(weights_up_mat[2,:,:], cmap = 'seismic', norm=Normalize())
axs[2].set_title('D=100, Up population')

cb = axs[3].imshow(weights_down_mat[2,:,:], cmap = 'seismic', norm=Normalize())
axs[3].set_title('D=100, Down population')


plt.figtext(0.45,0.27,"Time [ms]" , fontsize = 14)
plt.setp(axs, xticks=t_positions, xticklabels=t_labels, yticks=x_positions, yticklabels=x_labels)
fig.colorbar(cb, ax=axs, orientation= 'horizontal',aspect=60)

plt.show()

# Plotting (3.c) - Prior = delta

x_positions = np.array([0 ,np.rint(x_size/8), np.rint(x_size/4) ,np.rint(3*x_size/8), np.rint(x_size/2) ,np.rint(5*x_size/8) ,np.rint(6*x_size/8) ,np.rint(7*x_size/8), int(x_size)])
x_labels = [12 ,9, 6 ,3 ,0 ,-3 ,-6 ,-9 ,-12]
t_positions = [0, np.rint(t_size/5), np.rint(2*t_size/5), np.rint(3*t_size/5), np.rint(4*t_size/5), int(t_size)]
t_labels = [0, 20,40, 60, 80, 100]

fig , axs = plt.subplots(2,3)
fig.suptitle("Weights as a function of position and time - Prior=Delta", fontsize=18)
Accuracy_test_
cb = axs[0,0].imshow(weights_up_mat[3,:,:], cmap = 'seismic', norm=Normalize())
axs[0,0].set_title('D=100, Prior = uniform (40%)')
axs[0,0].set_ylabel("Up" , fontsize = 14)

axs[1,0].imshow(weights_down_mat[3,:,:], cmap = 'seismic', norm=Normalize())
axs[1,0].set_title('D=100, Prior = uniform (40%)')
axs[1,0].set_ylabel("Down" , fontsize = 14)

axs[0,1].imshow(weights_up_mat[4,:,:], cmap = 'seismic', norm=Normalize())
axs[0,1].set_title('D=100, Prior = uniform (68%)')
axs[0,1].set_ylabel("Up" , fontsize = 14)

axs[1,1].imshow(weights_down_mat[4,:,:], cmap = 'seismic', norm=Normalize())
axs[1,1].set_title('D=100, Prior = uniform (68%)')
axs[1,1].set_ylabel("Down" , fontsize = 14)

axs[0,2].imshow(weights_up_mat[5,:,:], cmap = 'seismic', norm=Normalize())
axs[0,2].set_title('D=100, Prior = uniform (93%)')
axs[0,2].set_ylabel("Up" , fontsize = 14)

axs[1,2].imshow(weights_down_mat[5,:,:], cmap = 'seismic', norm=Normalize())
axs[1,2].set_title('D=100, Prior = uniform (93%)')
axs[1,2].set_ylabel("Down" , fontsize = 14)

plt.subplots_adjust(bottom=0.06, top=0.92)

plt.figtext(0.13,0.6,"Position of neuron [arcmin.]" , fontsize = 14, rotation='vertical')
plt.figtext(0.45,0.01,"Time [ms]" , fontsize = 14)
plt.setp(axs, xticks=t_positions, xticklabels=t_labels, yticks=x_positions, yticklabels=x_labels)

fig.colorbar(cb, ax=axs, orientation= 'vertical',aspect=60)

plt.show()

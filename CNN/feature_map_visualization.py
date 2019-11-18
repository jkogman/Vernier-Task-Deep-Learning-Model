import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_gen_feature_map import data_generator_feature_map

# Generating data for feature map and plotting

sample , _ = data_generator_feature_map(x_max=10,prior='delta',n_sigs=0,D=100)
up_mat = sample[0,0,:,:]
down_mat = sample[0,1,:,:]

x_positions = np.array([0 ,np.rint(x_size/8), np.rint(x_size/4) ,np.rint(3*x_size/8), np.rint(x_size/2) ,np.rint(5*x_size/8) ,np.rint(6*x_size/8) ,np.rint(7*x_size/8), int(x_size)])
x_labels = [12 ,9, 6 ,3 ,0 ,-3 ,-6 ,-9 ,-12]
t_positions = [0, np.rint(t_size/5), np.rint(2*t_size/5), np.rint(3*t_size/5), np.rint(4*t_size/5), int(t_size)]
t_labels = [0, 20,40, 60, 80, 100]

fig, ax = plt.subplots(1,2)
fig.suptitle("Data sample for feature map visualization",fontsize=18)
ax[0].imshow(up_mat)
ax[0].set_title('Up population spikes',fontsize=14)
ax[0].set_ylabel("Position of neuron [arcmin.]" , fontsize = 14) #Change

mp = ax[1].imshow(up_mat)
ax[1].set_title('Down population spikes',fontsize=14)

plt.figtext(0.45,0.27,"Time [ms]" , fontsize = 14) # Change
plt.setp(ax, xticks=t_positions, xticklabels=t_labels, yticks=x_positions, yticklabels=x_labels)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mp, cax=cax)
plt.savefig("/sample.png")
plt.show()

# Creating feature maps and plotting

model = load_model("final_model.h5")

for i in range (3):
    
  end_layer = nums[i]
  part_model = Model(inputs=model.inputs, outputs=model.layers[end_layer].output)
  feature_map = part_model.predict(sample)
  feature_map = np.squeeze(feature_map)
  n = feature_map.shape[0]

  fig , axs = plt.subplots(3,2)
  fig.suptitle(f'Feature map channels after convolutional layer {i+1}', fontsize = 30)

  for j in range(3):
    for k in range(2):
      
      if j == 0:
        mini, maxi = np.amin(feature_map[k,:,:]) , np.amax(feature_map[k,:,:])
        #print(mini)
        feature_map[k,:,:] = (feature_map[k,:,:]-mini) / (maxi-mini)
        im = axs[j,k].imshow(feature_map[k,:,:])
        axs[j,k].set_title(f'Channel {k+1}/{n}', fontsize = 20)
      elif j == 1:
        axs[j,k].imshow(feature_map[int(n/2+k),:,:])
        axs[j,k].set_title(f'Channel {int(n/2+k+1)}/{n}', fontsize = 20)
      elif j==2:
        axs[j,k].imshow(feature_map[int(n-3+k),:,:])
        axs[j,k].set_title(f'Channel {int(n-3+k+1)}/{n}', fontsize = 20)
        
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.1, aspect = 40)
  plt.savefig(f"/content/gdrive/My Drive/Vernier ANN/CNN's/Final models/Feature maps/layer{i}.png")
  plt.rcParams["figure.figsize"] = (17,17)
  plt.subplots_adjust(wspace=0.9)
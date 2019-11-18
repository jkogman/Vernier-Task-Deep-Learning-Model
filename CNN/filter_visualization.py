import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Checking the size of filters

model = load_model("final_model.h5")
for layer in model.layers:

  if 'conv' not in layer.name:
    continue

filters, biases = layer.get_weights()
print(layer.name, filters.shape)

nums = [46, 52, 60]
x_size = 200
t_size = 125
filters, _ = model.layers[0].get_weights()

# Creating filters array

total = np.empty((50,30,2,3))
for i in range(3):
  total[:,:,:,i] = filters[:,:,:,nums[i]]

# Normalization of filters

for j in range(3):
  mini, maxi = np.amin(total[:,:,:,j]) , np.amax(total[:,:,:,j])
  total[:,:,:,j] = (total[:,:,:,j]-mini) / (maxi-mini)

fig,axs = plt.subplots(2,3)
fig.suptitle("Filters of first convolutional layer for both populations",fontsize=28)

for i in range(3):
  im = axs[0,i].imshow(total[:,:,0,i])
  axs[0,i].set_title(f"Filter {nums[i]}/64, Up",fontsize=20)
  axs[1,i].imshow(total[:,:,1,i])
  axs[1,i].set_title(f"Filter {nums[i]}/64, Down",fontsize=20)

plt.rcParams["figure.figsize"] = (17,17)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
fig.colorbar(im, ax=axs.ravel().tolist(), aspect = 40, shrink=0.95)
plt.savefig("filters.png")
plt.show()


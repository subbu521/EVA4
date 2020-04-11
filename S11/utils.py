import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import cv2

def imshow(img):
	img = denormalize(img)
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))

def normalize(tensor, mean=[0.4914, 0.4822, 0.4465],
						std=[0.2023, 0.1994, 0.2010]):
	single_img = False
	if tensor.ndimension() == 3:
		single_img = True
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
	    raise TypeError('tensor should be 4D')

	mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.sub(mean).div(std)
	return ret[0] if single_img else ret

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465],
						std=[0.2023, 0.1994, 0.2010]):
	single_img = False
	if tensor.ndimension() == 3:
		single_img = True
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
	    raise TypeError('tensor should be 4D')

	mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.mul(std).add(mean)
	return ret[0] if single_img else ret

def plot_images(img_data,classes,img_name):
  figure = plt.figure(figsize=(10, 10))
  
  num_of_images = len(img_data)
  print(num_of_images)
  for index in range(1, num_of_images + 1):
      img = denormalize(img_data[index-1][0])  # unnormalize
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
      plt.title("Actual: %s\nPredicted: %s" % (classes[img_data[index-1][1]], classes[img_data[index-1][2]]))
  
  plt.tight_layout()
  plt.savefig(img_name)


def print_triangular_lr(iteration, stepsize, base_lr, max_lr):
    cycle = np.floor(1 + iteration/(2  * stepsize))
    x = np.abs(iteration/stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
    return lr

  
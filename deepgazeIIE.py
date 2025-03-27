import numpy as np
# from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import cv2
import matplotlib.pyplot as plt

import deepgaze_pytorch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeI(pretrained=True).to(DEVICE)

image_path = r"I:\Saliency4asd\Saliency4asd\ASD_FixMaps\1_s.png"  # Replace 'sample.jpg' with actual image
image = cv2.imread(image_path)  # Read image using OpenCV
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
# you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
# alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
centerbias_template = np.zeros((1024, 1024))
# rescale to match image size
centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
# renormalize log density
centerbias -= logsumexp(centerbias)

image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

log_density_prediction = model(image_tensor, centerbias_tensor)

print(log_density_prediction.shape)
saliency_map = torch.exp(log_density_prediction).detach().cpu().numpy()[0]


# Show the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")

# Show the predicted saliency map
plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap="hot")
plt.axis("off")
plt.title("Predicted Saliency Map")

plt.imshow(saliency_map, cmap='hot')  # Use 'hot' colormap for visibility
plt.colorbar()
plt.title("Predicted Saliency Map")
plt.show()

plt.show()
import os
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import cv2
import matplotlib.pyplot as plt
import deepgaze_pytorch

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model
model = deepgaze_pytorch.DeepGazeI(pretrained=True).to(DEVICE)

# Define input and output directories
input_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMaps"  # Folder containing images
output_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMapsOutput"  # Folder to save saliency maps

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load centerbias (you can also use uniform `np.zeros()`)
centerbias_template = np.zeros((1024, 1024))

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        image_path = os.path.join(input_folder, filename)
        print(f"Processing {image_path}...")

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize centerbias to match image size
        centerbias = zoom(centerbias_template,
                          (image.shape[0] / centerbias_template.shape[0],
                           image.shape[1] / centerbias_template.shape[1]),
                          order=0, mode='nearest')
        centerbias -= logsumexp(centerbias)  # Normalize log density

        # Convert to tensors
        image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
        centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

        # Generate saliency map
        log_density_prediction = model(image_tensor, centerbias_tensor)
        print(f"Log Density Prediction Shape for {filename}: {log_density_prediction.shape}")
        saliency_map = torch.exp(log_density_prediction).detach().cpu().numpy()[0]

        # Save saliency map
        output_path = os.path.join(output_folder, f"{filename}")
        plt.imsave(output_path, saliency_map, cmap='hot')

        print(f"Saved: {output_path}")

print("Processing complete!")

#!/usr/bin/env python3.6
import roslib

roslib.load_manifest('video_stream_opencv')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import psutil
import os
from ros_referee.msg import ProcessStatus
import time
import json
import io

import torch
import torchvision
from PIL import Image as pilim
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box
from pandas import DataFrame

# from jetson_benchmarks import utilities, benchmark_argparser
import sys
import glob
import numpy as np
import signal
import csv

from proc_util import get_proc_status, get_proc_children, benchmark_pre, benchmark_post
# from PIL import Image
from matplotlib import cm
# from ssim import compute_ssim

#from skimage.metrics import structural_similarity as ssim
from torch.profiler import profile, record_function, ProfilerActivity
import cProfile

# import nvidia_dlprof_pytorch_nvtx as nvtx
# nvtx.init(enable_function_stack=True)

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model

# Get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

# Load the pre-trained Faster R-CNN model
model = create_model(num_classes=91)
model.to(device)
model.eval()  # Set the model to evaluation mode
train_weights = "fasterrcnn_resnet50_fpn_coco.pth"
assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
# self.model.load_state_dict(torch.load(train_weights, map_location=self.device)["model"])
model.load_state_dict(torch.load(train_weights, map_location=device), strict=False)

# Prepare the data transformation
data_transform = transforms.Compose([transforms.ToTensor()])

# Paths
sequence_path = '/home/mobilitylab/dataset/sequences/00/image_2/'  # Path to directory containing images
output_path = 'adversarial_images/'  # Output directory for perturbed images
os.makedirs(output_path, exist_ok=True)

# Load and preprocess images into a list
image_paths = sorted([os.path.join(sequence_path, fname) for fname in os.listdir(sequence_path) if fname.endswith('.png')])[:1]
original_images = []
perturbations = []

# Since images may have different sizes, we need to handle perturbations accordingly
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    pil_image = pilim.fromarray(image)
    tensor_image = data_transform(pil_image).to(device)
    original_images.append(tensor_image)

    # # Create a perturbation tensor with the same shape as the image
    # perturbation = torch.zeros_like(tensor_image, requires_grad=True)
    # perturbations.append(perturbation)

# Stack images into a batch tensor
batch_images = torch.stack(original_images)  # Shape: (batch_size, 3, H, W)

# Create a single perturbation tensor with requires_grad=True
perturbation = torch.zeros_like(original_images[0], requires_grad=True)

# # Keep a copy of original images
# original_images = [img.clone().detach() for img in batch_images]
# Optimization parameters
epsilon = 0.03  # Maximum perturbation
alpha = 0.01    # Step size
num_steps = 500

# Variables to store results
steps = []
losses = []
proposals_counts = []

# Optimizer for the perturbation
optimizer = torch.optim.SGD([perturbation], lr=alpha)


# Optimizing the batch of images
for step in range(num_steps):
    optimizer.zero_grad()

    # Create perturbed images by adding the same perturbation to all images
    perturbed_images = batch_images + perturbation.unsqueeze(0)  # Broadcast perturbation to all images
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Convert list of images to list for model input
    perturbed_images_list = [img for img in perturbed_images]

    # Forward pass
    outputs = model(perturbed_images.to(device))

    print("check1")


    # Extract objectness logits and pre-NMS proposals
    objectness_logits = [output['objectness_logits'] for output in outputs]
    pre_nms_proposals = [output['pre_nms_proposals'] for output in outputs]

    # Compute the loss as the negative sum of objectness logits
    loss = 0
    for obj_logit in objectness_logits:
        loss -= torch.sum(obj_logit)
    # loss /= len(objectness_logits)  # Optional normalization

    # Backward pass
    loss.backward()

    # # Compute average gradient across the batch
    # avg_grad = torch.mean(torch.stack([img.grad for img in batch_images]), dim=0)

     # Gradient ascent step on the perturbation
    with torch.no_grad():
        # Update the perturbation
        perturbation += alpha * perturbation.grad.sign()

        # Clamp the perturbation to be within [-epsilon, epsilon]
        perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)

        # Zero the gradients for the next step
        perturbation.grad.zero_()

    # Record the loss and number of proposals
    steps.append(step + 1)
    losses.append(-loss.item())
    proposals_counts.append(total_num_proposals.item())

    print(f"Step [{step + 1}/{num_steps}], Loss: {-loss.item()}, Total Proposals: {total_num_proposals.item()}")

 # Save the optimized images
for i in range(len(original_images)):
    perturbed_image = torch.clamp(original_images[i] + perturbation, 0, 1)
    img = perturbed_image.cpu().detach().numpy().transpose(1, 2, 0) * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, f"optimized_{i:04d}.png"), img)


# Plotting the loss and total number of proposals over steps
plt.figure(figsize=(14, 6))

# Plot the loss
plt.subplot(1, 2, 1)
plt.plot(steps, losses, label='Loss', color='blue')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over Steps')
plt.grid(True)
plt.legend()

# Plot the total number of proposals
plt.subplot(1, 2, 2)
plt.plot(steps, proposals_counts, label='Total Proposals', color='green')
plt.xlabel('Step')
plt.ylabel('Total Number of Proposals')
plt.title('Total Proposals over Steps')
plt.grid(True)
plt.legend()

plt.tight_layout()
plot_file_path = 'optimization_results_plot.png'
plt.savefig(plot_file_path)
print(f'Optimization results plot saved to {plot_file_path}')

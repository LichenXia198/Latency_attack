#!/usr/bin/env python3.6
import roslib
import rospy
import cv2
from sensor_msgs.msg import Image
import os
from ros_referee.msg import ProcessStatus
import time
from torchvision import transforms
import torchvision
# import deeplab
import torch
from PIL import Image as pilim
import sys
import glob
import numpy as np
from proc_util import get_proc_status, get_proc_children, benchmark_pre, benchmark_post
from src import UNet
import json

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class unet_node:
    
    def __init__(self):

        classes = 1  # exclude background
        weights_path = "/home/mobilitylab/projects/PDNN/catkin_ws/src/ros_segmentation/scripts/unet/save_weights/best_model.pth"
        img_path = "/home/mobilitylab/projects/PDNN/catkin_ws/src/ros_segmentation/scripts/unet/DRIVE/test/images/01_test.tif"
        roi_mask_path = "/home/mobilitylab/projects/PDNN/catkin_ws/src/ros_segmentation/scripts/unet/DRIVE/test/mask/01_test_mask.gif"
        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(img_path), f"image {img_path} not found."
        assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        # get devices
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(self.device))

        # create model
        self.model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

        # load weights
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        self.model.to(self.device)

        # load roi mask
        roi_img = pilim.open(roi_mask_path).convert('L')
        roi_img = np.array(roi_img)

        # load image
        original_img = pilim.open(img_path).convert('RGB')

        # from pil image to tensor and normalize
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
        img = self.data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        self.model.eval()  # 进入验证模式

        with torch.no_grad():
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=self.device)
            self.model(init_img)

        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.status_pub = rospy.Publisher('unet_status', ProcessStatus, queue_size=1)
        print("ready")

    def callback(self, frame):
        sf = rospy.get_param("semantic_skip_frames")
        if sf == 0:
            with torch.no_grad():
                t_1 = time_synchronized()
                # print(frame.width,frame.height,type(frame.width))
                test_img = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, -1)
                # cv2.imwrite("test.png",test_img)
                test_img = pilim.fromarray(test_img)

                tran_img = self.data_transform(test_img)
                tran_img = torch.unsqueeze(tran_img, dim=0)
                t_2 = time_synchronized()

                output = self.model(tran_img.to(self.device))
                t_3 = time_synchronized()

                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                # 将前景对应的像素值改成255(白色)
                prediction[prediction == 1] = 255
                # 将不敢兴趣的区域像素设置成0(黑色)
                # prediction[roi_img == 0] = 0
                # mask = pilim.fromarray(prediction)
                # mask.save("test_result.png")
                
                t_4 = time_synchronized()
                print(t_2-t_1,t_3-t_2,t_4-t_3, t_4-t_1)
                
            pid = os.getpid()
            msg = get_proc_status(pid)
            # msg.header.stamp = rospy.Time.now()
            msg.header.stamp = frame.header.stamp
            # msg.proposals = midres[0]
            # msg.objects = midres[1]
            # msg.probability = predict_scores
            msg.runtime = t_4-t_1
                
            # rospy.loginfo(msg)
            self.status_pub.publish(msg)
        elif sf < 0:
            rospy.set_param("semantic_skip_frames", 0)
        else:
            rospy.set_param("semantic_skip_frames", sf-1)

if __name__ == '__main__':
    rospy.init_node('unet_node')
    ic = unet_node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
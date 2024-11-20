import os
import time
import json

# import sys
# py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
# os.environ['PATH'] += py_dll_path

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box


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


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=21)

    # load train weights
    train_weights = "/home/mobilitylab/projects/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/save_weights/fasterrcnn_voc2012.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = '/home/mobilitylab/projects/PDNN/catkin_ws/src/ros_object_detection/scripts/faster_rcnn/pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    original_img = Image.open("/home/mobilitylab/projects/PDNN/catkin_ws/src/ros_object_detection/scripts/faster_rcnn/test.png")

    # from pil image to tensor, do not normalize image
    # data_transform = transforms.Compose([transforms.ToTensor()])
    data_transform = transforms.Compose([transforms.Resize(size = (5,3)),transforms.ToTensor()])
    imgd = data_transform(original_img)
    # expand batch dimension
    imgd = torch.unsqueeze(imgd, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = imgd.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        for i in range(100):
            t_1 = time_synchronized()
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            
            t_2 = time_synchronized()
            img_cuda = img.to(device)
            t_2_m = time_synchronized()
            predictions = model(img_cuda)[0]
            # predictions = model(img.to(device))[0]
            t_3 = time_synchronized()

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            pt = predictions["time"]
            midres = predictions["midres"]

            print(t_2-t_1,t_2_m-t_2, t_3-t_2_m)

            # print(t_2-t_1, t_3-t_2, pt[0],pt[1],pt[2],pt[3],pt[4],pt[5], t_3-t_1,midres[0],midres[1])

        # if len(predict_boxes) == 0:
        #     print("没有检测到任何目标!")

        # draw_box(original_img,
        #          predict_boxes,
        #          predict_classes,
        #          predict_scores,
        #          category_index,
        #          thresh=0.05,
        #          line_thickness=3)
        # plt.imshow(original_img)
        # plt.show()
        # 保存预测的图片结果
        # original_img.save("test_result.jpg")


if __name__ == '__main__':
    main()


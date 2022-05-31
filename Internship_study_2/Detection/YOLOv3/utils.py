from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

"""predict_transform 함수는 detection feature map을 취하고, 이것을 2-D tensor로 변경합니다.
"""
def predict_transform_ttt(prediction, inp_dim, anchors, num_classes, CUDA = True):
    inp_dim = 416   # yolov3.cfg file의 [net]의 width, height를 입력크기(416, 416)에 맞춰 바꿔주어야 한다.
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors] # a[0]: [net] height, a[1]: [net] width

    # prediction[:, :, 0:5] = x, y, W, H, c(obj conf)
    # 중심 x,y 좌표와 object confidence를 SIgmoid 합니다.
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # 높이와 넓이를 log space 변환합니다.
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
        prediction = prediction.cuda()
        
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4] * anchors)

    # class score에 sigmoid activation을 적용합니다.
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:,5 : 5 + num_classes]))

    # detection map을 입력 이미지의 크기로 resize 합니다.
    # if input.shape = (416, 416) and detection_map.shape = (13, 13), then stride = 32
    prediction[:,:,:4] *= stride

    return prediction
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

""" predict_transform 함수는 detection feature map을 취하고, 이것을 2-D tensor로 변경합니다. 2-D tensor는 아래 그림의 순서로 바운딩 박스들의 속성에 해당하는 tensor의 각 행으로 이루어져 있습니다."""

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

"""Object confidence thresholding & NMS"""

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

# iou 계산하기
def bbox_iou(box1, box2):
    # bounding boxes의 좌표를 얻습니다.
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    # intersection rectangle 좌표를 얻습니다.
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection 영역
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    # Union 영역
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # threshold 보다 낮은 objectness score를 갖은 bounding box의 속성을 0으로 설정
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # Non-maximum suppression
    # bounding box의 중심 점을 좌측 상단, 우측 하단 모서리 좌표로 변환하기
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False

    # 한번에 하나의 이미지에 대하여 수행
    for ind in range(batch_size):
        image_pred = prediction[ind]    # Image Tensor
        # confidence threshholding 
        # NMS

        # 가장 높은 값을 가진 class score를 제외하고 모두 삭제
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # threshold보다 낮은 object confidence를 지닌 bounding box rows를 0으로 설정한 것을 제거
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:    # no detections를 얻었을 때 상황을 다루기 위한 것
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        # PyToch 0.4 호환성
        # scalar가 PyTorch 0.4에서 지원되기 때문에 no detection에 대한 
        # not raise exception 코드입니다.
        if image_pred_.shape[0] == 0:
            continue 

        # 이미지에서 검출된 다양한 classes를 얻기
        # 이때 동일한 class에 다수의 detections가 존재할 수 있기 때문에 unique함수 사용.
        img_classes = unique(image_pred_[:,-1]) # -1 index는 class index를 지니고 있습니다.

        for cls in img_classes:
            # NMS 실행하기
            # 특정 클래스에 대한 detections 얻기
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            # 가장 높은 objectness를 지닌 detections 순으로 정렬하기
            # confidence는 가장 위에 있습니다.
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #detections의 수

            for i in range(idx):
                # 모든 박스에 대해 하나하나 IOU 얻기
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:  # 경계가 벗어난 값(IndexError)을 인덱싱하거나
                    break           # NMS가 추가적인 bounding boxes를 제거할 수 없음을 확인할 수 있고 loop를 중지합니다.
                except IndexError:  # image_pred_class[i+1:] 슬라이싱은 빈 tensor를 반환하여 ValueError를 트리거
                    break           # NMS가 추가적인 bounding boxes를 제거할 수 없음을 확인할 수 있고 loop를 중지합니다.
                
                # IoU > threshhold인 detections를 0으로 만들기
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       

                # non-zero 항목 제거하기
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

                # i 인덱스를 갖고 있는 box의 IoU와 i보다 큰 인덱스를 지닌 bounding boxes 얻기
                ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])

                # IoU > threshhold인 detections를 0으로 만들기
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       

                # non-zero 항목 제거하기
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      
            # 이미지에 있는 class의 detections 만큼 batch_id를 반복합니다.
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
        
    try:
        return output
    except:
        return 0
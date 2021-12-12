#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
import random
import os
import time
import tensorflow as tf
from tensorflow import keras
from loguru import logger
import matplotlib.pyplot as plt
__all__ = ["vis"]
dct_name = {
    0:"FD",
    1:"GLC",
    2:"JL",
    3:"BHC",
    4:"WTKL",
    5:"SHNT",
    6:"XB"
}

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None,recog_model=None):
    #为防止框框和标签信息覆盖原有图像，生成副本。
    img_copy=img.copy()
    recog_time=0
    
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        
        #数据采集时：如果cls_id==39，意味着是bottle这一类
        # print(cls_id)
        # if cls_id==39:
        #     cropped = img_copy[y0:y1, x0:x1] # 裁剪坐标为[y0:y1, x0:x1]
        #     if cropped.shape[0]==0 or cropped.shape[1]==0: continue
        #     cv2.imwrite(os.getcwd()+r'/reallife_images/0/'+str(random.random())+'.jpg', cropped)
        # 对于cropped实时预测
        
        #frame_info先设为None，修改该函数传回realtime调用的地方，
        #它的作用是存储当前帧所识别到的饮料瓶信息和位置，并回传给主函数来判断饮料是否拿出
        #目前我的朴素算法支支持单次拿出一个多多个不同种类的饮料，还不支持拿出不同种类的饮料
        frame_info={i:{'loc':0,'cnt':0} for i in range(7)} 

        if cls_id==39:
            
            cropped = img_copy[y0:y1, x0:x1] 
            if cropped.shape[0]!=0 and cropped.shape[1]!=0:
                if cropped.shape[0]/cropped.shape[1]>1.5 or cropped.shape[1]/cropped.shape[0]>1.5:#避免检测瓶盖 
                    # cv2.imwrite(os.getcwd()+r'/problem/'+str(random.random())+'.jpg', np.array(cropped))
                    cropped=tf.image.resize(cropped,(96,96))
                    cropped = cropped[:,:,::-1]#bgr转rgb--这个问题困扰了我很久，且要/255才行
                    cropped/=255
                    # plt.imshow(cropped)
                    # plt.show() #调试
                    print(cropped.shape)
                    # cv2.imwrite(os.getcwd()+r'/problem/'+str(random.random())+'.jpg', np.array(cropped))
                    t0=time.time()
                    pred=recog_model.predict(np.array(cropped)[tf.newaxis,...])
                    recog_time+=time.time()-t0
                    pred_cls_id=np.argmax(pred[0])
                    pred_cla_name=dct_name[pred_cls_id]
                    text+=f'({pred_cla_name})'#预测出的类别的拼音缩写（因为opencv不支持中文）

                    #存储进frame_info
                    frame_info[pred_cls_id]['cnt']+=1
                    frame_info[pred_cls_id]['loc']=(x0+x1)/2/img_copy.shape[1] #记录x轴的相对位置，在函数外判断是否在向右移动
                    # print(frame_info)

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    
    
    logger.info("Recognition time: {:.4f}s".format(recog_time))
    
    return img,frame_info


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import os
import sys
import cv2
import onnxruntime
import torch
import torchvision
import argparse
import json
import copy
from tqdm import tqdm
from datetime import datetime
from common import *
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def preprocess_image(image, new_shape=(640, 640), RGB=True, normal=True):
    shape = image.shape[:2]  # current shape [height, width]

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw, dh = dw/2, dh/2  # divide padding into 2 sides

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

    if RGB:
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    else:
        image = image.transpose((2, 0, 1))
    if normal:
        image = image / 255.0
    image = np.ascontiguousarray(image)  # contiguous
    image = image.astype(np.float32)

    return image, ratio, (dw, dh)
    
def run_infer(model, image):
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    preds = model.run([], {model.get_inputs()[0].name: image})[0]
    
    return preds

def post_process(preds, conf_threshold=0.5, iou_threshold=0.5, with_confidence=False, need_trans=True, xywh=True, muti_label=True, agnostic=True):
    if need_trans:
        preds = np.transpose(preds, (0, 2, 1))
    preds = torch.from_numpy(preds)[0]
    if with_confidence:
        boxes, confs, scores = preds[..., :4], preds[..., 4:5], preds[..., 5:]
        pc = confs.amax(1) > conf_threshold 
        scores *= confs    
    else:
        boxes, scores = preds[..., :4], preds[..., 4:]
        pc = scores.amax(1) > conf_threshold            
    boxes, scores = boxes[pc], scores[pc]
    # boxes, scores = preds[:, :4], preds[:, 4:]
    if xywh:
        boxes = xywh2xyxy(boxes)
    if muti_label:
        i, j = (scores > conf_threshold).nonzero(as_tuple=False).T
        x = torch.cat((boxes[i], scores[i, j].unsqueeze(1), j.float().unsqueeze(1)), 1)
    else:
        conf, j = scores.max(1, keepdim=True)       # 所有类选最大score
        x = torch.cat((boxes, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
    x = x[x[:, 4].argsort(descending=True)]
    c = 0 if agnostic else x[:, 5:6] * 7680            # yolov5中的nms方法，会提高一点精度
    i = torchvision.ops.nms(x[:, :4] + c, x[:, 4], iou_threshold)
    results = x[i]
    return results

def scale_result(results, scale_factor, padw, padh):
    results[:, 0] -= int(padw)
    results[:, 1] -= int(padh)
    results[:, 2] -= int(padw)
    results[:, 3] -= int(padh)
    results[:, :4] /= scale_factor
    return results.numpy()

def draw_result(image, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f'{int(cls_id)}: {conf:.2f}'
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        image = cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)
    return image

def write_txt(results, width, height, path=""):
    if len(results) == 0:
        return
    if os.path.exists(path):
        os.remove(path)
    for i, box in enumerate(results):
        x1, y1, x2, y2, cls_id = box[0]/width, box[1]/height, box[2]/width, box[3]/height, box[5]
        x = round((x1 + x2) / 2, 4)
        y = round((y1 + y2) / 2, 4)
        w = round(x2 - x1, 4)
        h = round(y2 - y1, 4)
        with open(path, "a") as f:
            f.write("{} {} {} {} {}\n".format(int(cls_id), x, y, w, h))

def detect(model, image, conf_thresh=0.35, iou_thresh=0.5, version="v8"):
    
    # 图像预处理，准备模型输入
    input_size = model.get_inputs()[0].shape[2:]
    # image_normal, scale, (padw, padh) = preprocess_image(image, img_size=input_size,
    #             normal=False if version == 'x' else True)
    image_normal, scale, (padw, padh) = preprocess_image(image, new_shape=input_size, 
                normal=False if version == 'x' else True)
    # 使用模型进行推理，获取预测结果
    preds = run_infer(model, image_normal)
    # 对预测结果进行后处理，筛选符合阈值的检测结果
    boxes = post_process(preds, conf_threshold=conf_thresh, 
                iou_threshold = iou_thresh,
                need_trans=True if version == "v8" else False, 
                with_confidence=True if version in ["v5", "x"] else False,
                xywh=False if version == "pp" else True)
    # 将结果按图像原始比例反向变换
    boxes = scale_result(boxes, scale, padw, padh)

    return {'boxes': boxes, 'width': image.shape[1], 'height': image.shape[0]}

def run(args):

    project_name = os.path.join(args.project, os.path.basename(args.model).split('.')[0], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(project_name)

    model = onnxruntime.InferenceSession(args.model, providers=['CUDAExecutionProvider'])

    versions = ["v5", "v8", "x", "pp"]  
    if args.version not in versions:
        raise ValueError(f'Invalid version {args.version}, must be one of {versions}')

    results_list, source_list = [], []
    if os.path.isdir(args.source):
        source_list = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    else:
        source_list = [args.source]
    for source in tqdm(source_list):
        image = cv2.imread(source)
        if image is None:
            print("Failed to read image:", source)
            continue
        results = detect(model, image, conf_thresh=args.conf, iou_thresh=args.iou, version=args.version)
        name = os.path.basename(source)
        if args.draw:
            draw_dir = os.path.join(project_name, "draw")
            if not os.path.exists(draw_dir):
                os.mkdir(draw_dir)
            image = draw_result(image, results['boxes'])
            cv2.imwrite(os.path.join(draw_dir, name), image)

        results_list.append({'source': name, 'results': results})


    json_dict = []
    is_coco = True
    coco80to91 = coco80_to_coco91_class()
    
    for results in results_list:
        name = results['source'].split('.')[0]
        if args.save_txt:
            txt_dir = os.path.join(project_name, "txt")
            if not os.path.exists(txt_dir):
                os.mkdir(txt_dir)
            write_txt(results['results']['boxes'], results['results']['width'], results['results']['height'], 
                os.path.join(txt_dir, name + '.txt'))
        if args.save_json:
            boxes = results['results']['boxes'].tolist()
            for box in boxes:
                b = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                json_dict.append(
                {
                    'image_id': int(name) if name.isnumeric() else name,
                    'category_id': coco80to91[int(box[5])] if is_coco else int(box[5]), 
                    'bbox': [round(b0, 3) for b0 in b],
                    'score': round(box[4], 4),
                })
    if args.save_json:
        if len(json_dict) == 0:
            print('WARNING: No detections were made. Check that the weights and parameters are correct and that the images are correctly labelled.')
        with open(os.path.join(project_name, 'results.json'), 'w') as f:
            json.dump(json_dict, f)
    return results_list

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="YOLO Object Detection with ONNX Runtime")
    argparser.add_argument('--model', type=str, default='yolov8n.onnx', help='model.onnx path')
    argparser.add_argument('--source', type=str, default='bus.jpg', help='image path')
    argparser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold')
    argparser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    argparser.add_argument('--version', type=str, default='v8', help='yolo version')
    argparser.add_argument('--save_txt', action='store_true', help='save results as txt')
    argparser.add_argument('--save_json', action='store_true', help='save results as json')
    argparser.add_argument('--draw', default=True, action='store_true', help='draw results')
    argparser.add_argument('--project', type=str, default='runs', help='save results to project/name')
    args = argparser.parse_args()
    
    run(args)
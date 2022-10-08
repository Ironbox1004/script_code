import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image

SHOW = True
SHOW_GT = True
COLOR = np.array([
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
    0.50, 0.5, 0]).astype(np.float32).reshape(-1, 3)
# NAME = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
NAME = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
DEVICE = torch.device('cuda:0')
FONT = cv2.FONT_HERSHEY_SIMPLEX


class TMTools:
    def __init__(self, image, gt, pred):
        # gt: class,x,y,x,y
        # pred: x,y,x,y,conf,class
        # num_gt, num_pred = gt.shape[0], pred.shape[0]
        self.gt = gt.to(DEVICE)
        self.pred = pred.to(DEVICE)
        self.image = image
        self.predn = pred.clone()  # x,y,x,y,conf,class
        self.right_match = None
        self.error_match = None

    def draw_gt(self):
        gt = self.gt.clone()
        image = self.image.copy()
        for cls, x1, y1, x2, y2 in gt:
            name = NAME[int(cls)]
            color = (COLOR[int(cls)] * 255).astype(np.uint8)
            x1, y1, x2, y2 = round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))
            txt_color = (0, 0, 0) if np.mean(color) > 0.5 else (255, 255, 255)
            txt_size = cv2.getTextSize(name, FONT, 0.4, 1)[0]
            txt_bk_color = (color * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), 2)
            cv2.rectangle(image, (x1, y1 + 1), (x1 + txt_size[0] + 1, y1 + int(1.5 * txt_size[1])), txt_bk_color, -1)
            cv2.putText(image, name, (x1, y1 + txt_size[1]), FONT, 0.4, txt_color, thickness=1)
        return image

    def draw_pred(self):
        pred = self.pred.clone()
        image = self.image.copy()
        for x1, y1, x2, y2, conf, cls in pred:
            name = NAME[int(cls)]
            color = (COLOR[int(cls)] * 255).astype(np.uint8)
            x1, y1, x2, y2 = round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))
            txt_color = (0, 0, 0) if np.mean(color) > 0.5 else (255, 255, 255)
            txt_size = cv2.getTextSize(name, FONT, 0.4, 1)[0]
            txt_bk_color = (color * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), 2)
            cv2.rectangle(image, (x1, y1 + 1), (x1 + txt_size[0] + 1, y1 + int(1.5 * txt_size[1])), txt_bk_color, -1)
            cv2.putText(image, name, (x1, y1 + txt_size[1]), FONT, 0.4, txt_color, thickness=1)
        return image

    def draw_no_match(self):
        gt = self.gt.clone()
        pred = self.pred.clone()
        image_no_match, image_error_match = self.image.copy(), self.image.copy()
        no_match, match_wrong = self._get_error_det(pred, gt)

        name = 'no_match'
        for cls, x0, y0, x1, y1 in no_match:
            cls = int(cls)
            color = (COLOR[cls] * 255).astype(np.uint8)
            x0, y0, x1, y1 = torch.tensor([x0, y0, x1, y1]).round().int().tolist()
            txt_color = (0, 0, 0) if np.mean(color) > 0.5 else (255, 255, 255)
            txt_size = cv2.getTextSize(name, FONT, 0.4, 1)[0]
            cv2.rectangle(image_no_match, (x0, y0), (x1, y1), color.tolist(), 2)
            txt_bk_color = (color * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(image_no_match, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                          txt_bk_color, -1)
            cv2.putText(image_no_match, name, (x0, y0 + txt_size[1]), FONT, 0.4, txt_color, thickness=1)

        name = 'error_match'
        for x0, y0, x1, y1, conf, cls in match_wrong:
            cls = int(cls)
            color = (COLOR[cls] * 255).astype(np.uint8)
            x0, y0, x1, y1 = torch.tensor([x0, y0, x1, y1]).round().int().tolist()
            txt_color = (0, 0, 0) if np.mean(color) > 0.5 else (255, 255, 255)
            txt_size = cv2.getTextSize(name, FONT, 0.4, 1)[0]
            cv2.rectangle(image_error_match, (x0, y0), (x1, y1), color.tolist(), 2)
            txt_bk_color = (color * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(image_error_match, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                          txt_bk_color, -1)
            cv2.putText(image_error_match, name, (x0, y0 + txt_size[1]), FONT, 0.4, txt_color, thickness=1)

        return image_no_match, image_error_match

    def _get_error_det(self, detections, labels):
        if not detections.shape[0]:
            no_match = labels.clone().to(labels.device)
            match_wrong = torch.empty(0, 6).to(labels.device)
            return no_match, match_wrong

        gt_idx = set(range(labels.shape[0]))
        pred_idx = set(range(detections.shape[0]))

        iou = self._box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]

        no_match, match_wrong = torch.empty(0, 5).to(labels.device), torch.empty(0, 6).to(labels.device)
        pred_more = match_wrong.clone()
        match_wrong_list = []
        correct_stat = (iou >= 0.5) & correct_class

        x = torch.where(correct_stat)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] >= 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                self.right_match = matches.copy()

                match_gt_idx = set(matches[:, 0].astype(np.uint8).tolist())
                match_pred_idx = set(matches[:, 1].astype(np.uint8).tolist())

                no_match = labels[list(gt_idx - match_gt_idx)] # gt 没被 match
                no_match_idx = set(range(no_match.shape[0]))

                pred_more = detections[list(pred_idx - match_pred_idx)] # 预测多的
                pred_more_idx = set(range(pred_more.shape[0]))

                _iou = self._box_iou(no_match[:, 1:], pred_more[:, :4]) # 剩下的 gt 和 剩下的 pred
                _stat = (_iou >= 0.5)
                _x = torch.where(_stat)
                if _x[0].shape[0]:
                    _matches = torch.cat((torch.stack(_x, 1), _iou[_x[0], _x[1]][:, None]), 1).cpu().numpy()
                    if _x[0].shape[0] >= 1:
                        _matches = _matches[_matches[:, 2].argsort()[::-1]]
                        _matches = _matches[np.unique(_matches[:, 1], return_index=True)[1]]
                        # _matches = _matches[_matches[:, 2].argsort()[::-1]]
                        _matches = _matches[np.unique(_matches[:, 0], return_index=True)[1]]

                        no_match_but_iou_idx = set(_matches[:, 0].astype(np.uint8).tolist())  # 没被 match 的 gt 被剩下 pred 预测了
                        pred_more_but_iou_idx = set(_matches[:, 1].astype(np.uint8).tolist()) # 剩下 pred 预测了没被 match 的 gt

                        # pred_more = pred_more[list(pred_more_idx-pred_more_but_iou_idx)]
                        no_match = no_match[list(no_match_idx - no_match_but_iou_idx)]

        return no_match, pred_more

    def gt_pred_map(self):
        gt = self.gt.clone()
        pred = self.pred.clone()
        if self.right_match is None or self.error_match is None:
            _,_ = self._get_error_det(pred, gt)
        return self.right_match, self.error_match

    def _box_area(self, box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])

    def _box_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self._box_area(box1.T)[:, None] + self._box_area(box2.T) - inter + eps)


if __name__ == '__main__':
    img_path = Path(r'E:\detection\fold\images')
    gt_path = Path(r'E:\detection\fold\gts')
    pred_path = Path(r'E:\detection\fold\preds')
    result = Path(r'E:\detection\fold\results')

    for i in sorted(img_path.iterdir()):
        if i.suffix == '.jpg':
            img = cv2.imread(str(i))
            res = result/i.name
            if res.exists():
                res = cv2.imread(str(res))
            h,w,_ = img.shape
            gt = (gt_path/(i.stem+'.txt'))
            if gt.exists():
                gt =[list(map(float,line.split())) for line in  gt.read_text().strip().splitlines()]
                if gt == []:
                    gt = torch.empty([0,5],dtype=torch.float32)
                else:
                    gt = torch.tensor(gt,dtype=torch.float32)
                    bbox = gt[:, 1:].clone() * torch.tensor([w, h, w, h], dtype=torch.float32)
                    gt[:, 1:2] = bbox[:, 0:1] - 0.5 * bbox[:, 2:3]
                    gt[:, 2:3] = bbox[:, 1:2] - 0.5 * bbox[:, 3:4]
                    gt[:, 3:4] = gt[:, 1:2] + bbox[:, 2:3]
                    gt[:, 4:5] = gt[:, 2:3] + bbox[:, 3:4]

            else:
                gt = torch.empty([0,5],dtype=torch.float32)

            pred = (pred_path / (i.stem+'.txt'))
            if pred.exists():
                pred = [list(map(float,line.split())) for line in  pred.read_text().strip().splitlines()]
                if pred == []:
                    pred = torch.empty([0, 6], dtype=torch.float32)
                else:
                    pred = torch.tensor(pred,dtype=torch.float32)
                    bbox = pred[:, 1:5].clone() * torch.tensor([w, h, w, h], dtype=torch.float32)
                    cl,conf = pred[:,0:1].clone(),pred[:,5:6].clone()
                    pred[:,0:1] = bbox[:, 0:1] - 0.5 * bbox[:, 2:3]
                    pred[:,1:2] = bbox[:, 1:2] - 0.5 * bbox[:, 3:4]
                    pred[:,2:3] = pred[:, 0:1] + bbox[:, 2:3]
                    pred[:,3:4] = pred[:, 1:2] + bbox[:, 3:4]
                    pred[:,4:5] = conf
                    pred[:, 5:6] = cl
            else:
                pred = torch.empty([0, 6], dtype=torch.float32)

            tools = TMTools(img,gt,pred)

            cv2.namedWindow("gt",cv2.WINDOW_FREERATIO)
            cv2.namedWindow("res", cv2.WINDOW_FREERATIO)
            cv2.namedWindow("no_match",cv2.WINDOW_FREERATIO)
            cv2.namedWindow("error_match",cv2.WINDOW_FREERATIO)
            image_no_match, image_error_match = tools.draw_no_match()
            image_gt = tools.draw_gt()
            cv2.imshow('gt',image_gt)
            if isinstance(res,np.ndarray):
                cv2.imshow('res',res)
            cv2.imshow('no_match', image_no_match)
            cv2.imshow('error_match', image_error_match)
            cv2.waitKey(0)

            # break


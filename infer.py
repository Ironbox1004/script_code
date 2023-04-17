#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 14:50
# @Author  : chen_zhen
# @Email   : ha_bseligkeit@163.com
# @File    : dete_detion.py
import concurrent.futures
import threading
import uuid
import asyncio
from queue import Queue
from ..config.configs import general_image_model
import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from locale import currency
from mmdet.apis import init_detector, inference_detector
from mmcls.apis import init_model as init_classifier, inference_modelV2 as inference_classifier

__all__ = ["create_general_image_infer"]

def draw_labels_and_boxs(img, labels, boxes, thickness=1, box_color=(0,255,0), label_color=(0,255,0)):


    def draw_one_label_and_box(draw_img, label, box):
        cv2.rectangle(draw_img, (box[0], box[1]), (box[2], box[3]), color=box_color, thickness=thickness)
        cv2.putText(draw_img, label, (box[0], box[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, thickness=1)
        return draw_img

    for label, box in zip(labels, boxes):
        img = draw_one_label_and_box(img, label, box)
        return img

class Job():
    """推理任务封装"""
    def __init__(self, inp, future):
        self.future = future
        self.inp = inp

def create_job(inp, loop):
    if loop is None:
        future = currency.futures.Future()
    else:
        future = loop.create_future()
    return Job(inp, future)

class Infer():
    """模型推理封装"""
    @staticmethod
    def forward(self, job):
        pass

class GeneralImageInferResult():
    """图像识别结果封装"""

    class Target():
        def __init__(self):
            self.bbox = []
            self.detect_label = None
            self.detect_score = 0.0
            self.classify_label = None
            self.classify_score = 0.0
            self.keep = True
            self.id = uuid.uuid1().hex

        @staticmethod
        def ltrb_to_four_point(bbox):
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[1]
            x3, y3 = bbox[2], bbox[3]
            x4, y4 = bbox[0], bbox[3]
            return [x1, y1, x2, y2, x3, y3, x4, y4]

        @property
        def label(self):
            return f"unknown {self.detect_label}" if self.classify_label is None else self.classify_label

        @property
        def score(self):
            return self.detect_score

        def state_dict(self):
            return {
                "BBOX": self.ltrb_to_four_point(self.bbox),
                "LABEL": self.label,
                "SCORE": self.score,
                "DETECT_LABEL": self.detect_label,
                "DETECT_SCORE": self.detect_score,
                "CLASSIFY_LABEL": self.classify_label,
                "CLASSIFY_SCORE": self.classify_score,
                "TARGETID": self.id
            }

    def make_target(self):
        return self.Target()

    def __init__(self):
        self.width = None
        self.height = None
        self.tragets = []

    def state_dict(self):
        return {"WIDTH": self.width, "HEIGHT": self.height, "TRAGETS": [t.state_dict() for t in self.tragets if t.keep]}

class GeneralImageInferImpl():
    """识别图片，推理结果"""
    def __init__(self,
                 detect_config,
                 detect_checkpoint,
                 detect_score_thr,
                 classify_config,
                 classify_checkpoint,
                 classify_score_thr,
                 job_limit_size,
                 max_batch_size,
                 local_save=False
                 ):
        self.device = 'cuda:0'

        self._detect_config = detect_config
        self._detect_checkpoint = detect_checkpoint
        self._detect_score_thr = detect_score_thr

        self._classify_config = classify_config
        self._classify_checkpoint = classify_checkpoint
        self._classify_score_thr = classify_score_thr

        self._detect_model = None
        self._classify_model = None

        self._running = False
        self._jobs = Queue()
        self._cv = threading.Condition()
        self._worker_thread = None
        self._job_limit_size = job_limit_size
        self._max_batch_size = max_batch_size

        self._event_loop = None  # 异步推理事件循环，保证在消费线程中触发其他线程（例如生产线程）中的事件
        self.local_save = local_save


    def startup(self):
            """创建模型"""
            self._running = True
            future = concurrent.futures.Future()
            self._worker_thread = threading.Thread(target=self.dowork, args=(future,))
            self._worker_thread.start()
            return future.result()

    def dowork(self, future:asyncio.Future):
        """推理线程，消费者"""
        try:
            self._detect_model = init_detector(self._detect_config, self._detect_config, device=self.device)
            self._classify_model = init_classifier(self._classify_config, self._classify_checkpoint, device=self.device)
            print("GeneralImageInfer UP...")
        except:
            future.set_result(False)
            return

        future.set_result(True)
        fetch_job = []
        while self._running:
            with self._cv:
                if self._jobs.empty():
                    self._cv.notify() #通知生产，解除生产阻塞
                    self._cv.wait() #阻塞，避免轮询浪费CPU
                if not self._running: break
                for _ in range(min(self._jobs.qsize(), self._max_batch_size)):
                    fetch_job.append(self._jobs.get())

            print(f"inference with batch size {len(fetch_job)}")
            results = self.pipeline([job.inp for job in fetch_job])
            for result, job in zip(results, fetch_job):
                if self._enent_loop is None:
                    job.future.set_result(result)
                else:
                    self._enent_loop.call_soon_threadsafe(job.future.set_result, result)
            fetch_job.clear()

    def forward(self, inps):
        inps = inps if isinstance(inps, list) else [inps]
        jobs = [create_job(inp, self._enent_loop) for inp in inps]
        with self._cv:
            if self._jobs.qsize() > self._job_limit_size:
                self._cv.wait()
            for job in jobs:
                self.jobs.put(job)
            print(f"incerase job num:{len(jobs)},current job num:{self._jobs.qsize()}")
        return [job.future for job in jobs]

    def download(self, urls):
        images = list(map(lambda url: np.array(Image.open(BytesIO(requests.get(url).content))), urls))
        # images =Image.open(BytesIO(urls))
        return images

    def detect(self,images):
        results = inference_detector(self._detect_model, images)
        # print(results)
        def transform(image, result):
            r = GeneralImageInferResult()
            r.height = image.shape[0]
            r.width = image.shape[1]
            for label_index, boxes in enumerate(result):
                boxes = boxes.tolist()
                for box_index in range(len(boxes)):
                    score = boxes[box_index][4]
                    if score < self._detect_score_thr: continue
                    traget = r.make_target()
                    traget.detect_label = self._detect_model.CLASSES[label_index]
                    traget.detect_score = round(float(score), 3)
                    traget.bbox = list(map(int, boxes[box_index][:4]))
                    traget.raw = image[traget.bbox[1]:traget.bbox[3],traget.bbox[0]:traget.bbox[2]].copy()
                    r.tragets.append(traget)
            return r
        return [transform(image, result) for image, result in zip(images, results)]

    def classify(self,targets):
        pred_scores, pred_labels = inference_classifier(self._classify_model, [target.raw for target in targets])
        def transform(target, score, label):
            target.classify_label = label
            target.classify_score = round(float(score), 3)
            target.raw = None
            target.keep = True if score > self._classify_score_thr else False
            for target, score, label in zip(targets, pred_scores, pred_labels):
                transform(target, score, label)

    def pipeline(self,urls):
        images = self.download(urls)
        results = self.detect(images)
        targets = [traget for result in results for traget in result.tragets]
        self.classify(targets)

        if self.local_save:
            self.save(images,results)

        return [result.state_dict() for result in results]

    def save(self, images, results):
        for index, (image, result) in enumerate(zip(images, results)):
            boxes = [target.bbox for target in result.tragets if target.keep]
            labels = [f"{target.label}" for target in result.tragets if target.keep]
            draw_image = draw_labels_and_boxs(image, labels, boxes)
            cv2.imwrite(f"{index}.png", draw_image[:, :, ::-1])# BGR to RGB

    def shutdown(self):
        if self._running:
            self.running = False
            with self._cv:
                self._cv.notify()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join()

    def bind_event_loop(self, loop):
        self._event_loop = loop
        return self

    def __del__(self):
        self.shutdown()

def create_general_image_infer():
    infer = GeneralImageInferImpl(**general_image_model)
    if infer.startup():
        return infer
    else:
        return None
#
# if __name__ == '__main__':
#     path = ["https://img0.baidu.com/it/u=2374104030,2827647279&fm=253&fmt=auto&app=138&f=JPEG?w=640&h=296"]
#     img_path = "E:\work_code\mm_about_code\\result.jpg"
#     G = GeneralImageInferImpl(
#         detect_config_file="E:\work_code\mm_about_code\config\yolox_s_8x8_300e_coco.py",
#         detect_checkpoint_file= "E:\work_code\mm_about_code\chickpoint\\best_bbox_mAP_epoch_297.pth",
#         cla_config_file="E:\work_code\mm_about_code\config\\resnet34_8xb32_in1k.py",
#         cla_checkpoint_file="E:\work_code\mm_about_code\chickpoint\latest.pth",)
#     img = G.download(path)
#     result = G.pipeline(img)
#
#     print(result)

    # img = Image.open(img_path)
    # img.show()
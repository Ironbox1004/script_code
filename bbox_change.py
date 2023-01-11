import numpy as np
import torch


#  ===============================================================================#
#  坐标转换系列函数
#  输入：可能是 列表、np矩阵、tensor矩阵 以下六个函数可以保证输入输出的维度一致
#  输入的维度可能是一个向量shape=(4,)（.T转置之后的到的是原变量）
#  ===============================================================================#


def ltwh2center(bbox):
    """

    :param bbox:[left, top, w, h]
    :return:[cx, cy, w, h]
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    if bbox.shape[-1] != 4:
        raise ValueError('bbox.shape[-1] should equal 4')
    else:
        if isinstance(bbox, np.ndarray):
            left, top, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            # cx=left+w/2; cy=top+h/2;w;h
            _bbox = np.array([left + w / 2, top + h / 2, w, h])
            _bbox = _bbox.T
            return _bbox

        if isinstance(bbox, torch.Tensor):
            left, top, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            # cx=left+w/2; cy=top+h/2;w;h
            _bbox = torch.stack((left + w / 2, top + h / 2, w, h), dim=-1)
            return _bbox


def ltwh2corner(bbox):
    """

    :param bbox:[left, top, w, h]
    :return:[left, top, right, bottom]
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    if bbox.shape[-1] != 4:
        raise ValueError('bbox.shape[-1] should equal 4')
    else:
        if isinstance(bbox, np.ndarray):
            left, top, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            # left; top; right=left+w; bottom=top+h
            _bbox = np.stack([left, top, left + w, top + h], axis=-1)
            return _bbox

        if isinstance(bbox, torch.Tensor):
            left, top, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            _bbox = torch.stack((left, top, left + w, top + h), dim=-1)
            return _bbox


def corner2ltwh(bbox):
    """

    :param bbox:[left, top, right, bottom]
    :return:[left, top, w, h]
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    if bbox.shape[-1] != 4:
        raise ValueError('bbox.shape[-1] should equal 4')
    else:
        if isinstance(bbox, np.ndarray):
            left, top, right, bottom = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            # left; top; w=right-left; h=bottom-top
            _bbox = np.stack([left, top, right - left, bottom - top], axis=-1)
            return _bbox

        if isinstance(bbox, torch.Tensor):
            left, top, right, bottom = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            _bbox = torch.stack((left, top, right - left, bottom - top), dim=-1)
            return _bbox


def corner2center(bbox):
    """

    :param bbox:[left, top, right, bottom]
    :return:[cx,cy, w, h]
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    if bbox.shape[-1] != 4:
        raise ValueError('bbox.shape[-1] should equal 4')
    else:
        if isinstance(bbox, np.ndarray):
            left, top, right, bottom = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            # cx=(left+right)/2; cy=(top+bottom)/2; w=right-left; h=bottom-top
            _bbox = np.stack([(left + right) / 2, (top + bottom) / 2, right - left, bottom - top], axis=-1)
            return _bbox

        if isinstance(bbox, torch.Tensor):
            left, top, right, bottom = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            _bbox = torch.stack(((left + right) / 2, (top + bottom) / 2, right - left, bottom - top), dim=-1)
            return _bbox


def center2corner(bbox):
    """

    :param bbox: [cx,cy,w,h]
    :return: [left, top, right, bottom]
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    if bbox.shape[-1] != 4:
        raise ValueError('bbox.shape[-1] should equal 4')
    else:
        if isinstance(bbox, np.ndarray):
            cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            # left=cx-w/2; top=cy-h/2; right=cx+w/2; bottom=cy+h/2
            _bbox = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
            return _bbox

        if isinstance(bbox, torch.Tensor):
            cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            _bbox = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
            return _bbox


def center2ltwh(bbox):
    """

    :param bbox: [cx, cy, w, h]
    :return: [left, top, w, h]
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    if bbox.shape[-1] != 4:
        raise ValueError('bbox.shape[-1] should equal 4')
    else:
        if isinstance(bbox, np.ndarray):
            cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            # left=cx-w/2; top=cy-h/2; w; h
            _bbox = np.stack([cx - w / 2, cy - h / 2, w, h], axis=-1)  # cx,cy,w,h
            return _bbox

        if isinstance(bbox, torch.Tensor):
            cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            _bbox = torch.stack((cx - w / 2, cy - h / 2, w, h), dim=-1)  # 将数据坐标拼接起来
            return _bbox


if __name__ == '__main__':
    print('Start...')
    box1 = [50, 50, 100, 200]  # list
    box2 = np.array([50, 50, 120, 220])  # 一个坐标
    box3 = np.array([[50, 50, 100, 200], [50, 50, 120, 220], [50, 50, 120, 220]])  # 多个坐标
    box4 = torch.FloatTensor([50, 50, 100, 200])  # 一个tensor坐标数据
    box5 = torch.FloatTensor([[50, 50, 100, 200], [50, 50, 120, 220], [50, 50, 120, 220]])  # 多个tensor坐标数据

    for box in [box1, box2, box3, box4, box5]:
        box_ = ltwh2center(box)
        print('\n', 'input (%s):\n' % type(box), box, '\n', 'output(%s):\n' % type(box_), box_)
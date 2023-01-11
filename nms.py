import numpy as np
import torch

# 类间nms
def nms(bboxes, scores, thresh):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score降序排序（保存的是索引）
    # values, indices = torch.sort(scores, descending=True)
    indices = scores.sort(descending=True)[1]  # torch

    indice_res = torch.randn([1, 4]).to(bboxes)
    while indices.size()[0] > 0:  # indices.size()是一个Size对象，我们要取第一个元素是int，才能比较
        save_idx, other_idx = indices[0], indices[1:]
        indice_res = torch.cat((indice_res, bboxes[save_idx].unsqueeze(0)),
                               dim=0)  # unsqueeze是添加一个维度，让bboxes.shape从[4]-->[1,4]

        inter_x1 = torch.max(x1[save_idx], x1[other_idx])
        inter_y1 = torch.max(y1[save_idx], y1[other_idx])
        inter_x2 = torch.min(x2[save_idx], x2[other_idx])
        inter_y2 = torch.min(y2[save_idx], y2[other_idx])
        inter_w = torch.max(inter_x2 - inter_x1 + 1, torch.tensor(0).to(bboxes))
        inter_h = torch.max(inter_y2 - inter_y1 + 1, torch.tensor(0).to(bboxes))

        inter_area = inter_w * inter_h
        union_area = areas[save_idx] + areas[other_idx] - inter_area + 1e-6
        iou = inter_area / union_area

        indices = other_idx[iou < thresh]
    return indice_res[1:]


# 类内nms，把不同类别的乘以一个偏移量，把不同类别的bboxes给偏移到不同位置。
def class_nms(bboxes, scores, cat_ids, iou_threshold):
    '''
    :param bboxes: torch.tensor([n, 4], dtype=torch.float32)
    :param scores: torch.tensor([n], dtype=torch.float32)
    :param cat_ids: torch.tensor([n], dtype=torch.int32)
    :param iou_threshold: float
    '''
    max_coordinate = bboxes.max()

    # 为每一个类别/每一层生成一个很大的偏移量
    offsets = cat_ids * (max_coordinate + 1)
    # bboxes加上对应类别的偏移量后，保证不同类别之间bboxes不会有重合的现象
    bboxes_for_nms = bboxes + offsets[:, None]
    indice_res = nms(bboxes_for_nms, scores, iou_threshold)
    return indice_res


if __name__ == '__main__':
    results = np.array([
        [687.59113, 539.78339, 805.35858, 649.05060, 0.96555],
        [1013.53613, 512.57947, 1138.52246, 623.96619, 0.95386],
        [874.64288, 478.80649, 966.48077, 583.75922, 0.95210],
        [576.93927, 663.84839, 760.81537, 831.45447, 0.94741],
        [1085.92249, 741.70770, 1339.45789, 1029.21619, 0.93562],
        [1220.95044, 555.54449, 1393.67383, 702.83319, 0.93431],
        [81.90627, 453.37640, 147.86510, 491.30829, 0.92621],
        [804.80139, 739.23035, 999.60278, 975.12268, 0.92448],
        [923.35809, 304.13034, 945.79144, 319.63284, 0.91898],
        [668.47394, 342.76257, 704.24835, 371.92175, 0.91715],
        [513.74420, 576.99683, 647.66852, 703.31836, 0.91407]

    ])

    bboxes = results[:, :4]
    score = results[:, 4]
    ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).reshape(-1, 1)
    indice_res = class_nms(torch.tensor(bboxes), torch.tensor(score), torch.tensor(ids), 0.65)
    print(indice_res)
def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (
        (inter_x_max - inter_x_min) *
        (inter_y_max - inter_y_min)
    )

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def find_best_iou(target_label, box, bounding_boxes, matched_gts):
    best_iou = 0.0
    best_gt_idx = -1
    
    # Iterate with index so we can track which ground truths are "consumed"
    for idx, (gt_label, tl_, br_) in enumerate(bounding_boxes):
        # Only compare if the labels match AND the ground truth hasn't been matched yet
        if gt_label == target_label and idx not in matched_gts:
            iou_value = iou(box, tl_ + br_)
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_idx = idx
                
    return best_iou, best_gt_idx

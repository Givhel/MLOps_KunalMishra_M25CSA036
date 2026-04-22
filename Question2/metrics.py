import torch

def compute_iou(preds, masks, num_classes=23):
    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (masks == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue

        ious.append(intersection / union)

    return torch.mean(torch.tensor(ious))


def compute_dice(preds, masks, num_classes=23):
    preds = torch.argmax(preds, dim=1)
    dices = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (masks == cls)

        intersection = (pred_inds & target_inds).sum().float()
        total = pred_inds.sum() + target_inds.sum()

        if total == 0:
            continue

        dices.append((2 * intersection) / total)

    return torch.mean(torch.tensor(dices))

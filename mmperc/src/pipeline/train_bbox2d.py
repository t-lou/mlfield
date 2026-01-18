from label.bev_labels import generate_bev_labels_bbox2d
from losses.detection_losses import focal_loss, l1_loss


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        points = batch["points"].to(device)
        images = batch["camera"].to(device)
        gt_boxes = batch["gt_boxes"]  # list of tensors, not moved to device yet

        optimizer.zero_grad()

        # 1. Forward pass
        pred = model(points, images)
        heatmap_pred = pred["heatmap"]
        reg_pred = pred["reg"]

        # 2. Generate BEV labels
        heatmap_gt, reg_gt, mask_gt = generate_bev_labels_bbox2d(gt_boxes)

        heatmap_gt = heatmap_gt.to(device)
        reg_gt = reg_gt.to(device)
        mask_gt = mask_gt.to(device)

        # 3. Compute losses
        loss_hm = focal_loss(heatmap_pred, heatmap_gt)
        loss_reg = l1_loss(reg_pred, reg_gt, mask_gt)

        loss = loss_hm + loss_reg

        # 4. Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

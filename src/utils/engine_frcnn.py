from loguru import logger
# Placeholder training loop for Faster R-CNN is in the model file.
# utils/engine_frcnn.py
import torch
from loguru import logger

@torch.no_grad()
def _yolo_to_xyxy(bboxes_yolo, H, W):
    """
    Convert a list of YOLO-format boxes [(cx, cy, w, h), ...] to
    absolute XYXY in pixels for a single image.
    """
    xyxy = []
    for cx, cy, w, h in bboxes_yolo:
        cx *= W; cy *= H; w *= W; h *= H
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        xyxy.append([x1, y1, x2, y2])
    return xyxy

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, scaler=None, max_norm=None):
    """
    Train Faster R-CNN for one epoch.

    Args:
        model: torchvision detection model
        optimizer: torch optimizer
        data_loader: yields (images, targets) where
            - images: list[Tensor(C,H,W)] in [0..1]
            - targets: list[dict] with keys:
                'boxes_yolo': list[(cx,cy,w,h)] normalized
                'size': (H, W)
        device: 'cuda' or 'cpu'
        epoch: int (for logging)
        print_freq: log every N iters
        scaler: optional GradScaler for mixed precision
        max_norm: optional gradient clipping max norm
    """
    model.train()
    use_amp = scaler is not None

    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        # Move images to device
        images = [img.to(device) for img in images]

        # Convert YOLO-format targets to torchvision format
        tv_targets = []
        for t in targets:
            H, W = t['size']  # original image size before transforms
            boxes_yolo = t.get('boxes_yolo', [])
            boxes_xyxy = _yolo_to_xyxy(boxes_yolo, H, W)
            if len(boxes_xyxy) > 0:
                boxes = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device)
                labels = torch.ones((len(boxes_xyxy),), dtype=torch.int64, device=device)  # single class: car=1
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                labels = torch.zeros((0,), dtype=torch.int64, device=device)

            tv_targets.append({
                "boxes": boxes,
                "labels": labels,
                # Optional fields like "image_id", "area", "iscrowd" can be added if needed
            })

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, tv_targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, tv_targets)
            loss = sum(loss_dict.values())
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        running_loss += float(loss.item())

        if (i + 1) % print_freq == 0:
            avg = running_loss / print_freq
            logger.info(
                f"Epoch {epoch} | Iter {i+1}/{len(data_loader)} "
                f"loss={avg:.4f} "
                + " ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            )
            running_loss = 0.0

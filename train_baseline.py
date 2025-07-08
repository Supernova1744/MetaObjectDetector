import os
import csv
import torch
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
from loguru import logger
from datetime import datetime
from torch.utils.data import DataLoader
from FSODFlexibleDataset import FSODFlexibleDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ✅ Build model
def get_faster_rcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    Image_dir = r"C:\Users\A.Mohammed.ext\Downloads\data"
    Annotation_path = r"C:\Users\A.Mohammed.ext\Downloads\data\annotations\fsod_train.json"
    Annotation_path_test = r"C:\Users\A.Mohammed.ext\Downloads\data\annotations\fsod_test.json"

    # ✅ Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((416, 416)),                         # Resize to 416x416
        T.ToTensor(),                                 # Convert to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406],        # Imagenet normalization
                    std=[0.229, 0.224, 0.225]),
    ])

    # ✅ FSOD episodic sampler (for training and validation split)
    fsod = FSODFlexibleDataset(
        image_dir=Image_dir,
        annotation_path=Annotation_path,
        n_ways=2, k_shots=2, q_queries=2, max_samples=10, transform=transform, mode='standard'
    )

    fsod_test = FSODFlexibleDataset(
        image_dir=Image_dir,
        annotation_path=Annotation_path_test,
        n_ways=2, k_shots=2, q_queries=2, max_samples=10, transform=transform, mode='standard'
    )


    train_loader = DataLoader(fsod, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(fsod_test, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_faster_rcnn(num_classes=fsod.num_classes + 1).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # ✅ Logger
    log_file = f"baseline_loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("checkpoints", exist_ok=True)
    best_val_map = 0.0
    no_epochs = 100

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Batch', 'TrainLoss', 'Val_mAP'])

        for epoch in range(no_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                
                loss = sum(loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                writer.writerow([epoch, batch_idx, loss.item(), ''])

            avg_train_loss = epoch_loss / len(train_loader)

            # ✅ Validation
            model.eval()
            map_metric = MeanAveragePrecision()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    preds = model(images)

                    # Format targets/preds for torchmetrics
                    formatted_preds = [
                        {
                            "boxes": p["boxes"].cpu(),
                            "scores": p["scores"].cpu(),
                            "labels": p["labels"].cpu()
                        } for p in preds
                    ]
                    formatted_tgts = [
                        {
                            "boxes": t["boxes"].cpu(),
                            "labels": t["labels"].cpu()
                        } for t in targets
                    ]

                    map_metric.update(formatted_preds, formatted_tgts)

            val_map = map_metric.compute()["map"].item()
            logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val mAP: {val_map:.4f}")
            writer.writerow([epoch, '', avg_train_loss, val_map])

            # ✅ Save best model
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(model.state_dict(), f"checkpoints/best_fasterrcnn.pth")
                logger.info(f"✔️ Best model saved (mAP={val_map:.4f})")

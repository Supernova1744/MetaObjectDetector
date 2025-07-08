import learn2learn as l2l
import torchvision
import torch
import torchvision.transforms as T
from FSODFlexibleDataset import FSODFlexibleDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import csv, os
from datetime import datetime
from loguru import logger
from tqdm import tqdm

def get_faster_rcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class MetaDetectionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        return self.model(images, targets)

if __name__ == "__main__":
    # --- Setup ---
    Image_dir = r"C:\Users\A.Mohammed.ext\Downloads\data"
    Annotation_path = r"C:\Users\A.Mohammed.ext\Downloads\data\annotations\fsod_train.json"
    episodes_per_epoch = 5
    val_episodes = 5
    num_epochs = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    fsod = FSODFlexibleDataset(
        image_dir=Image_dir,
        annotation_path=Annotation_path,
        n_ways=2, k_shots=2, q_queries=2,
        max_samples=20,
        mode='episodic',
        transform=transform
    )

    # --- Model ---
    model = get_faster_rcnn(fsod.num_classes + 1)
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    meta_model = MetaDetectionWrapper(model).to(device)
    maml = l2l.algorithms.MAML(meta_model, lr=1e-2, first_order=True)
    meta_optimizer = torch.optim.SGD(maml.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # --- Logging ---
    os.makedirs("checkpoints", exist_ok=True)
    log_file = f"maml_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    best_val_map = 0.0

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'SupportLoss', 'QueryLoss', 'Val_mAP'])

        for epoch in range(num_epochs):
            total_support_loss = 0.0
            total_query_loss = 0.0

            for episode in tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch}"):
                support, query = fsod.sample_episode()

                learner = maml.clone()
                learner.train()

                # --- Support ---
                s_imgs, s_tgts = zip(*support)
                s_imgs = [img.to(device) for img in s_imgs]
                s_tgts = [{k: v.to(device) for k, v in t.items()} for t in s_tgts]
                support_loss_dict = learner(s_imgs, s_tgts)
                support_loss = sum(support_loss_dict.values())
                learner.adapt(support_loss)
                total_support_loss += support_loss.detach().item()

                # --- Query ---
                q_imgs, q_tgts = zip(*query)
                q_imgs = [img.to(device) for img in q_imgs]
                q_tgts = [{k: v.to(device) for k, v in t.items()} for t in q_tgts]
                query_loss_dict = learner(q_imgs, q_tgts)
                query_loss = sum(query_loss_dict.values())
                query_loss.backward()
                meta_optimizer.step()
                meta_optimizer.zero_grad()
                total_query_loss += query_loss.detach().item()

            # --- Validation ---
            map_metric = MeanAveragePrecision()
            model.eval()

            for _ in range(val_episodes):
                support, query = fsod.sample_episode()

                val_learner = maml.clone()
                val_learner.train()

                # Adapt
                s_imgs, s_tgts = zip(*support)
                s_imgs = [img.to(device) for img in s_imgs]
                s_tgts = [{k: v.to(device) for k, v in t.items()} for t in s_tgts]
                val_learner.adapt(sum(val_learner(s_imgs, s_tgts).values()))
                # support_loss_dict = val_learner(s_imgs, s_tgts)
                # support_loss = sum(support_loss_dict.values())
                # val_learner.adapt(support_loss, allow_nograd=True)

                with torch.no_grad():
                    val_learner.eval()
                    # Evaluate
                    q_imgs, q_tgts = zip(*query)
                    q_imgs = [img.to(device) for img in q_imgs]
                    q_tgts = [{k: v.to(device) for k, v in t.items()} for t in q_tgts]
                    preds = val_learner(q_imgs)

                    formatted_preds = [
                        {
                            "boxes": p["boxes"].cpu(),
                            "scores": p["scores"].cpu(),
                            "labels": p["labels"].cpu()
                        } for p in preds
                    ]
                    formatted_targets = [
                        {
                            "boxes": t["boxes"].cpu(),
                            "labels": t["labels"].cpu()
                        } for t in q_tgts
                    ]
                    map_metric.update(formatted_preds, formatted_targets)

            val_map = map_metric.compute()["map"].item()
            model.train()

            avg_support_loss = total_support_loss / episodes_per_epoch
            avg_query_loss = total_query_loss / episodes_per_epoch
            logger.info(f"[Epoch {epoch}] Support: {avg_support_loss:.4f} | Query: {avg_query_loss:.4f} | Val mAP: {val_map:.4f}")
            writer.writerow([epoch, avg_support_loss, avg_query_loss, val_map])

            # Save best model
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(model.state_dict(), f"checkpoints/best_maml_frcnn.pth")
                logger.info(f"✔️ Best model saved at epoch {epoch} (val mAP: {val_map:.4f})")

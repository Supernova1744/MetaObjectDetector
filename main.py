from pycocotools.coco import COCO
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms
import random
from PIL import Image
import learn2learn as l2l
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FewShotCOCO(CocoDetection):
    def __init__(self, root, annFile, k_shots=2, q_queries=2, transform=None, target_transform=None):
        super().__init__(root, annFile)
        self.coco = COCO(annFile)
        self.transform = transform
        self.target_transform = target_transform
        self.cat_ids = self.coco.getCatIds()
        self.img_ids_per_cat = {
            cat_id: self.coco.getImgIds(catIds=[cat_id])
            for cat_id in self.cat_ids
        }
        # Keep only classes that have enough images
        self.cat_ids = [
            cat_id for cat_id in self.cat_ids
            if len(self.coco.getImgIds(catIds=[cat_id])) >= (k_shots + q_queries)
        ]
        self.img_ids_per_cat = {
            cat_id: self.coco.getImgIds(catIds=[cat_id])
            for cat_id in self.cat_ids
        }

    def sample_task(self, n_ways=2, k_shots=5, q_queries=5):
        task_classes = random.sample(self.cat_ids, n_ways)
        support_set, query_set = [], []

        for cat_id in task_classes:
            img_ids = random.sample(self.img_ids_per_cat[cat_id], k_shots + q_queries)
            for i, img_id in enumerate(img_ids):
                img_info = self.coco.loadImgs([img_id])[0]
                path = img_info['file_name']
                image = Image.open(f"{self.root}/{path}").convert("RGB")
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id], catIds=[cat_id]))
                if len(anns) == 0:
                    continue  # skip empty samples
                target = {
                    "boxes": torch.tensor([xywh_to_xyxy(ann["bbox"]) for ann in anns], dtype=torch.float32),
                    "labels": torch.tensor([ann["category_id"] for ann in anns], dtype=torch.int64)
                }

                if self.transform:
                    image = self.transform(image)
                if i < k_shots:
                    support_set.append((image, target))
                else:
                    query_set.append((image, target))

        return support_set, query_set

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + max(1.0, w), y + max(1.0, h)]  # avoid zero width/height

def get_faster_rcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class MetaDetectionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        return self.model(images, targets)

from torch.optim import SGD

def detection_loss(outputs, targets):
    # outputs is a list of dicts with 'loss_classifier', 'loss_box_reg', ...
    total_loss = sum(loss for loss in outputs.values())
    return total_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
coco_dataset = FewShotCOCO(
    root=r"C:\Users\aly17\OneDrive\Desktop\MetaObjectDetector\chess\train",
    annFile=r"C:\Users\aly17\OneDrive\Desktop\MetaObjectDetector\chess\train\mini_annotations.coco.json",
    k_shots=2,
    q_queries=2,
    transform=transforms.ToTensor()
)

model = get_faster_rcnn(num_classes=13)  # For COCO
model.train()
for param in model.parameters():
    param.requires_grad = True
meta_model = MetaDetectionWrapper(model).to(device)
maml = l2l.algorithms.MAML(meta_model, lr=1e-2, first_order=True)
meta_optimizer = SGD(maml.parameters(), lr=1e-3)
num_epochs = 10

for epoch in range(num_epochs):
    support, query = coco_dataset.sample_task(n_ways=2, k_shots=2, q_queries=2)
    learner = maml.clone()

    # Fast adaptation
    learner.train()
    s_imgs, s_tgts = zip(*support)
    s_imgs = [img.to(device) for img in s_imgs]
    s_tgts = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in s_tgts]
    loss_dict = learner(s_imgs, s_tgts)  # dict of losses
    loss = sum(loss_dict.values())
    # import pdb; pdb.set_trace()
    # loss = detection_loss(support_loss, s_tgts)
    learner.adapt(loss)

    # Meta-update
    q_imgs, q_tgts = zip(*query)
    q_imgs = [img.to(device) for img in q_imgs]
    q_tgts = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in q_tgts]
    query_loss = detection_loss(learner(q_imgs, q_tgts), q_tgts)
    query_loss.backward()
    meta_optimizer.step()
    meta_optimizer.zero_grad()

    print(f"Epoch {epoch}: query_loss={query_loss.item():.4f}")

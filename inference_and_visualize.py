import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os
from FSODFlexibleDataset import FSODFlexibleDataset
import torchvision.transforms as T
import numpy as np

# --- Config ---
BASELINE_WEIGHTS = r'checkpoints\best_fasterrcnn_20250708_105102.pth'
MAML_WEIGHTS = r'checkpoints\best_maml_frcnn_20250708_141656.pth'
IMAGE_DIR = r'C:\Users\A.Mohammed.ext\Downloads\data'
ANNOTATION_PATH = r"C:\Users\A.Mohammed.ext\Downloads\data\annotations\fsod_test.json"
OUTPUT_DIR = 'inference_results'
N_SAMPLES = 5  # Number of images to visualize
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper: Model Loader ---
def get_faster_rcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --- Dataset ---
transform = T.Compose([
    T.Resize((416, 416)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = FSODFlexibleDataset(
    image_dir=IMAGE_DIR,
    annotation_path=ANNOTATION_PATH,
    n_ways=10, k_shots=1, q_queries=1, max_samples=20, mode='episodic', transform=transform
)

# --- Sample Images ---
support, query = ds.sample_episode()
samples = support[:N_SAMPLES] + query[:N_SAMPLES]

# --- Load Models ---
num_classes = ds.num_classes + 1

baseline_model = get_faster_rcnn(num_classes)
baseline_model.load_state_dict(torch.load(BASELINE_WEIGHTS, map_location=DEVICE))
baseline_model.eval().to(DEVICE)

maml_model = get_faster_rcnn(num_classes)
maml_model.load_state_dict(torch.load(MAML_WEIGHTS, map_location=DEVICE))
maml_model.eval().to(DEVICE)

# --- Inference and Visualization ---
for idx, (img, target) in enumerate(samples):
    img_input = img.unsqueeze(0).to(DEVICE)
    orig_img = img.clone()
    # Undo normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    vis_img = orig_img * std + mean
    vis_img = torch.clamp(vis_img, 0, 1)
    vis_img = (vis_img * 255).to(torch.uint8).cpu()

    # Baseline
    with torch.no_grad():
        preds = baseline_model(img_input)[0]
    # Filter by score threshold
    score_thresh = 0.25
    keep = preds['scores'].cpu() >= score_thresh
    boxes = preds['boxes'].cpu()[keep]
    labels = preds['labels'].cpu()[keep]
    scores = preds['scores'].cpu()[keep]

    # Optionally, create label strings with scores
    label_names = [f"{l.item()}:{s:.2f}" for l, s in zip(labels, scores)]

    drawn = draw_bounding_boxes(
        vis_img,
        boxes,
        labels=label_names,
        colors="red",
        width=3,
        font_size=18
    )
    out_img = to_pil_image(drawn)
    out_img.save(os.path.join(OUTPUT_DIR, f'baseline_sample_{idx}.png'))

    # MAML
    with torch.no_grad():
        preds = maml_model(img_input)[0]
    # Filter by score threshold
    score_thresh = 0.25
    keep = preds['scores'].cpu() >= score_thresh
    boxes = preds['boxes'].cpu()[keep]
    labels = preds['labels'].cpu()[keep]
    scores = preds['scores'].cpu()[keep]

    # Optionally, create label strings with scores
    label_names = [f"{l.item()}:{s:.2f}" for l, s in zip(labels, scores)]

    drawn = draw_bounding_boxes(
        vis_img,
        boxes,
        labels=label_names,
        colors="red",
        width=3,
        font_size=18
    )
    out_img = to_pil_image(drawn)
    out_img.save(os.path.join(OUTPUT_DIR, f'maml_sample_{idx}.png'))

print(f"Saved {2*len(samples)} images to {OUTPUT_DIR}/") 
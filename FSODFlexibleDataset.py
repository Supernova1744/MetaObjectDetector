import torch
from pycocotools.coco import COCO
from PIL import Image
import os
import random
import torchvision.transforms as T

class FSODFlexibleDataset:
    def __init__(self, image_dir, annotation_path,
                 mode='episodic',
                 n_ways=5, k_shots=5, q_queries=5,
                 target_size=(416, 416),
                 max_samples=None,
                 transform=None):
        
        assert mode in ['episodic', 'standard'], "mode must be 'episodic' or 'standard'"
        self.mode = mode
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.q_queries = q_queries
        self.max_samples = max_samples
        self.target_size = target_size
        self.cat_ids = sorted(self.coco.getCatIds())[:10]
        # import pdb; pdb.set_trace()
        self.num_classes = len(self.cat_ids)
        self.transform = transform

        self.img_ids_per_class = {
            cat_id: self.coco.getImgIds(catIds=[cat_id])[:self.max_samples]
            for cat_id in self.cat_ids
        }

        # Filter to ensure enough examples per class
        self.cat_ids = [
            cid for cid in self.cat_ids
            if len(self.img_ids_per_class[cid]) >= (k_shots + q_queries)
        ]

        # For standard mode: build flat image list
        if self.mode == 'standard':
            self.flat_img_ids = list(set(
                img_id
                for img_list in self.img_ids_per_class.values()
                for img_id in img_list
            ))
            if self.max_samples:
                self.flat_img_ids = self.flat_img_ids[:self.max_samples]

    def xywh_to_xyxy(self, box):
        x, y, w, h = box
        return [x, y, x + max(w, 1), y + max(h, 1)]

    def scale_boxes(self, boxes, orig_size, target_size):
        orig_w, orig_h = orig_size
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            scaled_boxes.append([x1, y1, x2, y2])
        return scaled_boxes

    def load_item(self, img_id, class_restrict=None):
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size

        # Load annotations
        if class_restrict is not None:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[class_restrict])
        else:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = [self.xywh_to_xyxy(ann["bbox"]) for ann in anns if ann["iscrowd"] == 0]
        labels = [ann["category_id"] for ann in anns if ann["iscrowd"] == 0]

        if not boxes:
            return None  # Skip invalid sample

        if self.transform:
            image = self.transform(image)

        boxes = self.scale_boxes(boxes, orig_size, self.target_size)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return image, target

    def sample_episode(self):
        assert self.mode == 'episodic', "sample_episode only valid in episodic mode"
        support, query = [], []
        selected_classes = random.sample(self.cat_ids, self.n_ways)

        for cat_id in selected_classes:
            img_ids = random.sample(self.img_ids_per_class[cat_id], self.k_shots + self.q_queries)
            for i, img_id in enumerate(img_ids):
                result = self.load_item(img_id, class_restrict=cat_id)
                if result is None:
                    continue
                image, target = result

                if i < self.k_shots:
                    support.append((image, target))
                else:
                    query.append((image, target))

        return support, query

    def __getitem__(self, idx):
        assert self.mode == 'standard', "__getitem__ only valid in standard mode"
        img_id = self.flat_img_ids[idx]
        result = self.load_item(img_id)
        while result is None:  # Skip invalid
            idx = (idx + 1) % len(self.flat_img_ids)
            img_id = self.flat_img_ids[idx]
            result = self.load_item(img_id)
        return result

    def __len__(self):
        assert self.mode == 'standard', "__len__ only valid in standard mode"
        return len(self.flat_img_ids)

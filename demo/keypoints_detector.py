from pathlib import Path
import sys

import cv2

sys.path.append("../lib")
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from yacs.config import CfgNode as CN
from models import pose_hrnet
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

# Константы для нормализации (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Средние значения RGB для ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]   # Стандартные отклонения RGB для ImageNet

# Константа для масштабирования
PIXEL_STD = 200  # Стандартный размер области интереса в пикселях (HRNet/COCO)


class KeypointsDetector:

    def __init__(self, model_yml: Path, model_path: Path, device: str = 'cpu') -> None:
        self.device = torch.device(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Загрузка конфигурации модели
        with model_yml.open("r", encoding="utf-8") as f:
            self.model_cfg = CN(yaml.safe_load(f))

        # Инициализация модели
        self.model = pose_hrnet.get_pose_net(self.model_cfg, is_train=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict_keypoints(self, image: np.ndarray) -> list[list[float]]:
        orig_height, orig_width = image.shape[:2]
        input_w, input_h = self.model_cfg.MODEL.IMAGE_SIZE  # [192, 256] — (width, height)
        center = np.array([orig_width / 2, orig_height / 2], dtype=np.float32)
        scale = np.array([orig_width / PIXEL_STD, orig_height / PIXEL_STD], dtype=np.float32)

        trans = get_affine_transform(center, scale, 0, [input_w, input_h])
        input_image = cv2.warpAffine(
            image,
            trans,
            (input_w, input_h),
            flags=cv2.INTER_LINEAR
        )
        model_input = self.transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(model_input)

        coords, _ = get_final_preds(
            self.model_cfg,
            output.cpu().numpy(),
            np.array([center]),
            np.array([scale])
        )
        return coords[0].tolist()

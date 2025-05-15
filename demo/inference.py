from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from yacs.config import CfgNode as CN
from pydantic import BaseModel, Field

# Предполагается, что HRNet находится в ../lib
import sys
sys.path.append("../lib")
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
import models

# Константы для нормализации (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Средние значения RGB для ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]   # Стандартные отклонения RGB для ImageNet

# Константа для масштабирования
PIXEL_STD = 200  # Стандартный размер области интереса в пикселях (HRNet/COCO)

# Контекст устройства
# CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CTX = torch.device('cpu')

# Имена ключевых точек для паспорта
PASSPORT_KEYPOINT_INDEXES = {
    0: 'point1',
    1: 'point2',
    2: 'point3',
    3: 'point4',
    4: 'point5',
    5: 'point6'
}

class InferenceConfig(BaseModel):
    source_dir: Path = Field(..., description="Directory with source images")
    extensions: list[str] = Field(..., description="List of valid image extensions")
    model_yml: Path = Field(..., description="Path to HRNet model config YAML")
    model_path: Path = Field(..., description="Path to trained HRNet model")
    display_height: int = Field(..., gt=0, description="Height of displayed image")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "InferenceConfig":
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

class InferenceApp:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.images = sorted([
            f for f in config.source_dir.iterdir() if f.suffix.lower() in self.config.extensions
        ])
        self.index = 0
        self.current_points: list[list[float]] = []
        self.current_image = None
        self.display_image = None
        self.scale = 1.0

        # Загрузка конфигурации модели
        with config.model_yml.open("r", encoding="utf-8") as f:
            model_cfg = CN(yaml.safe_load(f))

        # Настройка HRNet
        self.model = eval('models.' + model_cfg.MODEL.NAME + '.get_pose_net')(model_cfg, is_train=False)
        self.model.load_state_dict(torch.load(config.model_path, map_location=CTX))
        self.model.to(CTX)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.window_name = "Inference"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.run()

    def load_image(self) -> None:
        image_path = self.images[self.index]
        self.current_image = cv2.imread(str(image_path))
        self.current_points.clear()

        # Масштабирование для отображения
        display_height = self.config.display_height
        orig_height, orig_width = self.current_image.shape[:2]
        self.scale = display_height / orig_height
        new_width = int(orig_width * self.scale)
        self.display_image = cv2.resize(self.current_image, (new_width, display_height), interpolation=cv2.INTER_AREA)

        # Предсказание ключевых точек
        self.current_points = self.predict_keypoints()

    def predict_keypoints(self) -> list[list[float]]:
        with self.config.model_yml.open("r", encoding="utf-8") as f:
            model_cfg = CN(yaml.safe_load(f))

        image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        input_w, input_h = model_cfg.MODEL.IMAGE_SIZE  # [192, 256] — (width, height)
        center = np.array([orig_width / 2, orig_height / 2], dtype=np.float32)
        scale = np.array([orig_width / PIXEL_STD, orig_height / PIXEL_STD], dtype=np.float32)

        trans = get_affine_transform(center, scale, 0, [input_w, input_h])
        input_image = cv2.warpAffine(image, trans, (input_w, input_h), flags=cv2.INTER_LINEAR)
        model_input = self.transform(input_image).unsqueeze(0).to(CTX)

        with torch.no_grad():
            output = self.model(model_input)

        coords, _ = get_final_preds(model_cfg, output.cpu().numpy(), np.array([center]), np.array([scale]))
        return coords[0].tolist()

    def draw_interface(self, img: np.ndarray) -> np.ndarray:
        display = img.copy()
        for i, (x, y) in enumerate(self.current_points):
            display_x = int(x * self.scale)
            display_y = int(y * self.scale)
            cv2.circle(display, (display_x, display_y), 6, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (display_x + 5, display_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        image_name = self.images[self.index].name
        status_text = f"{image_name} | {self.index + 1}/{len(self.images)} | {len(self.current_points)}/6 points"
        help_text = "A/D: prev/next image, ESC: exit"
        y_offset = img.shape[0] - 5
        cv2.putText(display, status_text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, help_text, (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return display

    def run(self) -> None:
        while True:
            self.load_image()
            display = self.draw_interface(self.display_image)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(0)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return
            if key == ord("d") or key == 83:  # Right or 'd'
                self.index = min(len(self.images) - 1, self.index + 1)
            if key == ord("a") or key == 81:  # Left or 'a'
                self.index = max(0, self.index - 1)

if __name__ == "__main__":
    app_config = InferenceConfig.from_yaml(Path("inference.yml"))
    InferenceApp(app_config)

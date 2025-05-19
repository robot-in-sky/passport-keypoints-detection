from pathlib import Path

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

from validate import validate_pages
from keypoints_detector import KeypointsDetector


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
        self.is_valid = False

        # Инициализация детектора ключевых точек
        self.detector = KeypointsDetector(
            model_yml=config.model_yml,
            model_path=config.model_path,
            device='cpu'
        )

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
        image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        self.current_points = self.detector.predict_keypoints(image_rgb)

        # Валидация документа
        self.is_valid = validate_pages(
            self.current_points,
            (orig_width, orig_height)
        )

    def draw_interface(self, img: np.ndarray) -> np.ndarray:
        display = img.copy()
        for i, (x, y) in enumerate(self.current_points):
            display_x = int(x * self.scale)
            display_y = int(y * self.scale)
            color = (0, 255, 0) if self.is_valid else (0, 0, 255)
            cv2.circle(display, (display_x, display_y), 6, color, -1)
            cv2.putText(display, str(i + 1), (display_x + 5, display_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

        image_name = self.images[self.index].name
        status_text = f"{image_name} | {self.index + 1}/{len(self.images)} | {len(self.current_points)}/4 points"
        validity_text = "VALID" if self.is_valid else "INVALID"
        validity_color = (0, 255, 0) if self.is_valid else (0, 0, 255)
        help_text = "A/D: prev/next image, ESC: exit"

        # Добавляем текст статуса
        cv2.putText(display, status_text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, validity_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, validity_color, 2)
        cv2.putText(display, help_text, (5, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
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
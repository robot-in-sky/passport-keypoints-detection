from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field


class AnnotatorConfig(BaseModel):
    source_dir: Path = Field(..., description="Directory with source images")
    extensions: list[str] = Field(..., description="List of valid image extensions")
    annotations_path: Path = Field(..., description="Path to annotations YAML file")
    num_points: int = Field(..., gt=0, description="Number of keypoints to annotate")
    display_height: int = Field(..., gt=0, description="Height of displayed image")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AnnotatorConfig":
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class ImageAnnotation:
    image_name: str
    points: list[list[int]]  # pixel coordinates [X, Y]


class AnnotatorApp:

    def __init__(self, config: AnnotatorConfig) -> None:
        self.config = config
        self.images = sorted([
            f for f in config.source_dir.iterdir() if f.suffix.lower() in self.config.extensions
        ])
        self.annotations: dict[str, ImageAnnotation] = self.load_annotations()

        self.index = 0
        self.current_points: list[list[int]] = []
        self.current_image = None
        self.display_image = None
        self.scale = 1.0

        self.window_name = "Annotator"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_click)

        self.run()


    def load_annotations(self) -> dict[str, ImageAnnotation]:
        if self.config.annotations_path.exists():
            with self.config.annotations_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or []
            return {
                entry["image_name"]: ImageAnnotation(
                    image_name=entry["image_name"],
                    points=entry["points"],
                ) for entry in data
            }
        return {}


    def save_annotations(self) -> None:
        valid = [asdict(a) for a in self.annotations.values()]
        with self.config.annotations_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(valid, f, sort_keys=False, allow_unicode=True)


    def load_image(self) -> None:
        image_path = self.images[self.index]
        self.current_image = cv2.imread(str(image_path))
        self.current_points.clear()

        display_height = self.config.display_height
        orig_height, orig_width = self.current_image.shape[:2]
        self.scale = self.config.display_height / orig_height
        new_width = int(orig_width * self.scale)
        self.display_image = cv2.resize(self.current_image, (new_width, display_height), interpolation=cv2.INTER_AREA)

        image_name = image_path.name
        if image_name in self.annotations:
            self.current_points = self.annotations[image_name].points.copy()


    def on_click(self, event: int, x: int, y: int, flags: int, param: Any) -> None:  # noqa: PLR0913, ARG002
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.current_points) >= self.config.num_points:
            self.current_points.clear()

        # Convert click coordinates back to original image size
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)
        self.current_points.append([orig_x, orig_y])

        if len(self.current_points) == self.config.num_points:
            image_name = self.images[self.index].name
            self.annotations[image_name] = ImageAnnotation(image_name=image_name, points=self.current_points.copy())
            self.save_annotations()


    def draw_interface(self, img: np.ndarray) -> np.ndarray:
        display = img.copy()
        for i, (x, y) in enumerate(self.current_points):
            # Scale points for display
            display_x = int(x * self.scale)
            display_y = int(y * self.scale)
            cv2.circle(display, (display_x, display_y), 6, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (display_x + 5, display_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        image_name = self.images[self.index].name
        status_text = (f"{image_name} | "
                       f"{self.index + 1}/{len(self.images)} | "
                       f"{len(self.current_points)}/{self.config.num_points} points")
        help_text = "A/D: prev/next image, R: reset points, U: undo, ESC: save & exit"
        y_offset = img.shape[0] - 5
        cv2.putText(display, status_text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, help_text, (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return display


    def reset_current(self) -> None:
        self.current_points.clear()
        image_name = self.images[self.index].name
        if image_name in self.annotations:
            del self.annotations[image_name]
            self.save_annotations()


    def undo_current(self) -> None:
        if self.current_points:
            self.current_points.pop()


    def run(self) -> None:
        while True:
            self.load_image()
            while True:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) <= 0:
                    self.save_annotations()
                    cv2.destroyAllWindows()
                    return

                display = self.draw_interface(self.display_image)
                cv2.imshow(self.window_name, display)

                key = cv2.waitKey(20) & 0xFF

                if key == 27:  # ESC  # noqa: PLR2004
                    self.save_annotations()
                    cv2.destroyAllWindows()
                    return

                if key == ord("d") or key == 83:  # Right or 'd'  # noqa: PLR2004
                    self.index = min(len(self.images) - 1, self.index + 1)
                    break
                if key == ord("a") or key == 81:  # Left or 'a'  # noqa: PLR2004
                    self.index = max(0, self.index - 1)
                    break
                if key == ord("u"):  # Undo last point
                    self.undo_current()
                if key == ord("r"):  # Reset points
                    self.reset_current()


if __name__ == "__main__":
    app_config = AnnotatorConfig.from_yaml(Path("annotator.yml"))
    AnnotatorApp(app_config)

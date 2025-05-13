import os
import yaml
import json
import cv2
import shutil

from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image, ImageOps


def remove_exif_rotation(src_path, dst_path):
    """
    Удаляет EXIF-ориентацию из изображения, сохраняя его в правильном положении.

    Args:
        src_path (str): Путь к исходному изображению.
        dst_path (str): Путь для сохранения обработанного изображения.
    """
    try:
        # Открываем изображение
        img = Image.open(src_path)

        # Применяем EXIF-ориентацию, если она есть (автоматически поворачивает изображение)
        img = ImageOps.exif_transpose(img)

        # Сохраняем изображение без EXIF-данных
        img.save(dst_path, exif=b'')

        print(f"Изображение успешно обработано и сохранено в {dst_path}")
        return True
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return False

def yaml_to_coco(yaml_path, image_dir, output_dir,
                 train_ratio=0.8, bbox_padding=20,
                 remove_exif=False, split_seed=None):
    """
    Конвертирует YAML-аннотации в COCO-формат с копированием и переименованием изображений.
    Опционально удаляет EXIF-поворот, сохраняя аннотации, основанные на сырых пикселях.

    Args:
        yaml_path (str): Путь к YAML-файлу с аннотациями
        image_dir (str): Папка с исходными изображениями
        output_dir (str): Корневая папка для выходных данных (создаст подпапки)
        train_ratio (float): Доля данных для обучения (0.8 = 80%)
        bbox_padding (int): Отступ для bbox вокруг ключевых точек
        remove_exif (bool): Если True, удаляет EXIF-поворот из изображений
        split_seed (int | None): Сид разделения на train/val
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Создаем структуру папок
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/valid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    # Разделяем данные на train/val
    train_data, val_data = train_test_split(data, train_size=train_ratio, random_state=split_seed)

    # Обрабатываем train и val отдельно
    for split_name, split_data in [("train", train_data), ("valid", val_data)]:
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,
                "name": "passport",
                "keypoints": ["tl", "tr", "ml", "mr", "bl", "br"],
                "skeleton": []
            }]
        }

        for idx, item in enumerate(split_data, 1):
            original_img_path = Path(image_dir) / item['image_name']
            if not original_img_path.exists():
                print(f"⚠️ Ошибка: изображение {original_img_path} не найдено. Пропускаем.")
                continue

            # Новое имя файла (12 цифр с ведущими нулями)
            new_img_name = f"{idx:012d}.jpg"
            split_image_dir = os.path.join(output_dir, 'images', split_name)
            new_img_path = os.path.join(split_image_dir, new_img_name)

            if remove_exif:
                # Удаляем EXIF-поворот и сохраняем изображение
                remove_exif_rotation(original_img_path, new_img_path)
            else:
                # Копируем изображение без изменений
                shutil.copy2(original_img_path, new_img_path)

            # Получаем размеры изображения
            img = cv2.imread(str(new_img_path))
            if img is None:
                print(f"⚠️ Ошибка: не удалось загрузить {new_img_path}. Пропускаем.")
                continue
            h, w = img.shape[:2]

            # Добавляем изображение в аннотации
            coco["images"].append({
                "id": idx,
                "file_name": new_img_name,
                "height": h,
                "width": w
            })

            # Подготавливаем ключевые точки
            keypoints = []
            for x, y in item['points']:
                keypoints.extend([float(x), float(y), 2])  # 2 = точка видима

            # Вычисляем bbox с паддингом
            x_coords = [p[0] for p in item['points']]
            y_coords = [p[1] for p in item['points']]
            x_min = max(0, min(x_coords) - bbox_padding)
            y_min = max(0, min(y_coords) - bbox_padding)
            x_max = min(w, max(x_coords) + bbox_padding)
            y_max = min(h, max(y_coords) + bbox_padding)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Добавляем аннотацию
            coco["annotations"].append({
                "id": idx,
                "image_id": idx,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": 6,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })

        # Сохраняем JSON
        output_json_path = os.path.join(output_dir, 'annotations', f'person_keypoints_{split_name}.json')
        with open(output_json_path, 'w') as f:
            json.dump(coco, f, indent=2)

        print(f"✅ {split_name.upper()}:")
        print(f"   Изображения сохранены в {os.path.join(output_dir, 'images', split_name)}")
        print(f"   Аннотации сохранены в {output_json_path}\n")

# Пример вызова
if __name__ == "__main__":
    yaml_to_coco(
        yaml_path="../data/passport/annotations/annotations.yml",
        image_dir="../data/passport/images/src",
        output_dir="../data/passport",
        train_ratio=0.9,
        bbox_padding=20,
        remove_exif=True,
        split_seed=42,
    )

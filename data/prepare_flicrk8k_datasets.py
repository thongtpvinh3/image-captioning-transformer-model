import csv
import json
from pathlib import Path

# File hiện tại:
CURRENT_FILE = Path(__file__).resolve()

# Thư mục data/
CURRENT_DIR = CURRENT_FILE.parent

# Thư mục gốc project:
PROJECT_ROOT = CURRENT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "datasets"

RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

IMAGE_DIR = Path("C:/Users/Thong/Downloads/archive (1)/Images")
CAPTION_FILE = RAW_DATA_DIR / "captions.txt"
OUTPUT_FILE = PROCESSED_DATA_DIR / "image_to_captions.json"


def build_image_caption_mapping(image_dir: str | Path, caption_file: str | Path) -> dict[str, dict]:
    """
    Đọc caption.txt và tạo ánh xạ:

        image_name -> {
            "image_path": đường dẫn ảnh,
            "captions": danh sách caption
        }
    """

    image_dir = Path(image_dir)
    caption_file = Path(caption_file)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

    if not caption_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file caption: {caption_file}")

    image_to_captions: dict[str, dict] = {}

    with caption_file.open(mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("File caption không có header.")

        if "image" not in reader.fieldnames:
            raise ValueError("File caption phải có cột 'image'.")

        if "caption" not in reader.fieldnames:
            raise ValueError("File caption phải có cột 'caption'.")

        for line_number, row in enumerate(reader, start=2):
            image_name = row["image"].strip()
            caption = row["caption"].strip()

            if not image_name:
                print(f"Bỏ qua dòng {line_number}: thiếu tên ảnh.")
                continue

            if not caption:
                print(f"Bỏ qua dòng {line_number}: thiếu caption.")
                continue

            image_path = image_dir / image_name

            if not image_path.is_file():
                print(f"Bỏ qua dòng {line_number}: "f"không tìm thấy ảnh {image_path}")
                continue

            # Nếu ảnh chưa tồn tại trong mapping thì khởi tạo record.
            if image_name not in image_to_captions:
                image_to_captions[image_name] = {
                    "image_path": image_path.as_posix(),
                    "captions": [],
                }

            # Thêm caption vào danh sách của ảnh.
            image_to_captions[image_name]["captions"].append(caption)

    return image_to_captions


def save_mapping(mapping: dict[str, dict], output_file: str | Path) -> None:
    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(mode="w", encoding="utf-8") as file:
        json.dump(
            mapping,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_dataset_stats(mapping: dict[str, dict]) -> None:
    total_images = len(mapping)

    total_captions = sum(
        len(image_record["captions"])
        for image_record in mapping.values()
    )

    caption_counts = [
        len(image_record["captions"])
        for image_record in mapping.values()
    ]

    minimum_captions = (
        min(caption_counts)
        if caption_counts
        else 0
    )

    maximum_captions = (
        max(caption_counts)
        if caption_counts
        else 0
    )

    average_captions = (
        total_captions / total_images
        if total_images > 0
        else 0.0
    )

    print(f"Tổng số ảnh: {total_images}")
    print(f"Tổng số caption: {total_captions}")
    print(f"Caption ít nhất mỗi ảnh: {minimum_captions}")
    print(f"Caption nhiều nhất mỗi ảnh: {maximum_captions}")
    print(f"Caption trung bình mỗi ảnh: "f"{average_captions:.2f}")


def prepare_dataset() -> None:
    mapping = build_image_caption_mapping(
        image_dir=IMAGE_DIR,
        caption_file=CAPTION_FILE,
    )

    save_mapping(
        mapping=mapping,
        output_file=OUTPUT_FILE,
    )

    print_dataset_stats(mapping)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Caption file: {CAPTION_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    print(f"Đã lưu mapping tại: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_dataset()

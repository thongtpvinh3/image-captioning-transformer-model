from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from config import (
    IMAGE_TO_CAPTIONS_FILE,
    TEST_DATA_FILE,
    TRAIN_DATA_MAPPING,
    VALIDATION_DATA_FILE,
)

JsonMapping = dict[str, dict[str, Any]]


def load_json_mapping(input_file: str | Path) -> JsonMapping:
    input_file = Path(input_file)

    if not input_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {input_file}")

    with input_file.open(mode="r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, dict):
        raise ValueError("image_to_captions.json phải chứa một JSON object.")

    if not mapping:
        raise ValueError("image_to_captions.json không chứa dữ liệu.")

    return mapping


def validate_split_ratios(
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
) -> None:
    """
    Kiểm tra tỉ lệ train/validation/test.
    """

    ratios = {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
    }

    for ratio_name, ratio_value in ratios.items():
        if ratio_value < 0 or ratio_value > 1:
            raise ValueError(
                f"{ratio_name} phải nằm trong [0, 1], "
                f"nhận được {ratio_value}."
            )

    total_ratio = (train_ratio + validation_ratio + test_ratio)

    if not math.isclose(
            total_ratio,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
    ):
        raise ValueError(
            "Tổng train_ratio, validation_ratio và "
            f"test_ratio phải bằng 1.0, hiện tại là {total_ratio}."
        )


def validate_image_record(image_name: str, image_record: dict[str, Any]) -> None:
    """
    Kiểm tra mỗi record ảnh có image_path và captions hợp lệ.
    """

    if not isinstance(image_record, dict):
        raise ValueError(f"Record của {image_name} phải là dictionary.")

    image_path = image_record.get("image_path")
    captions = image_record.get("captions")

    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"{image_name} không có image_path hợp lệ.")

    if not isinstance(captions, list):
        raise ValueError(f"{image_name}: captions phải là list.")

    if len(captions) == 0:
        raise ValueError(f"{image_name} không có caption.")


def create_subset(
        mapping: JsonMapping,
        image_names: list[str],
) -> JsonMapping:
    # Tạo mapping con từ danh sách tên ảnh. Toàn bộ caption của một ảnh được giữ nguyên.
    return {
        image_name: mapping[image_name]
        for image_name in image_names
    }


def split_dataset_by_image(
        mapping: JsonMapping,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
) -> tuple[JsonMapping, JsonMapping, JsonMapping]:
    """
    Chia dataset theo image name.

    Một ảnh và toàn bộ caption của nó chỉ xuất hiện trong đúng một tập: train, validation hoặc test.
    """

    validate_split_ratios(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    for image_name, image_record in mapping.items():
        validate_image_record(
            image_name=image_name,
            image_record=image_record,
        )

    image_names = list(mapping.keys())

    random_generator = random.Random(seed)
    random_generator.shuffle(image_names)

    total_images = len(image_names)

    train_size = int(total_images * train_ratio)
    validation_size = int(
        total_images * validation_ratio
    )

    # Phần còn lại được đưa hết vào test để không mất ảnh do phép làm tròn số nguyên.
    test_size = (total_images - train_size - validation_size)

    train_end = train_size
    validation_end = train_size + validation_size

    train_image_names = image_names[:train_end]

    validation_image_names = image_names[train_end:validation_end]

    test_image_names = image_names[validation_end:]

    if len(test_image_names) != test_size:
        raise RuntimeError("Số lượng ảnh test không khớp với tính toán.")

    train_mapping = create_subset(mapping=mapping, image_names=train_image_names)

    validation_mapping = create_subset(mapping=mapping, image_names=validation_image_names)

    test_mapping = create_subset(mapping=mapping, image_names=test_image_names)

    validate_split_result(
        original_mapping=mapping,
        train_mapping=train_mapping,
        validation_mapping=validation_mapping,
        test_mapping=test_mapping,
    )

    return train_mapping, validation_mapping, test_mapping


def validate_split_result(
        original_mapping: JsonMapping,
        train_mapping: JsonMapping,
        validation_mapping: JsonMapping,
        test_mapping: JsonMapping,
) -> None:
    """
    Kiểm tra:

    1. Không có ảnh bị mất.
    2. Không có ảnh xuất hiện ở nhiều tập.
    3. Tổng số ảnh sau chia bằng số ảnh ban đầu.
    """

    original_images = set(original_mapping.keys())
    train_images = set(train_mapping.keys())
    validation_images = set(validation_mapping.keys())
    test_images = set(test_mapping.keys())

    train_validation_overlap = (train_images & validation_images)

    train_test_overlap = (train_images & test_images)

    validation_test_overlap = (validation_images & test_images)

    if train_validation_overlap:
        raise ValueError(
            "Có ảnh xuất hiện đồng thời trong train và "
            f"validation: {train_validation_overlap}"
        )

    if train_test_overlap:
        raise ValueError(
            "Có ảnh xuất hiện đồng thời trong train và "
            f"test: {train_test_overlap}"
        )

    if validation_test_overlap:
        raise ValueError(
            "Có ảnh xuất hiện đồng thời trong validation và "
            f"test: {validation_test_overlap}"
        )

    split_images = (train_images | validation_images | test_images)

    missing_images = original_images - split_images
    unknown_images = split_images - original_images

    if missing_images:
        raise ValueError(f"Có ảnh bị mất sau khi chia: {missing_images}")

    if unknown_images:
        raise ValueError(
            "Xuất hiện ảnh không có trong dữ liệu ban đầu: "
            f"{unknown_images}"
        )

    if len(split_images) != len(original_images):
        raise ValueError("Tổng số ảnh sau khi chia không bằng dữ liệu gốc.")


def save_json_mapping(mapping: JsonMapping, output_file: str | Path) -> None:
    """
    Lưu mapping thành JSON.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open(mode="w", encoding="utf-8", ) as file:
        json.dump(mapping, file, ensure_ascii=False, indent=2)


def count_captions(mapping: JsonMapping) -> int:
    """
    Đếm tổng số caption trong một mapping.
    """

    return sum(
        len(image_record["captions"])
        for image_record in mapping.values()
    )


def print_split_statistics(
        original_mapping: JsonMapping,
        train_mapping: JsonMapping,
        validation_mapping: JsonMapping,
        test_mapping: JsonMapping,
) -> None:
    """
    In thống kê sau khi chia dataset.
    """

    total_images = len(original_mapping)
    total_captions = count_captions(original_mapping)

    datasets = {
        "Train": train_mapping,
        "Validation": validation_mapping,
        "Test": test_mapping,
    }

    print("\n===== DATASET SPLIT STATISTICS =====")
    print(f"Tổng số ảnh: {total_images}")
    print(f"Tổng số caption: {total_captions}")
    print()

    for dataset_name, dataset_mapping in datasets.items():
        image_count = len(dataset_mapping)
        caption_count = count_captions(dataset_mapping)

        image_percentage = (
            image_count / total_images * 100
            if total_images > 0
            else 0.0
        )

        print(
            f"{dataset_name}: "
            f"{image_count} ảnh "
            f"({image_percentage:.2f}%), "
            f"{caption_count} caption"
        )


def split_dataset() -> None:
    mapping = load_json_mapping(IMAGE_TO_CAPTIONS_FILE)

    train_mapping, validation_mapping, test_mapping = (
        split_dataset_by_image(
            mapping=mapping,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
    )

    save_json_mapping(mapping=train_mapping, output_file=TRAIN_DATA_MAPPING)

    save_json_mapping(mapping=validation_mapping, output_file=VALIDATION_DATA_FILE)

    save_json_mapping(mapping=test_mapping, output_file=TEST_DATA_FILE)

    print_split_statistics(
        original_mapping=mapping,
        train_mapping=train_mapping,
        validation_mapping=validation_mapping,
        test_mapping=test_mapping,
    )

    print()
    print(f"Train file: {TRAIN_DATA_MAPPING}")
    print(f"Validation file: {VALIDATION_DATA_FILE}")
    print(f"Test file: {TEST_DATA_FILE}")


if __name__ == "__main__":
    split_dataset()

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode

from config import Config
from data.image_caption_dataset import IMAGE_TRANSFORM
from data.tokenizer import CaptionTokenizer
from data.vocabulary import Vocabulary
from models._0_image_captioning_transformer import (
    ImageCaptioningTransformer
)
from train import create_or_load_vocabulary

CHECKPOINT_PATH = Config.checkpoint_path

IMAGE_PATH = (
    "datasets/raw/Images/41999070_838089137e.jpg"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_vocabulary() -> Vocabulary:
    return create_or_load_vocabulary()


def load_model(checkpoint_path: str | Path, vocabulary: Vocabulary, device: torch.device) -> ImageCaptioningTransformer:
    model = ImageCaptioningTransformer(
        image_size=Config.image_size,
        patch_size=Config.patch_size,
        vocabulary_size=len(vocabulary),
        max_caption_length=Config.max_caption_length,
        pad_token_id=vocabulary.pad_token_id,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        d_ff=Config.d_ff,
        num_encoder_layers=Config.num_encoder_layers,
        num_decoder_layers=Config.num_decoder_layers,
        dropout=0.1,
        layer_norm_eps=1e-6
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Tắt dropout khi inference.
    model.eval()

    print(f"Đã tải checkpoint epoch {checkpoint['epoch']}")

    print(f"Training loss: {checkpoint['average_loss']:.4f}")

    return model


def preprocess_image(image_path: str | Path, device: torch.device) -> torch.Tensor:
    """
    Áp dụng đúng cách resize và normalize khi train.

    Output: [1, 3, 224, 224]
    """

    image_path = Path(image_path)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image = TF.resize(
            image,
            size=Config.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )

        image = TF.center_crop(
            image,
            output_size=[
                Config.image_size,
                Config.image_size
            ]
        )

        image_tensor = IMAGE_TRANSFORM(image)

    # [3,224,224] -> [1,3,224,224]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(device=device)


@torch.inference_mode()
def generate_caption(
        model: ImageCaptioningTransformer,
        image_tensor: torch.Tensor,
        vocabulary: Vocabulary,
        max_length: int
) -> tuple[str, list[int]]:
    """
    Sinh caption bằng Greedy Decoding.
    """

    device = image_tensor.device

    # Chỉ encode ảnh một lần.
    visual_features = model.encode_images(image_tensor)

    # Caption bắt đầu bằng BOS.
    generated_ids = torch.tensor(
        [[vocabulary.bos_token_id]],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_length - 1):
        decoder_padding_mask = (generated_ids == vocabulary.pad_token_id)

        logits = model.decode_captions(
            decoder_input_ids=generated_ids,
            visual_features=visual_features,
            decoder_padding_mask=decoder_padding_mask
        )

        # Chỉ lấy logits ở vị trí cuối cùng.
        next_token_logits = logits[:, -1, :]

        # Không cho model sinh BOS hoặc PAD.
        next_token_logits[
            :,
            vocabulary.bos_token_id
        ] = float("-inf")

        next_token_logits[
            :,
            vocabulary.pad_token_id
        ] = float("-inf")

        # Greedy decoding: chọn token xác suất cao nhất.
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token_id
            ],
            dim=1
        )

        if next_token_id.item() == vocabulary.eos_token_id:
            break

    generated_token_ids = (generated_ids[0].tolist())

    caption = vocabulary.decode(generated_token_ids, skip_special_tokens=True)

    return caption, generated_token_ids


def main() -> None:
    device = get_device()

    print(f"Device: {device}")

    vocabulary = create_or_load_vocabulary()

    model = load_model(checkpoint_path=CHECKPOINT_PATH, vocabulary=vocabulary, device=device)

    image_tensor = preprocess_image(image_path=IMAGE_PATH, device=device)

    caption, token_ids = generate_caption(model=model, image_tensor=image_tensor, vocabulary=vocabulary,
                                          max_length=Config.max_caption_length)

    print("\n===== INFERENCE RESULT =====")
    print(f"Image: {IMAGE_PATH}")
    print(f"Token IDs: {token_ids}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()

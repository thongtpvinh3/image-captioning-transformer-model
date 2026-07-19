from __future__ import annotations

from json import JSONDecodeError
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config
from data.collate import ImageCaptionCollator
from data.image_caption_dataset import ImageCaptionDataset
from data.prepare_flicrk8k_datasets import prepare_dataset
from data.resize_image import resize_image
from data.split_dataset import split_dataset
from data.tokenizer import CaptionTokenizer
from data.vocabulary import Vocabulary
from models._0_image_captioning_transformer import ImageCaptioningTransformer

RUN_DATA_PREPARATION = False
NUM_WORKERS = 4
RANDOM_SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def prepare_training_data() -> None:
    if not RUN_DATA_PREPARATION:
        return

    prepare_dataset()
    split_dataset()
    resize_image()


def create_or_load_vocabulary() -> Vocabulary:
    tokenizer = CaptionTokenizer()

    vocabulary_path = Path(Config.vocabulary_file)

    if vocabulary_path.exists():
        try:
            vocabulary = Vocabulary.load(vocabulary_path=vocabulary_path, tokenizer=tokenizer)
            print(f"Đã tải vocabulary: "f"{vocabulary_path}")
            return vocabulary

        except (JSONDecodeError, KeyError, TypeError, ValueError) as error:
            print(f"Vocabulary hiện tại không hợp lệ: {error}")
            print("Tiến hành tạo lại vocabulary.")

    vocabulary = Vocabulary(tokenizer=tokenizer)
    vocabulary.build_from_json(
        json_path=Config.train_data_file,
        min_frequency=Config.min_word_frequency
    )

    vocabulary.save(output_path=vocabulary_path)
    print(f"Đã tạo vocabulary mới: {vocabulary_path}")
    return vocabulary


def create_train_dataloader(device: torch.device, vocabulary: Vocabulary) -> DataLoader:
    train_dataset = ImageCaptionDataset(
        json_path=Config.train_data_file,
        vocabulary=vocabulary,
        max_caption_length=Config.max_caption_length
    )

    collator = ImageCaptionCollator(pad_token_id=vocabulary.pad_token_id)

    return DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
        collate_fn=collator
    )


def train_one_epoch(
        model: ImageCaptioningTransformer,
        train_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: AdamW,
        device: torch.device,
        pad_token_id: int,
        epoch: int
) -> float:
    model.train()

    total_loss = 0.0
    total_valid_tokens = 0

    for batch_index, batch in enumerate(train_dataloader, start=1):
        images = batch["images"].to(device=device, non_blocking=True)
        caption_ids = batch["caption_ids"].to(device=device, non_blocking=True)
        caption_padding_mask = (batch["caption_padding_mask"].to(device=device, non_blocking=True))

        # Teacher forcing.
        decoder_input_ids = caption_ids[:, :-1]
        decoder_target_ids = caption_ids[:, 1:]

        decoder_padding_mask = caption_padding_mask[:, :-1]

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            images=images,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask
        )

        vocabulary_size = logits.shape[-1]

        loss = criterion(logits.reshape(-1, vocabulary_size), decoder_target_ids.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=Config.max_grad_norm)
        optimizer.step()

        valid_token_count = (decoder_target_ids != pad_token_id).sum().item()
        total_loss += (loss.item() * valid_token_count)
        total_valid_tokens += valid_token_count

        if batch_index == 1 or batch_index % Config.log_interval == 0 or batch_index == len(train_dataloader):
            average_loss = (total_loss / total_valid_tokens)
            print(
                f"Epoch {epoch:02d} | "
                f"Batch {batch_index:04d}/"
                f"{len(train_dataloader):04d} | "
                f"Loss {loss.item():.4f} | "
                f"Average Loss {average_loss:.4f}"
            )

    return total_loss / total_valid_tokens


def save_checkpoint(
        model: ImageCaptioningTransformer,
        optimizer: AdamW,
        epoch: int,
        average_loss: float,
        vocabulary_size: int
) -> Path:
    checkpoint_directory = Path(Config.checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (checkpoint_directory / f"image_captioning_epoch_{epoch:03d}.pt")
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "average_loss": average_loss,
        "vocabulary_size": vocabulary_size
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def main() -> None:
    # --------------------------------------------------
    # Random seed.
    # --------------------------------------------------
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # --------------------------------------------------
    # Dataset và device.
    # --------------------------------------------------
    prepare_training_data()
    device = get_device()

    print(f"Device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # Vocabulary và DataLoader.
    # --------------------------------------------------
    vocabulary = create_or_load_vocabulary()
    vocabulary_size = len(vocabulary)
    print(f"Vocabulary size: {vocabulary_size}")
    train_dataloader = create_train_dataloader(device=device, vocabulary=vocabulary)
    print(f"Số batch huấn luyện: {len(train_dataloader)}")

    # --------------------------------------------------
    # Model.
    # --------------------------------------------------
    model = ImageCaptioningTransformer(
        image_size=Config.image_size,
        patch_size=Config.patch_size,
        vocabulary_size=vocabulary_size,
        max_caption_length=(
            Config.max_caption_length
        ),
        pad_token_id=vocabulary.pad_token_id,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        d_ff=Config.d_ff,
        num_encoder_layers=Config.num_encoder_layers,
        num_decoder_layers=Config.num_decoder_layers,
        dropout=0.1,
        layer_norm_eps=1e-6
    ).to(device)

    # --------------------------------------------------
    # Loss.
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_token_id)

    # --------------------------------------------------
    # Optimizer.
    # --------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=Config.weight_decay
    )

    # --------------------------------------------------
    # Train epochs.
    # --------------------------------------------------
    print("\n===== BẮT ĐẦU HUẤN LUYỆN =====")

    for epoch in range(1, Config.num_epochs + 1):
        average_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pad_token_id=vocabulary.pad_token_id,
            epoch=epoch
        )

        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            average_loss=average_loss,
            vocabulary_size=vocabulary_size
        )

        print(f"Epoch {epoch:02d} hoàn thành | Average Loss: {average_loss:.4f}")
        print(f"Checkpoint: {checkpoint_path}")

    print("\n===== HUẤN LUYỆN HOÀN THÀNH =====")

    # # --------------------------------------------------
    # # Bước 6: Tính số patch.
    # # --------------------------------------------------
    # patches_per_side = Config.image_size // Config.patch_size
    #
    # num_patches = patches_per_side ** 2
    #
    # # --------------------------------------------------
    # # Bước 7: Khởi tạo Patch Embedding.
    # # --------------------------------------------------
    # patch_embedding = PatchEmbedding(in_channels=3).to(device)
    #
    # # --------------------------------------------------
    # # Bước 8: Khởi tạo Image Positional Embedding.
    # # --------------------------------------------------
    # positional_embedding = PositionalEmbedding(num_patches=num_patches).to(device)
    #
    # # --------------------------------------------------
    # # Bước 9: Khởi tạo Vision Transformer Encoder.
    # # --------------------------------------------------
    # vision_transformer_encoder = VisionTransformerEncoder().to(device)
    #
    # # --------------------------------------------------
    # # Bước 10: Text Embedding.
    # #
    # # Decoder input IDs:
    # # [B, sequence_length]
    # #
    # # Text embeddings:
    # # [B, sequence_length, d_model]
    # # --------------------------------------------------
    # text_embedding = TextEmbedding(
    #     vocabulary_size=vocabulary_size,
    #     d_model=Config.d_model,
    #     max_sequence_length=(
    #         Config.max_caption_length
    #     ),
    #     pad_token_id=vocabulary.pad_token_id,
    #     dropout=0.1
    # ).to(device)
    #
    # # --------------------------------------------------
    # # Bước 9: Text Transformer Decoder.
    # # --------------------------------------------------
    # text_transformer_decoder = TextTransformerDecoder(
    #     d_model=Config.d_model,
    #     num_heads=Config.num_heads,
    #     d_ff=Config.d_ff,
    #     num_layers=Config.num_decoder_layers,
    #     dropout=0.1,
    #     layer_norm_eps=1e-6
    # ).to(device)
    #
    # # --------------------------------------------------
    # # Bước 10: Language Modeling Head.
    # # --------------------------------------------------
    # language_modeling_head = LanguageModelingHead(d_model=Config.d_model, vocabulary_size=vocabulary_size).to(device)
    #
    # # --------------------------------------------------
    # # Bước 11: Cross-Entropy Loss.
    # #
    # # PAD token không tham gia tính loss.
    # # --------------------------------------------------
    # criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_token_id)
    #
    # patch_embedding.train()
    # positional_embedding.train()
    # vision_transformer_encoder.train()
    # text_embedding.train()
    # text_transformer_decoder.train()
    # language_modeling_head.train()
    #
    # # --------------------------------------------------
    # # Bước 11: Lấy một batch.
    # # --------------------------------------------------
    # batch = next(iter(train_dataloader))
    #
    # images = batch["images"].to(device=device, non_blocking=True)
    #
    # caption_ids = batch["caption_ids"].to(device=device, non_blocking=True)
    #
    # caption_padding_mask = (
    #     batch["caption_padding_mask"].to(device=device, non_blocking=True)
    # )
    #
    # captions = batch["captions"]
    # filenames = batch["filenames"]
    #
    # # --------------------------------------------------
    # # Bước 12: Tạo Decoder Input và Target.
    # # --------------------------------------------------
    # decoder_input_ids = caption_ids[:, :-1]
    # decoder_target_ids = caption_ids[:, 1:]
    #
    # decoder_padding_mask = (caption_padding_mask[:, :-1])
    #
    # # --------------------------------------------------
    # # Bước 13: Patch Embedding.
    # # --------------------------------------------------
    # patch_tokens = patch_embedding(images)
    #
    # # --------------------------------------------------
    # # Bước 14: Image Positional Embedding.
    # # --------------------------------------------------
    # image_tokens = positional_embedding(patch_tokens)
    #
    # # --------------------------------------------------
    # # Bước 15: Vision Transformer Encoder.
    # # --------------------------------------------------
    # visual_features = vision_transformer_encoder(image_tokens)
    #
    # # --------------------------------------------------
    # # Bước 16: Text Embedding và Positional Embedding.
    # #
    # # [B, L] -> [B, L, d_model]
    # # --------------------------------------------------
    # text_embeddings = text_embedding(decoder_input_ids)
    #
    # # --------------------------------------------------
    # # Bước 14: Text Transformer Decoder.
    # # --------------------------------------------------
    # decoder_features = text_transformer_decoder(
    #     text_embeddings=text_embeddings,
    #     visual_features=visual_features,
    #     padding_mask=decoder_padding_mask
    # )
    #
    # # --------------------------------------------------
    # # Bước 17: Language Modeling Head.
    # #
    # # [B,L,512] -> [B,L,vocabulary_size]
    # # --------------------------------------------------
    # logits = language_modeling_head(decoder_features)
    #
    # # --------------------------------------------------
    # # Bước 18: Cross-Entropy Loss.
    # #
    # # Logits:
    # # [B,L,V] -> [B*L,V]
    # #
    # # Targets:
    # # [B,L] -> [B*L]
    # # --------------------------------------------------
    # loss = criterion(
    #     logits.reshape(-1, vocabulary_size),
    #     decoder_target_ids.reshape(-1)
    # )
    #
    # # --------------------------------------------------
    # # Kết quả hiện tại.
    # # --------------------------------------------------
    # print("\n===== BATCH =====")
    # print(f"Filename: {filenames[0]}")
    # print(f"Caption: {captions[0]}")
    #
    # print("\n===== IMAGE ENCODER =====")
    # print(f"Images: {images.shape}")
    # print(f"Patch tokens: {patch_tokens.shape}")
    # print(f"Image tokens: {image_tokens.shape}")
    # print(f"Visual features: {visual_features.shape}")
    #
    # print("\n===== TEXT DATA =====")
    # print(f"Caption IDs: {caption_ids.shape}")
    # print(f"Decoder input IDs: {decoder_input_ids.shape}")
    # print(f"Decoder target IDs: {decoder_target_ids.shape}")
    # print(f"Decoder padding mask: {decoder_padding_mask.shape}")
    #
    # print("\n===== TEXT EMBEDDING =====")
    # print(f"Text embeddings: {text_embeddings.shape}")
    #
    # print("\n===== TEXT TRANSFORMER DECODER =====")
    # print(f"Decoder features: "f"{decoder_features.shape}")
    #
    # print("\n===== LANGUAGE MODELING =====")
    # print(f"Logits: {logits.shape}")
    # print(f"Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from audio_project.config import ProjectConfig
from audio_project.data.audio import AudioPreprocessor
from audio_project.data.dataset import AudioManifestDataset, collate_batch
from audio_project.models.onset_frame_model import OnsetFrameModel
from audio_project.training.reporting import EpochMetrics, plot_history, save_history


@dataclass(slots=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class EpochSummary:
    loss: float
    onset_metrics: BinaryMetrics
    frame_metrics: BinaryMetrics


def build_dataloaders(config: ProjectConfig) -> tuple[DataLoader, DataLoader]:
    preprocessor = AudioPreprocessor(config.audio)
    train_dataset = AudioManifestDataset(
        config.data.train_manifest,
        preprocessor,
        config.labels.min_midi,
        config.labels.max_midi,
    )
    valid_dataset = AudioManifestDataset(
        config.data.valid_manifest,
        preprocessor,
        config.labels.min_midi,
        config.labels.max_midi,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return train_loader, valid_loader


def build_model(config: ProjectConfig) -> OnsetFrameModel:
    return OnsetFrameModel(
        audio_config=config.audio,
        num_pitches=config.labels.num_pitches,
        config=config.model,
    )


def focal_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Focal BCE loss that down-weights easy (confident) examples."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    if gamma <= 0.0:
        return bce
    probs = logits.sigmoid()
    pt = torch.where(targets >= 0.5, probs, 1.0 - probs)
    return (1.0 - pt).pow(gamma) * bce


def compute_losses(
    config: ProjectConfig,
    onset_logits: torch.Tensor,
    frame_logits: torch.Tensor,
    onset_targets: torch.Tensor,
    frame_targets: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    mask = frame_mask.unsqueeze(-1).to(onset_logits.dtype)
    onset_positive_weight = _positive_weight(onset_targets, frame_mask)
    frame_positive_weight = _positive_weight(frame_targets, frame_mask)
    gamma = config.training.focal_gamma
    onset_loss = focal_binary_cross_entropy(onset_logits, onset_targets, gamma, onset_positive_weight)
    frame_loss = focal_binary_cross_entropy(frame_logits, frame_targets, gamma, frame_positive_weight)
    total = (
        (
            config.training.onset_loss_weight * onset_loss
            + config.training.frame_loss_weight * frame_loss
        )
        * mask
    ).sum()
    normalizer = mask.sum().clamp_min(1.0) * onset_logits.size(-1)
    return total / normalizer


def _positive_weight(targets: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    mask = frame_mask.unsqueeze(-1).to(targets.dtype)
    positives = (targets * mask).sum()
    negatives = (((1.0 - targets) * mask)).sum()
    ratio = negatives / positives.clamp_min(1.0)
    return ratio.clamp(1.0, 200.0)


def run_epoch(
    config: ProjectConfig,
    model: OnsetFrameModel,
    dataloader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
) -> EpochSummary:
    total_loss = 0.0
    onset_tp = 0
    onset_tn = 0
    onset_fp = 0
    onset_fn = 0
    frame_tp = 0
    frame_tn = 0
    frame_fp = 0
    frame_fn = 0
    training = optimizer is not None
    model.train(mode=training)

    for batch in dataloader:
        features = batch.features.to(device)
        feature_lengths = batch.feature_lengths.to(device)
        onset_targets = batch.onset_targets.to(device)
        frame_targets = batch.frame_targets.to(device)
        frame_mask = batch.frame_mask.to(device)

        outputs = model(features, feature_lengths)
        loss = compute_losses(
            config=config,
            onset_logits=outputs["onset_logits"],
            frame_logits=outputs["frame_logits"],
            onset_targets=onset_targets,
            frame_targets=frame_targets,
            frame_mask=frame_mask,
        )

        if training:
            optimizer.zero_grad()
            loss.backward()
            if config.training.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        batch_mask = frame_mask.unsqueeze(-1)
        onset_predictions = (outputs["onset_logits"].sigmoid() >= 0.5) & batch_mask
        onset_truth = (onset_targets >= 0.5) & batch_mask
        frame_predictions = (outputs["frame_logits"].sigmoid() >= 0.5) & batch_mask
        frame_truth = (frame_targets >= 0.5) & batch_mask

        onset_tp += (onset_predictions & onset_truth).sum().item()
        onset_tn += ((~onset_predictions) & (~onset_truth) & batch_mask).sum().item()
        onset_fp += (onset_predictions & (~onset_truth)).sum().item()
        onset_fn += ((~onset_predictions) & onset_truth).sum().item()
        frame_tp += (frame_predictions & frame_truth).sum().item()
        frame_tn += ((~frame_predictions) & (~frame_truth) & batch_mask).sum().item()
        frame_fp += (frame_predictions & (~frame_truth)).sum().item()
        frame_fn += ((~frame_predictions) & frame_truth).sum().item()

    return EpochSummary(
        loss=total_loss / max(len(dataloader), 1),
        onset_metrics=_metrics_from_counts(onset_tp, onset_tn, onset_fp, onset_fn),
        frame_metrics=_metrics_from_counts(frame_tp, frame_tn, frame_fp, frame_fn),
    )


def _metrics_from_counts(
    true_positive: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
) -> BinaryMetrics:
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def save_checkpoint(model: OnsetFrameModel, output_dir: Path, epoch: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"music_transcription_epoch_{epoch:03d}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def train(config: ProjectConfig) -> None:
    device = torch.device(config.training.device)
    model = build_model(config).to(device)
    train_loader, valid_loader = build_dataloaders(config)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(config.training.epochs, 1),
        eta_min=config.training.min_learning_rate,
    )
    history: list[EpochMetrics] = []

    for epoch in range(1, config.training.epochs + 1):
        train_summary = run_epoch(config, model, train_loader, optimizer, device)
        valid_summary = run_epoch(config, model, valid_loader, None, device)
        checkpoint_path = save_checkpoint(model, config.training.checkpoint_dir, epoch)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_summary.loss,
                valid_loss=valid_summary.loss,
                train_onset_accuracy=train_summary.onset_metrics.accuracy,
                valid_onset_accuracy=valid_summary.onset_metrics.accuracy,
                train_onset_precision=train_summary.onset_metrics.precision,
                valid_onset_precision=valid_summary.onset_metrics.precision,
                train_onset_recall=train_summary.onset_metrics.recall,
                valid_onset_recall=valid_summary.onset_metrics.recall,
                train_onset_f1=train_summary.onset_metrics.f1,
                valid_onset_f1=valid_summary.onset_metrics.f1,
                train_frame_accuracy=train_summary.frame_metrics.accuracy,
                valid_frame_accuracy=valid_summary.frame_metrics.accuracy,
                train_frame_precision=train_summary.frame_metrics.precision,
                valid_frame_precision=valid_summary.frame_metrics.precision,
                train_frame_recall=train_summary.frame_metrics.recall,
                valid_frame_recall=valid_summary.frame_metrics.recall,
                train_frame_f1=train_summary.frame_metrics.f1,
                valid_frame_f1=valid_summary.frame_metrics.f1,
                checkpoint_path=str(checkpoint_path),
            )
        )
        save_history(history, config.training.checkpoint_dir)
        plot_history(history, config.training.checkpoint_dir)
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_summary.loss:.4f} "
            f"valid_loss={valid_summary.loss:.4f} "
            f"train_onset_f1={train_summary.onset_metrics.f1:.4f} "
            f"valid_onset_f1={valid_summary.onset_metrics.f1:.4f} "
            f"train_frame_f1={train_summary.frame_metrics.f1:.4f} "
            f"valid_frame_f1={valid_summary.frame_metrics.f1:.4f} "
            f"checkpoint={checkpoint_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline music transcription model.")
    parser.add_argument("--device", default="cpu", help="Training device, for example cpu or cuda.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--train-manifest", type=Path, help="Path to the training manifest JSONL.")
    parser.add_argument("--valid-manifest", type=Path, help="Path to the validation manifest JSONL.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory for model checkpoints.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate.")
    parser.add_argument("--min-learning-rate", type=float, help="Minimum learning rate for cosine decay.")
    parser.add_argument("--feature-mode", choices=["mel", "hybrid"], help="Feature mode: mel or hybrid.")
    parser.add_argument("--decoder-mode", choices=["baseline", "onset_conditioned"], help="Decoder mode.")
    parser.add_argument("--focal-gamma", type=float, help="Focal loss gamma (0 disables focal loss).")
    parser.add_argument("--gradient-clip-norm", type=float, help="Max gradient norm for clipping (0 disables).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    config.training.device = args.device
    config.training.epochs = args.epochs
    if args.train_manifest is not None:
        config.data.train_manifest = args.train_manifest
    if args.valid_manifest is not None:
        config.data.valid_manifest = args.valid_manifest
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.min_learning_rate is not None:
        config.training.min_learning_rate = args.min_learning_rate
    if args.feature_mode is not None:
        config.model.feature_mode = args.feature_mode
    if args.decoder_mode is not None:
        config.model.decoder_mode = args.decoder_mode
    if args.focal_gamma is not None:
        config.training.focal_gamma = args.focal_gamma
    if args.gradient_clip_norm is not None:
        config.training.gradient_clip_norm = args.gradient_clip_norm
    train(config)


if __name__ == "__main__":
    main()

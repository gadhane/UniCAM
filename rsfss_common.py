from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirst
from pytorch_grad_cam.utils.image import deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from torchvision import datasets
from torchvision.transforms import functional as TF

from common import (
    DEFAULT_DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PET_CLASSES,
    build_dataloader,
    build_eval_transform,
    build_resnet,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_split_dataset,
    seed_everything,
)
from pdCor import Distance_Correlation, New_DC, P_DC

DEFAULT_LAYER_DEPTHS = (2, 3, 4, 5)
LAYER_NAME_BY_DEPTH = {
    2: "layer_2",
    3: "layer_3",
    4: "layer_4",
    5: "layer_5",
}
TARGET_LAYER_NAME_BY_DEPTH = {
    2: "layer4",
    3: "layer3",
    4: "layer2",
    5: "layer1",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    checkpoint: str | Path


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, depth: int, model: torch.nn.Module):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-depth])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(inputs)


def _tensor_to_rgb_image(image_tensor: torch.Tensor) -> np.ndarray:
    return np.array(TF.to_pil_image(image_tensor.cpu()))


def _build_cam_visualization(
    image_tensor: torch.Tensor,
    label: int,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    percentile: int,
    device: torch.device,
) -> torch.Tensor:
    rgb_image = np.float32(_tensor_to_rgb_image(image_tensor)) / 255.0
    input_tensor = preprocess_image(
        rgb_image,
        mean=list(IMAGENET_MEAN),
        std=list(IMAGENET_STD),
    ).to(device)
    targets = [ClassifierOutputSoftmaxTarget(label)]

    with HiResCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

    cam_metric = ROADLeastRelevantFirst(percentile=percentile)
    _, visualizations = cam_metric(
        input_tensor,
        grayscale_cams,
        targets,
        model,
        return_visualization=True,
    )
    visualization = visualizations[0].detach().cpu().numpy().transpose((1, 2, 0))
    visualization = deprocess_image(visualization)
    return TF.to_tensor(Image.fromarray(visualization))


def _normalize_road_tensor_batch(tensor_batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor_batch.dtype).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor_batch.dtype).view(1, -1, 1, 1)
    return (tensor_batch - mean) / std


def _get_target_layer_for_depth(model: torch.nn.Module, depth: int) -> torch.nn.Module:
    layer_name = TARGET_LAYER_NAME_BY_DEPTH[depth]
    return getattr(model, layer_name)[-1]


def build_relevance_tensor_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    model_names: List[str],
    models: List[torch.nn.Module],
    percentile: int,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rendered_by_model: Dict[int, Dict[int, List[torch.Tensor]]] = {
        index: {depth: [] for depth in DEFAULT_LAYER_DEPTHS}
        for index in range(len(models))
    }
    label_values = [int(label.item()) for label in labels]

    for image_tensor, label in zip(images, label_values):
        for model_index, model in enumerate(models):
            for depth in DEFAULT_LAYER_DEPTHS:
                rendered = _build_cam_visualization(
                    image_tensor=image_tensor,
                    label=label,
                    model=model,
                    target_layer=_get_target_layer_for_depth(model, depth),
                    percentile=percentile,
                    device=device,
                )
                rendered_by_model[model_index][depth].append(rendered)

    return {
        model_names[model_index]: {
            LAYER_NAME_BY_DEPTH[depth]: _normalize_road_tensor_batch(
                torch.stack(rendered_by_model[model_index][depth])
            ).to(device)
            for depth in DEFAULT_LAYER_DEPTHS
        }
        for model_index in range(len(models))
    }


def extract_layer_feature(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    features = ResnetFeatureExtractor(depth, model)(inputs)
    return features.view(features.shape[0], -1)


def configure_hf_env() -> Path:
    cache_dir = Path(__file__).resolve().parent / ".hf_cache"
    ensure_dir(cache_dir)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    # Older local protobuf installs can break transformers imports on Windows.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return cache_dir


def build_clip_visual_class_prototypes(
    data_dir: str | Path,
    split: str,
    device: torch.device,
    clip_model_name_or_path: str,
    local_files_only: bool = True,
    batch_size: int = 16,
) -> torch.Tensor:
    configure_hf_env()
    from transformers import CLIPImageProcessor, CLIPModel

    try:
        clip_model = CLIPModel.from_pretrained(
            clip_model_name_or_path,
            local_files_only=local_files_only,
        ).to(device)
        image_processor = CLIPImageProcessor.from_pretrained(
            clip_model_name_or_path,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise RuntimeError(
            "CLIP model files were not found locally. Provide a local CLIP checkpoint path "
            "with --clip-model or pre-cache the model, or rerun with --allow-clip-download "
            "in an environment with network access."
        ) from exc

    clip_model.eval()
    root = Path(data_dir).expanduser().resolve() / split
    dataset = datasets.ImageFolder(root)
    per_class_features: Dict[int, List[torch.Tensor]] = {
        class_index: [] for class_index in range(len(dataset.classes))
    }

    batch_images: List[Image.Image] = []
    batch_labels: List[int] = []
    with torch.no_grad():
        for path, label in dataset.samples:
            image = dataset.loader(path).convert("RGB")
            batch_images.append(image)
            batch_labels.append(label)

            if len(batch_images) < batch_size:
                continue

            inputs = image_processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            image_features = clip_model.get_image_features(pixel_values=pixel_values)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

            for feature, class_label in zip(image_features, batch_labels):
                per_class_features[class_label].append(feature.detach().cpu())

            batch_images = []
            batch_labels = []

        if batch_images:
            inputs = image_processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            image_features = clip_model.get_image_features(pixel_values=pixel_values)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

            for feature, class_label in zip(image_features, batch_labels):
                per_class_features[class_label].append(feature.detach().cpu())

    prototypes = []
    for class_index in range(len(dataset.classes)):
        class_features = per_class_features[class_index]
        if not class_features:
            raise RuntimeError(f"No CLIP prototype features collected for class index {class_index}.")
        prototype = torch.stack(class_features).mean(dim=0)
        prototype = torch.nn.functional.normalize(prototype, dim=0)
        prototypes.append(prototype)

    return torch.stack(prototypes).to(device)


def load_or_build_clip_prototypes(
    data_dir: str | Path,
    split: str,
    device: torch.device,
    clip_model_name_or_path: str,
    local_files_only: bool = True,
    force_rebuild: bool = False,
) -> torch.Tensor:
    safe_name = clip_model_name_or_path.replace("/", "__").replace("\\", "__").replace(":", "_")
    cache_path = (
        Path(__file__).resolve().parent
        / "prototype_cache"
        / f"clip_image_prototypes_{safe_name}_{split}.pt"
    )

    if cache_path.exists() and not force_rebuild:
        return torch.load(cache_path, map_location=device).to(device)

    prototypes = build_clip_visual_class_prototypes(
        data_dir=data_dir,
        split=split,
        device=device,
        clip_model_name_or_path=clip_model_name_or_path,
        local_files_only=local_files_only,
    )
    ensure_dir(cache_path.parent)
    torch.save(prototypes.cpu(), cache_path)
    return prototypes


def _strength_label(score: float) -> str:
    magnitude = abs(score)
    if magnitude >= 0.9:
        return "very strong"
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.15:
        return "weak"
    return "negligible"


def _direction_label(score: float, section_name: str) -> str:
    magnitude = abs(score)
    if magnitude < 0.15:
        return "little measurable relation"

    if section_name == "partial":
        return "aligned residual signal" if score > 0 else "anti-aligned residual signal"

    if section_name == "progression":
        return "aligned progression" if score > 0 else "inverted progression"

    return "aligned geometry" if score > 0 else "anti-aligned geometry"


def _interpret_score(metric_name: str, section_name: str, layer_name: str, score: float) -> str:
    strength = _strength_label(score)
    direction = _direction_label(score, section_name)
    return f"{strength} {direction} at {layer_name} for {metric_name}"


def enrich_metric_section(
    section_name: str,
    section: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    enriched: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for metric_name, values in section.items():
        enriched[metric_name] = {}
        for layer_name, signed_value in values.items():
            enriched[metric_name][layer_name] = {
                "signed": float(signed_value),
                "absolute": float(abs(signed_value)),
                "interpretation": _interpret_score(metric_name, section_name, layer_name, signed_value),
            }
    return enriched


def compute_similarity_metrics(
    model_features: Dict[str, Dict[str, torch.Tensor]],
    teacher_name: str,
    student_names: List[str],
    ground_truth: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    layer_names = list(model_features[teacher_name].keys())

    for student_name in student_names:
        section = {}
        for layer_name in layer_names:
            section[layer_name] = Distance_Correlation(
                model_features[teacher_name][layer_name],
                model_features[student_name][layer_name],
            ).detach()
        metrics[f"teacher_vs_{student_name}"] = section

    for model_name, features in model_features.items():
        section = {}
        for layer_name in layer_names:
            section[layer_name] = New_DC(features[layer_name], ground_truth).detach()
        metrics[f"{model_name}_vs_ground_truth"] = section

    return {
        section_name: {layer_name: float(value.cpu().item()) for layer_name, value in section.items()}
        for section_name, section in metrics.items()
    }


def compute_layer_progression(
    model_features: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for model_name, features in model_features.items():
        layer_names = list(features.keys())
        anchor = features[layer_names[-1]]
        progression = {}
        for layer_name in layer_names[:-1]:
            progression[layer_name] = New_DC(features[layer_name], anchor).detach()
        metrics[f"{model_name}_progression"] = {
            key: float(value.cpu().item()) for key, value in progression.items()
        }
    return metrics


def compute_partial_metrics(
    model_features: Dict[str, Dict[str, torch.Tensor]],
    teacher_name: str,
    student_names: List[str],
    ground_truth: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    layer_names = list(model_features[teacher_name].keys())

    for student_name in student_names:
        teacher_conditioned = {}
        student_conditioned = {}
        for layer_name in layer_names:
            teacher_conditioned[layer_name] = P_DC(
                model_features[teacher_name][layer_name],
                model_features[student_name][layer_name],
                ground_truth,
            ).detach()
            student_conditioned[layer_name] = P_DC(
                model_features[student_name][layer_name],
                model_features[teacher_name][layer_name],
                ground_truth,
            ).detach()
        metrics[f"{teacher_name}_given_{student_name}"] = {
            key: float(value.cpu().item()) for key, value in teacher_conditioned.items()
        }
        metrics[f"{student_name}_given_{teacher_name}"] = {
            key: float(value.cpu().item()) for key, value in student_conditioned.items()
        }

    if len(student_names) >= 2:
        for index, first_name in enumerate(student_names):
            for second_name in student_names[index + 1 :]:
                first_section = {}
                second_section = {}
                for layer_name in layer_names:
                    first_section[layer_name] = P_DC(
                        model_features[first_name][layer_name],
                        model_features[second_name][layer_name],
                        ground_truth,
                    ).detach()
                    second_section[layer_name] = P_DC(
                        model_features[second_name][layer_name],
                        model_features[first_name][layer_name],
                        ground_truth,
                    ).detach()
                metrics[f"{first_name}_given_{second_name}"] = {
                    key: float(value.cpu().item()) for key, value in first_section.items()
                }
                metrics[f"{second_name}_given_{first_name}"] = {
                    key: float(value.cpu().item()) for key, value in second_section.items()
                }

    return metrics


def load_models(
    model_specs: List[ModelSpec],
    checkpoints_dir: Path,
    device: torch.device,
) -> Dict[str, torch.nn.Module]:
    loaded_models: Dict[str, torch.nn.Module] = {}
    for spec in model_specs:
        checkpoint_path = checkpoints_dir / Path(spec.checkpoint)
        model = build_resnet(spec.architecture, pretrained=False)
        model = load_checkpoint(model, checkpoint_path, device)
        model = model.to(device)
        model.eval()
        loaded_models[spec.name] = model
    return loaded_models


def _build_rsfss_report(
    model_specs: List[ModelSpec],
    split: str,
    prototype_split: str,
    batch_size: int,
    percentile: int,
    requested_max_batches: int | None,
    considered_batches: int,
    processed_batches: int,
    skipped_single_class_batches: int,
    skipped_low_count_batches: int,
    min_distinct_classes_per_batch: int,
    min_samples_per_class: int,
    processed_label_counts: Dict[str, int],
    processed_batch_class_counts: List[Dict[str, int]],
    clip_model_name_or_path: str,
    road_refeed_normalized: bool,
    cam_is_layerwise: bool,
    feature_source: str,
    similarity: Dict[str, Dict[str, float]],
    progression: Dict[str, Dict[str, float]],
    partial: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "meta": {
            "split": split,
            "prototype_source": "clip_image_encoder",
            "prototype_split": prototype_split,
            "clip_model_name_or_path": clip_model_name_or_path,
            "batch_size": batch_size,
            "percentile": percentile,
            "requested_max_batches": requested_max_batches,
            "considered_batches": considered_batches,
            "processed_batches": processed_batches,
            "skipped_single_class_batches": skipped_single_class_batches,
            "skipped_low_count_batches": skipped_low_count_batches,
            "min_distinct_classes_per_batch": min_distinct_classes_per_batch,
            "min_samples_per_class": min_samples_per_class,
            "processed_label_counts": processed_label_counts,
            "processed_batch_class_counts": processed_batch_class_counts,
            "road_refeed_normalized": road_refeed_normalized,
            "cam_is_layerwise": cam_is_layerwise,
            "feature_source": feature_source,
            "models": [
                {
                    "name": spec.name,
                    "architecture": spec.architecture,
                    "checkpoint": str(spec.checkpoint),
                }
                for spec in model_specs
            ],
        },
        "similarity": enrich_metric_section("similarity", similarity),
        "progression": enrich_metric_section("progression", progression),
        "partial": enrich_metric_section("partial", partial),
    }


def _label_count_map(labels: torch.Tensor, num_classes: int) -> Dict[int, int]:
    counts = torch.bincount(labels.detach().cpu(), minlength=num_classes)
    return {class_index: int(counts[class_index].item()) for class_index in range(num_classes)}


def _batch_is_valid(
    labels: torch.Tensor,
    num_classes: int,
    min_distinct_classes_per_batch: int,
    min_samples_per_class: int,
) -> tuple[bool, str, Dict[int, int]]:
    count_map = _label_count_map(labels, num_classes)
    present_counts = [count for count in count_map.values() if count > 0]
    distinct_classes = len(present_counts)

    if distinct_classes < min_distinct_classes_per_batch:
        return False, "single_class", count_map

    if min(present_counts) < min_samples_per_class:
        return False, "low_count", count_map

    return True, "ok", count_map


def _format_label_counts(count_map: Dict[int, int]) -> Dict[str, int]:
    formatted = {}
    for class_index, count in count_map.items():
        class_name = PET_CLASSES[class_index] if class_index < len(PET_CLASSES) else str(class_index)
        formatted[class_name] = int(count)
    return formatted


def evaluate_rsfss(
    model_specs: List[ModelSpec],
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str = "test",
    prototype_split: str = "train",
    batch_size: int = 4,
    num_workers: int = 1,
    percentile: int = 50,
    max_batches: int | None = 1,
    seed: int = 5,
    checkpoints_dir: str | Path | None = None,
    clip_model_name_or_path: str = "openai/clip-vit-base-patch32",
    clip_local_files_only: bool = True,
    force_rebuild_prototypes: bool = False,
    min_distinct_classes_per_batch: int = 2,
    min_samples_per_class: int = 2,
) -> Dict[str, Dict[str, float]]:
    if len(model_specs) < 1:
        raise ValueError("At least one model spec is required.")

    seed_everything(seed)
    device = get_device()
    data_dir = Path(data_dir).expanduser().resolve()
    checkpoints_dir = Path(checkpoints_dir).expanduser().resolve() if checkpoints_dir else Path(__file__).resolve().parent / "models"

    dataset = load_split_dataset(data_dir, split, build_eval_transform(normalize=False))
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    models_by_name = load_models(model_specs, checkpoints_dir, device)
    ordered_model_names = [spec.name for spec in model_specs]
    ordered_models = [models_by_name[name] for name in ordered_model_names]

    class_embeddings = load_or_build_clip_prototypes(
        data_dir=data_dir,
        split=prototype_split,
        device=device,
        clip_model_name_or_path=clip_model_name_or_path,
        local_files_only=clip_local_files_only,
        force_rebuild=force_rebuild_prototypes,
    )
    collected_features = {
        name: {LAYER_NAME_BY_DEPTH[depth]: [] for depth in DEFAULT_LAYER_DEPTHS}
        for name in ordered_model_names
    }
    collected_labels: List[torch.Tensor] = []

    num_classes = len(PET_CLASSES)
    considered_batches = 0
    processed_batches = 0
    skipped_single_class_batches = 0
    skipped_low_count_batches = 0
    processed_label_counts = {class_name: 0 for class_name in PET_CLASSES}
    processed_batch_class_counts: List[Dict[str, int]] = []

    for batch_index, (images, labels) in enumerate(dataloader):
        considered_batches += 1
        if max_batches is not None and processed_batches >= max_batches:
            break
        if len(images) < 2:
            continue

        is_valid, reason, count_map = _batch_is_valid(
            labels=labels,
            num_classes=num_classes,
            min_distinct_classes_per_batch=min_distinct_classes_per_batch,
            min_samples_per_class=min_samples_per_class,
        )
        if not is_valid:
            if reason == "single_class":
                skipped_single_class_batches += 1
            else:
                skipped_low_count_batches += 1
            continue

        rendered_by_name = build_relevance_tensor_batch(
            images=images,
            labels=labels,
            model_names=ordered_model_names,
            models=ordered_models,
            percentile=percentile,
            device=device,
        )
        with torch.no_grad():
            for name, model in zip(ordered_model_names, ordered_models):
                for depth in DEFAULT_LAYER_DEPTHS:
                    layer_name = LAYER_NAME_BY_DEPTH[depth]
                    layer_tensor = extract_layer_feature(
                        model,
                        rendered_by_name[name][layer_name],
                        depth,
                    )
                    collected_features[name][layer_name].append(layer_tensor.detach().cpu())
        collected_labels.append(labels.detach().cpu())
        formatted_counts = _format_label_counts(count_map)
        processed_batch_class_counts.append(formatted_counts)
        for class_name, count in formatted_counts.items():
            processed_label_counts[class_name] += count
        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No valid batches were processed. Check the dataset split, batch size, and class coverage."
        )

    merged_features = {
        name: {
            layer_name: torch.cat(layer_values, dim=0).to(device)
            for layer_name, layer_values in per_model.items()
        }
        for name, per_model in collected_features.items()
    }

    processed_labels = torch.cat(collected_labels, dim=0).to(device)
    ground_truth = class_embeddings[processed_labels]

    teacher_name = ordered_model_names[0]
    student_names = ordered_model_names[1:]

    similarity = compute_similarity_metrics(
        merged_features,
        teacher_name=teacher_name,
        student_names=student_names,
        ground_truth=ground_truth,
    )
    progression = compute_layer_progression(merged_features)
    partial = compute_partial_metrics(
        merged_features,
        teacher_name=teacher_name,
        student_names=student_names,
        ground_truth=ground_truth,
    )

    return _build_rsfss_report(
        model_specs=model_specs,
        split=split,
        prototype_split=prototype_split,
        batch_size=batch_size,
        percentile=percentile,
        requested_max_batches=max_batches,
        considered_batches=considered_batches,
        processed_batches=processed_batches,
        skipped_single_class_batches=skipped_single_class_batches,
        skipped_low_count_batches=skipped_low_count_batches,
        min_distinct_classes_per_batch=min_distinct_classes_per_batch,
        min_samples_per_class=min_samples_per_class,
        processed_label_counts=processed_label_counts,
        processed_batch_class_counts=processed_batch_class_counts,
        clip_model_name_or_path=clip_model_name_or_path,
        road_refeed_normalized=True,
        cam_is_layerwise=True,
        feature_source="cam_road_refeed",
        similarity=similarity,
        progression=progression,
        partial=partial,
    )


def evaluate_rsfss_raw(
    model_specs: List[ModelSpec],
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str = "test",
    prototype_split: str = "train",
    batch_size: int = 4,
    num_workers: int = 1,
    max_batches: int | None = 1,
    seed: int = 5,
    checkpoints_dir: str | Path | None = None,
    clip_model_name_or_path: str = "openai/clip-vit-base-patch32",
    clip_local_files_only: bool = True,
    force_rebuild_prototypes: bool = False,
    min_distinct_classes_per_batch: int = 2,
    min_samples_per_class: int = 2,
) -> Dict[str, Any]:
    if len(model_specs) < 1:
        raise ValueError("At least one model spec is required.")

    seed_everything(seed)
    device = get_device()
    data_dir = Path(data_dir).expanduser().resolve()
    checkpoints_dir = Path(checkpoints_dir).expanduser().resolve() if checkpoints_dir else Path(__file__).resolve().parent / "models"

    dataset = load_split_dataset(data_dir, split, build_eval_transform(normalize=True))
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    models_by_name = load_models(model_specs, checkpoints_dir, device)
    ordered_model_names = [spec.name for spec in model_specs]
    ordered_models = [models_by_name[name] for name in ordered_model_names]

    class_embeddings = load_or_build_clip_prototypes(
        data_dir=data_dir,
        split=prototype_split,
        device=device,
        clip_model_name_or_path=clip_model_name_or_path,
        local_files_only=clip_local_files_only,
        force_rebuild=force_rebuild_prototypes,
    )
    collected_features = {
        name: {LAYER_NAME_BY_DEPTH[depth]: [] for depth in DEFAULT_LAYER_DEPTHS}
        for name in ordered_model_names
    }
    collected_labels: List[torch.Tensor] = []

    num_classes = len(PET_CLASSES)
    considered_batches = 0
    processed_batches = 0
    skipped_single_class_batches = 0
    skipped_low_count_batches = 0
    processed_label_counts = {class_name: 0 for class_name in PET_CLASSES}
    processed_batch_class_counts: List[Dict[str, int]] = []

    for batch_index, (images, labels) in enumerate(dataloader):
        considered_batches += 1
        if max_batches is not None and processed_batches >= max_batches:
            break
        if len(images) < 2:
            continue

        is_valid, reason, count_map = _batch_is_valid(
            labels=labels,
            num_classes=num_classes,
            min_distinct_classes_per_batch=min_distinct_classes_per_batch,
            min_samples_per_class=min_samples_per_class,
        )
        if not is_valid:
            if reason == "single_class":
                skipped_single_class_batches += 1
            else:
                skipped_low_count_batches += 1
            continue

        input_tensor = images.to(device)
        with torch.no_grad():
            for name, model in zip(ordered_model_names, ordered_models):
                for depth in DEFAULT_LAYER_DEPTHS:
                    layer_name = LAYER_NAME_BY_DEPTH[depth]
                    layer_tensor = extract_layer_feature(model, input_tensor, depth)
                    collected_features[name][layer_name].append(layer_tensor.detach().cpu())

        collected_labels.append(labels.detach().cpu())
        formatted_counts = _format_label_counts(count_map)
        processed_batch_class_counts.append(formatted_counts)
        for class_name, count in formatted_counts.items():
            processed_label_counts[class_name] += count
        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No valid batches were processed. Check the dataset split, batch size, and class coverage."
        )

    merged_features = {
        name: {
            layer_name: torch.cat(layer_values, dim=0).to(device)
            for layer_name, layer_values in per_model.items()
        }
        for name, per_model in collected_features.items()
    }

    processed_labels = torch.cat(collected_labels, dim=0).to(device)
    ground_truth = class_embeddings[processed_labels]

    teacher_name = ordered_model_names[0]
    student_names = ordered_model_names[1:]

    similarity = compute_similarity_metrics(
        merged_features,
        teacher_name=teacher_name,
        student_names=student_names,
        ground_truth=ground_truth,
    )
    progression = compute_layer_progression(merged_features)
    partial = compute_partial_metrics(
        merged_features,
        teacher_name=teacher_name,
        student_names=student_names,
        ground_truth=ground_truth,
    )

    return _build_rsfss_report(
        model_specs=model_specs,
        split=split,
        prototype_split=prototype_split,
        batch_size=batch_size,
        percentile=0,
        requested_max_batches=max_batches,
        considered_batches=considered_batches,
        processed_batches=processed_batches,
        skipped_single_class_batches=skipped_single_class_batches,
        skipped_low_count_batches=skipped_low_count_batches,
        min_distinct_classes_per_batch=min_distinct_classes_per_batch,
        min_samples_per_class=min_samples_per_class,
        processed_label_counts=processed_label_counts,
        processed_batch_class_counts=processed_batch_class_counts,
        clip_model_name_or_path=clip_model_name_or_path,
        road_refeed_normalized=False,
        cam_is_layerwise=False,
        feature_source="raw_model_features",
        similarity=similarity,
        progression=progression,
        partial=partial,
    )


def write_report(report: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def format_report_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    meta = report.get("meta", {})
    lines.append("RS/FSS report")
    lines.append("================")
    lines.append(
        f"split={meta.get('split')} batch_size={meta.get('batch_size')} percentile={meta.get('percentile')} "
        f"processed_batches={meta.get('processed_batches')} considered_batches={meta.get('considered_batches')}"
    )
    lines.append(
        f"feature_source={meta.get('feature_source')} prototype_source={meta.get('prototype_source')} "
        f"prototype_split={meta.get('prototype_split')}"
    )
    lines.append(
        f"min_distinct_classes_per_batch={meta.get('min_distinct_classes_per_batch')} "
        f"min_samples_per_class={meta.get('min_samples_per_class')}"
    )
    lines.append(
        f"skipped_single_class_batches={meta.get('skipped_single_class_batches')} "
        f"skipped_low_count_batches={meta.get('skipped_low_count_batches')}"
    )
    lines.append(f"processed_label_counts={meta.get('processed_label_counts')}")

    for section_name in ("similarity", "progression", "partial"):
        lines.append("")
        lines.append(section_name.upper())
        lines.append("-" * len(section_name))
        section = report.get(section_name, {})
        for metric_name, values in section.items():
            lines.append(metric_name)
            for layer_name, value_info in values.items():
                lines.append(
                    f"  {layer_name}: signed={value_info['signed']:.6f} "
                    f"absolute={value_info['absolute']:.6f}"
                )
                lines.append(f"    {value_info['interpretation']}")
    lines.append("")
    return "\n".join(lines)


def write_text_report(report: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    ensure_dir(output_path.parent)
    output_path.write_text(format_report_text(report), encoding="utf-8")
    return output_path

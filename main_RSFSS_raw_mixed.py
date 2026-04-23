import argparse
from pathlib import Path

import torch

from common import DEFAULT_DATA_DIR, build_dataloader, build_eval_transform, get_device, load_split_dataset, seed_everything
from main_UniCAM_mixed import SUPPORTED_IMAGENET_MODELS, build_feature_extractor, build_imagenet_model
from rsfss_common import (
    DEFAULT_LAYER_DEPTHS,
    LAYER_NAME_BY_DEPTH,
    ModelSpec,
    _batch_is_valid,
    _build_rsfss_report,
    _format_label_counts,
    compute_layer_progression,
    compute_partial_metrics,
    compute_similarity_metrics,
    load_or_build_clip_prototypes,
    write_report,
    write_text_report,
)
from rsfss_table import write_fss_summary_csv, write_rsfss_summary_csv, write_rsfss_summary_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RS/FSS on raw features for mixed ImageNet-pretrained resnet/vgg teacher and student models."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--prototype-split", default="train")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--min-distinct-classes-per-batch", type=int, default=2)
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--teacher-arch",
        default="resnet152",
        choices=SUPPORTED_IMAGENET_MODELS,
        help="Teacher backbone loaded with ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--student-archs",
        nargs="+",
        default=["vgg11", "vgg13", "vgg16"],
        choices=SUPPORTED_IMAGENET_MODELS,
        help="Student backbones loaded with ImageNet-pretrained weights.",
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--allow-clip-download", action="store_true")
    parser.add_argument("--rebuild-prototypes", action="store_true")
    parser.add_argument(
        "--json-output",
        default=str(Path(__file__).resolve().parent / "distance" / "mixedModelsPetImages" / "RSFSS_raw_mixed.json"),
    )
    parser.add_argument(
        "--text-output",
        default=str(Path(__file__).resolve().parent / "distance" / "mixedModelsPetImages" / "RSFSS_raw_mixed.txt"),
    )
    parser.add_argument(
        "--table-output",
        default=str(Path(__file__).resolve().parent / "distance" / "mixedModelsPetImages" / "RSFSS_raw_mixed_table.md"),
    )
    parser.add_argument(
        "--table-csv-output",
        default=str(Path(__file__).resolve().parent / "distance" / "mixedModelsPetImages" / "RSFSS_raw_mixed_table.csv"),
    )
    parser.add_argument(
        "--fss-table-csv-output",
        default=str(Path(__file__).resolve().parent / "distance" / "mixedModelsPetImages" / "RSFSS_raw_mixed_fss_table.csv"),
    )
    return parser.parse_args()


def _extract_raw_features(models_by_name, ordered_model_names, input_tensor):
    collected = {name: {LAYER_NAME_BY_DEPTH[depth]: [] for depth in DEFAULT_LAYER_DEPTHS} for name in ordered_model_names}
    with torch.no_grad():
        for name in ordered_model_names:
            arch = name.split("_", 1)[1]
            model = models_by_name[name]
            for depth in DEFAULT_LAYER_DEPTHS:
                layer_name = LAYER_NAME_BY_DEPTH[depth]
                extractor = build_feature_extractor(model, arch, depth)
                features = extractor(input_tensor)
                collected[name][layer_name].append(features.reshape(features.shape[0], -1).detach().cpu())
    return collected


def evaluate_rsfss_raw_mixed(args):
    seed_everything(args.seed)
    device = get_device()
    data_dir = Path(args.data_dir).expanduser().resolve()

    dataset = load_split_dataset(data_dir, args.split, build_eval_transform(normalize=True))
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    models_by_name = {f"teacher_{args.teacher_arch}": build_imagenet_model(args.teacher_arch).to(device)}
    for student_arch in args.student_archs:
        key = f"student_{student_arch}"
        if key not in models_by_name:
            models_by_name[key] = build_imagenet_model(student_arch).to(device)

    ordered_model_names = list(models_by_name.keys())
    class_embeddings = load_or_build_clip_prototypes(
        data_dir=data_dir,
        split=args.prototype_split,
        device=device,
        clip_model_name_or_path=args.clip_model,
        local_files_only=not args.allow_clip_download,
        force_rebuild=args.rebuild_prototypes,
    )

    collected_features = {
        name: {LAYER_NAME_BY_DEPTH[depth]: [] for depth in DEFAULT_LAYER_DEPTHS}
        for name in ordered_model_names
    }
    collected_labels = []
    considered_batches = 0
    processed_batches = 0
    skipped_single_class_batches = 0
    skipped_low_count_batches = 0
    processed_label_counts = {"Cat": 0, "Dog": 0}
    processed_batch_class_counts = []

    for images, labels in dataloader:
        considered_batches += 1
        if args.max_batches is not None and processed_batches >= args.max_batches:
            break
        if len(images) < 2:
            continue

        is_valid, reason, count_map = _batch_is_valid(
            labels=labels,
            num_classes=2,
            min_distinct_classes_per_batch=args.min_distinct_classes_per_batch,
            min_samples_per_class=args.min_samples_per_class,
        )
        if not is_valid:
            if reason == "single_class":
                skipped_single_class_batches += 1
            else:
                skipped_low_count_batches += 1
            continue

        batch_features = _extract_raw_features(models_by_name, ordered_model_names, images.to(device))
        for name in ordered_model_names:
            for layer_name, tensors in batch_features[name].items():
                collected_features[name][layer_name].extend(tensors)

        collected_labels.append(labels.detach().cpu())
        formatted_counts = _format_label_counts(count_map)
        processed_batch_class_counts.append(formatted_counts)
        for class_name, count in formatted_counts.items():
            processed_label_counts[class_name] += count
        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("No valid batches were processed. Check the dataset split, batch size, and class coverage.")

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

    model_specs = [ModelSpec(name=name, architecture=name.split("_", 1)[1], checkpoint="imagenet_pretrained") for name in ordered_model_names]
    return _build_rsfss_report(
        model_specs=model_specs,
        split=args.split,
        prototype_split=args.prototype_split,
        batch_size=args.batch_size,
        percentile=0,
        requested_max_batches=args.max_batches,
        considered_batches=considered_batches,
        processed_batches=processed_batches,
        skipped_single_class_batches=skipped_single_class_batches,
        skipped_low_count_batches=skipped_low_count_batches,
        min_distinct_classes_per_batch=args.min_distinct_classes_per_batch,
        min_samples_per_class=args.min_samples_per_class,
        processed_label_counts=processed_label_counts,
        processed_batch_class_counts=processed_batch_class_counts,
        clip_model_name_or_path=args.clip_model,
        road_refeed_normalized=False,
        cam_is_layerwise=False,
        feature_source="raw_model_features_mixed",
        similarity=similarity,
        progression=progression,
        partial=partial,
    )


def main():
    args = parse_args()
    report = evaluate_rsfss_raw_mixed(args)
    write_report(report, args.json_output)
    write_text_report(report, args.text_output)
    write_rsfss_summary_table(report, args.table_output)
    write_rsfss_summary_csv(report, args.table_csv_output)
    write_fss_summary_csv(report, args.fss_table_csv_output)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from common import DEFAULT_DATA_DIR
from rsfss_common import ModelSpec, evaluate_rsfss_raw, write_report, write_text_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RS/FSS for PetImages teacher and student models using raw model features."
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
    parser.add_argument("--checkpoints-dir", default=str(Path(__file__).resolve().parent / "models"))
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--allow-clip-download", action="store_true")
    parser.add_argument("--rebuild-prototypes", action="store_true")
    parser.add_argument(
        "--json-output",
        default=str(Path(__file__).resolve().parent / "distance" / "comparingModelsPetImagesNewNoPre" / "CAM1_raw.json"),
    )
    parser.add_argument(
        "--text-output",
        default=str(Path(__file__).resolve().parent / "distance" / "comparingModelsPetImagesNewNoPre" / "CAM1_raw.txt"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = evaluate_rsfss_raw(
        model_specs=[
            ModelSpec(
                name="teacher",
                architecture="resnet50",
                checkpoint="PetImages_ModelResNet50.pt",
            ),
            ModelSpec(
                name="feature_kd",
                architecture="resnet50",
                checkpoint="PetImages_FeatureKdResNet50.pt",
            ),
            ModelSpec(
                name="attention_kd",
                architecture="resnet50",
                checkpoint="student_attKd_ResNet50_ResNet50.pt",
            ),
        ],
        data_dir=args.data_dir,
        split=args.split,
        prototype_split=args.prototype_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        min_distinct_classes_per_batch=args.min_distinct_classes_per_batch,
        min_samples_per_class=args.min_samples_per_class,
        seed=args.seed,
        checkpoints_dir=args.checkpoints_dir,
        clip_model_name_or_path=args.clip_model,
        clip_local_files_only=not args.allow_clip_download,
        force_rebuild_prototypes=args.rebuild_prototypes,
    )
    write_report(report, args.json_output)
    write_text_report(report, args.text_output)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (
    BASE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    build_dataloader,
    build_eval_transform,
    build_resnet,
    ensure_dir,
    get_device,
    get_resnet_target_layer,
    load_checkpoint,
    load_split_dataset,
    seed_everything,
)
from pdCor_CAM import PDC_CAM
from pdCor_model import PDC_Model
from utils import NormalizeLayer

PET_TO_IMAGENET_INDEX = {0: 281, 1: 253}
FEATURE_DEPTH_TO_TARGET_LAYER = {
    1: "layer4",
    2: "layer4",
    3: "layer3",
    4: "layer2",
    5: "layer1",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UniCAM overlays for PetImages models.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "UniCAM"),
        help="Directory used to save generated overlays.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--feature-depth",
        type=int,
        default=1,
        choices=sorted(FEATURE_DEPTH_TO_TARGET_LAYER),
        help="How many trailing ResNet blocks to drop when extracting features for PDC.",
    )
    parser.add_argument(
        "--target-layer",
        default="auto",
        choices=["auto", "layer1", "layer2", "layer3", "layer4"],
    )
    return parser.parse_args()


def load_models(checkpoints_dir: Path, device: torch.device):
    specs = {
        "teacher": ("resnet50", checkpoints_dir / "PetImages_ModelResNet50.pt"),
        "student_vanilla": ("resnet50", checkpoints_dir / "student_vanila_ResNet50_ResNet50.pt"),
        "student_feature": ("resnet50", checkpoints_dir / "student_Featu_ResNet50_ResNet50.pt"),
        "student_attention": ("resnet50", checkpoints_dir / "student_attKd_ResNet50_ResNet50.pt"),
    }

    models_by_name = {}
    for name, (arch, checkpoint_path) in specs.items():
        model = build_resnet(arch, pretrained=False)
        model = load_checkpoint(model, checkpoint_path, device)
        model = model.to(device)
        model.eval()
        models_by_name[name] = model
    return models_by_name


def create_pdc_pairs(models_by_name, normalize_x, normalize_y, feature_depth):
    device = normalize_x.means.device
    teacher = models_by_name["teacher"]
    students = {
        name: model
        for name, model in models_by_name.items()
        if name != "teacher"
    }

    teacher_given_student = {
        name: PDC_Model(teacher, student, normalize_x, normalize_y, feature_depth).to(device).eval()
        for name, student in students.items()
    }
    student_given_teacher = {
        name: PDC_Model(student, teacher, normalize_x, normalize_y, feature_depth).to(device).eval()
        for name, student in students.items()
    }
    return teacher_given_student, student_given_teacher


def load_label_embeddings(device: torch.device):
    embedding_path = BASE_DIR / "ImageNet_Class_Embedding.pt"
    embeddings = torch.load(embedding_path, map_location=device)
    return embeddings.to(device)


def labels_to_targets(labels, embeddings, device):
    imagenet_indices = [PET_TO_IMAGENET_INDEX[int(label)] for label in labels]
    target_index_tensor = torch.tensor(imagenet_indices, device=device)
    return embeddings[target_index_tensor]


def resolve_target_layer_name(feature_depth: int, requested_target_layer: str) -> str:
    derived_target_layer = FEATURE_DEPTH_TO_TARGET_LAYER[feature_depth]
    if requested_target_layer == "auto":
        return derived_target_layer
    if requested_target_layer != derived_target_layer:
        raise ValueError(
            "feature_depth and target-layer must describe the same residual stage. "
            f"Received feature_depth={feature_depth}, which maps to {derived_target_layer}, "
            f"but target-layer={requested_target_layer}."
        )
    return requested_target_layer


def save_overlay(image_rgb, heatmap, output_path: Path):
    overlay_rgb = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    blended = np.clip(0.5 * image_rgb + 0.5 * (overlay_rgb / 255.0), 0.0, 1.0)
    cv2.imwrite(str(output_path), cv2.cvtColor(np.uint8(blended * 255), cv2.COLOR_RGB2BGR))


def save_batch_results(output_dir, batch_index, images, cam_outputs):
    for image_index, image_tensor in enumerate(images):
        image_rgb = image_tensor.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        stem = f"batch{batch_index:03d}_img{image_index:02d}"

        cv2.imwrite(
            str(output_dir / f"{stem}_input.jpg"),
            cv2.cvtColor(np.uint8(image_rgb * 255), cv2.COLOR_RGB2BGR),
        )

        for label, heatmaps in cam_outputs.items():
            # Raw CAM arrays can be re-enabled here if numeric heatmap dumps are needed.
            save_overlay(
                image_rgb,
                heatmaps[image_index],
                output_dir / f"{stem}_{label}.jpg",
            )


def run_cam(model, target_layer, input_tensor, target_embeddings, use_cuda):
    cam = PDC_CAM(
        model=model,
        target_layers=[target_layer],
        use_cuda=use_cuda,
        reshape_transform=None,
    )
    return cam(input_tensor=input_tensor, targets=target_embeddings)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()
    use_cuda = device.type == "cuda"
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve()

    test_dataset = load_split_dataset(
        args.data_dir,
        "test",
        build_eval_transform(normalize=False),
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )

    models_by_name = load_models(checkpoints_dir, device)
    normalize_x = NormalizeLayer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device)
    normalize_y = NormalizeLayer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device)
    teacher_given_student, student_given_teacher = create_pdc_pairs(
        models_by_name,
        normalize_x,
        normalize_y,
        args.feature_depth,
    )
    embeddings = load_label_embeddings(device)
    target_layer_name = resolve_target_layer_name(args.feature_depth, args.target_layer)

    pdc_target_layers = {
        f"teacher_given_{name}": get_resnet_target_layer(pair.modelX, target_layer_name)
        for name, pair in teacher_given_student.items()
    }
    pdc_target_layers.update(
        {
            f"{name}_given_teacher": get_resnet_target_layer(pair.modelX, target_layer_name)
            for name, pair in student_given_teacher.items()
        }
    )

    for batch_index, (images, labels) in enumerate(test_loader):
        if batch_index >= args.max_batches:
            break

        input_tensor = images.to(device)
        target_embeddings = labels_to_targets(labels, embeddings, device)

        cam_outputs = {}
        for name, pair in teacher_given_student.items():
            key = f"teacher_given_{name}"
            cam_outputs[key] = run_cam(
                pair,
                pdc_target_layers[key],
                input_tensor,
                target_embeddings,
                use_cuda,
            )

        for name, pair in student_given_teacher.items():
            key = f"{name}_given_teacher"
            cam_outputs[key] = run_cam(
                pair,
                pdc_target_layers[key],
                input_tensor,
                target_embeddings,
                use_cuda,
            )

        save_batch_results(output_dir, batch_index, images, cam_outputs)


if __name__ == "__main__":
    main()

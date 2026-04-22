import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models

from common import (
    BASE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    build_dataloader,
    build_eval_transform,
    ensure_dir,
    get_device,
    get_resnet_target_layer,
    load_split_dataset,
    seed_everything,
)
from pdCor_CAM import PDC_CAM
from pdCor import P_Distance_Matrix, P_removal
from pdCor_model import PDC_Model
from utils import NormalizeLayer

PET_TO_IMAGENET_INDEX = {0: 281, 1: 253}
RESNET_TARGET_LAYER_BY_DEPTH = {
    1: "layer4",
    2: "layer4",
    3: "layer3",
    4: "layer2",
    5: "layer1",
}
SUPPORTED_IMAGENET_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate UniCAM overlays using mixed ImageNet-pretrained backbones and ImageNet label embeddings.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "UniCAM_mixed_imagenet"),
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
        choices=sorted(RESNET_TARGET_LAYER_BY_DEPTH),
        help="How many trailing ResNet blocks to drop when extracting features for PDC.",
    )
    parser.add_argument(
        "--target-layer",
        default="auto",
        help="Leave as auto for mixed-architecture runs. Manual overrides are only practical for single-family runs.",
    )
    parser.add_argument(
        "--teacher-arch",
        default="resnet152",
        choices=SUPPORTED_IMAGENET_MODELS,
        help="Teacher backbone loaded with ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--student-archs",
        nargs="+",
        default=["resnet18", "resnet34", "resnet50"],
        choices=SUPPORTED_IMAGENET_MODELS,
        help="Student backbones loaded with ImageNet-pretrained weights.",
    )
    return parser.parse_args()


def model_family(model_name: str) -> str:
    if model_name.startswith("resnet"):
        return "resnet"
    if model_name.startswith("vgg"):
        return "vgg"
    raise ValueError(f"Unsupported model family for '{model_name}'.")


def weight_enum_name_for_model(model_name: str) -> str:
    if model_name.startswith("resnet"):
        return f"{model_name.replace('resnet', 'ResNet')}_Weights"
    if model_name.startswith("vgg"):
        return f"{model_name.upper()}_Weights"
    raise ValueError(f"Unsupported model '{model_name}'.")


def build_imagenet_model(model_name: str) -> torch.nn.Module:
    weight_enum_name = weight_enum_name_for_model(model_name)
    if not hasattr(tv_models, model_name) or not hasattr(tv_models, weight_enum_name):
        raise ValueError(f"Unsupported ImageNet-pretrained model '{model_name}'.")

    builder = getattr(tv_models, model_name)
    weights_enum = getattr(tv_models, weight_enum_name)
    model = builder(weights=weights_enum.DEFAULT)
    if model_family(model_name) == "vgg":
        disable_inplace_relu(model)
    model.eval()
    return model


def disable_inplace_relu(module: nn.Module) -> None:
    for child in module.children():
        if isinstance(child, nn.ReLU):
            child.inplace = False
        disable_inplace_relu(child)


def load_models(args, device: torch.device):
    models_by_name = {
        f"teacher_{args.teacher_arch}": build_imagenet_model(args.teacher_arch).to(device),
    }

    for student_arch in args.student_archs:
        model_name = f"student_{student_arch}"
        if model_name in models_by_name:
            continue
        models_by_name[model_name] = build_imagenet_model(student_arch).to(device)

    return models_by_name


def get_vgg_stage_specs(model: nn.Module) -> list[dict[str, int]]:
    stages = []
    last_conv_index = None
    for index, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            last_conv_index = index
        elif isinstance(layer, nn.MaxPool2d):
            if last_conv_index is None:
                raise ValueError("Encountered VGG MaxPool before any convolution.")
            stages.append({"conv_idx": last_conv_index, "pool_idx": index})
    if len(stages) < 5:
        raise ValueError("Expected a 5-stage VGG feature extractor.")
    return stages


def resolve_vgg_stage(model: nn.Module, feature_depth: int) -> dict[str, int]:
    stages = get_vgg_stage_specs(model)
    return stages[-feature_depth]


def resolve_target_layer_spec(model_name: str, feature_depth: int, requested_target_layer: str) -> str:
    if model_family(model_name) == "resnet":
        derived_target_layer = RESNET_TARGET_LAYER_BY_DEPTH[feature_depth]
    else:
        raise ValueError(
            "Auto target resolution for VGG requires the concrete model instance. "
            "Use resolve_target_layer_spec_for_model instead."
        )

    if requested_target_layer == "auto":
        return derived_target_layer
    if requested_target_layer != derived_target_layer:
        raise ValueError(
            "feature_depth and target-layer must describe the same residual stage. "
            f"Received feature_depth={feature_depth}, which maps to {derived_target_layer} "
            f"for {model_name}, but target-layer={requested_target_layer}."
        )
    return requested_target_layer


def resolve_target_layer_spec_for_model(
    model: nn.Module,
    model_name: str,
    feature_depth: int,
    requested_target_layer: str,
) -> str:
    if model_family(model_name) == "resnet":
        return resolve_target_layer_spec(model_name, feature_depth, requested_target_layer)

    stage = resolve_vgg_stage(model, feature_depth)
    derived_target_layer = f"features.{stage['conv_idx']}"
    if requested_target_layer == "auto":
        return derived_target_layer
    if requested_target_layer != derived_target_layer:
        raise ValueError(
            "feature_depth and target-layer must describe the same residual stage. "
            f"Received feature_depth={feature_depth}, which maps to {derived_target_layer} "
            f"for {model_name}, but target-layer={requested_target_layer}."
        )
    return requested_target_layer


def get_target_layer_module(model: nn.Module, target_layer_spec: str) -> nn.Module:
    if target_layer_spec.startswith("layer"):
        layer = getattr(model, target_layer_spec)
        return layer[-1]
    if target_layer_spec.startswith("features."):
        block_index = int(target_layer_spec.split(".", 1)[1])
        return model.features[block_index]
    raise ValueError(f"Unsupported target layer spec '{target_layer_spec}'.")


def build_feature_extractor(model: nn.Module, model_name: str, feature_depth: int) -> nn.Module:
    family = model_family(model_name)
    if family == "resnet":
        return nn.Sequential(*list(model.children())[:-feature_depth])
    if family == "vgg":
        stage = resolve_vgg_stage(model, feature_depth)
        return nn.Sequential(*list(model.features.children())[: stage["pool_idx"] + 1])
    raise ValueError(f"Unsupported model family for '{model_name}'.")


class PDCBackboneModel(nn.Module):
    def __init__(
        self,
        model_x,
        model_y,
        normalize_x,
        normalize_y,
        extractor_x,
        extractor_y,
        target_layer_x,
        target_layer_y,
    ):
        super().__init__()
        self.modelX = model_x
        self.modelY = model_y
        self.normalize_X = normalize_x
        self.normalize_Y = normalize_y
        self.extractor_X = extractor_x
        self.extractor_Y = extractor_y
        self.target_layer_X = target_layer_x
        self.target_layer_Y = target_layer_y

    def forward(self, inputs):
        inputs_x = self.normalize_X(inputs)
        features_x = self.extractor_X(inputs_x)
        features_x = features_x.reshape(features_x.shape[0], -1)

        inputs_y = self.normalize_Y(inputs)
        features_y = self.extractor_Y(inputs_y)
        features_y = features_y.reshape(features_y.shape[0], -1)

        matrix_a = P_Distance_Matrix(features_x)
        matrix_b = P_Distance_Matrix(features_y)
        return P_removal(matrix_a, matrix_b)


def create_pdc_pairs(models_by_name, normalize_x, normalize_y, feature_depth, requested_target_layer):
    device = normalize_x.means.device
    teacher_name = next(name for name in models_by_name if name.startswith("teacher_"))
    teacher = models_by_name[teacher_name]
    students = {name: model for name, model in models_by_name.items() if name != teacher_name}

    teacher_given_student = {}
    student_given_teacher = {}
    for student_name, student in students.items():
        teacher_arch = teacher_name.removeprefix("teacher_")
        student_arch = student_name.removeprefix("student_")
        teacher_target_spec = resolve_target_layer_spec_for_model(
            teacher,
            teacher_arch,
            feature_depth,
            requested_target_layer,
        )
        student_target_spec = resolve_target_layer_spec_for_model(
            student,
            student_arch,
            feature_depth,
            requested_target_layer,
        )

        if model_family(teacher_arch) == "resnet" and model_family(student_arch) == "resnet":
            teacher_pair = PDC_Model(teacher, student, normalize_x, normalize_y, feature_depth).to(device).eval()
            teacher_pair.target_layer_X = get_resnet_target_layer(teacher_pair.modelX, teacher_target_spec)
            teacher_given_student[student_name] = teacher_pair

            student_pair = PDC_Model(student, teacher, normalize_x, normalize_y, feature_depth).to(device).eval()
            student_pair.target_layer_X = get_resnet_target_layer(student_pair.modelX, student_target_spec)
            student_given_teacher[student_name] = student_pair
            continue

        teacher_target_layer = get_target_layer_module(teacher, teacher_target_spec)
        student_target_layer = get_target_layer_module(student, student_target_spec)
        teacher_extractor = build_feature_extractor(teacher, teacher_arch, feature_depth)
        student_extractor = build_feature_extractor(student, student_arch, feature_depth)
        teacher_given_student[student_name] = PDCBackboneModel(
            teacher,
            student,
            normalize_x,
            normalize_y,
            teacher_extractor,
            student_extractor,
            teacher_target_layer,
            student_target_layer,
        ).to(device).eval()
        student_given_teacher[student_name] = PDCBackboneModel(
            student,
            teacher,
            normalize_x,
            normalize_y,
            student_extractor,
            teacher_extractor,
            student_target_layer,
            teacher_target_layer,
        ).to(device).eval()

    return teacher_name, teacher_given_student, student_given_teacher


def load_label_embeddings(device: torch.device):
    embedding_path = BASE_DIR / "ImageNet_Class_Embedding.pt"
    embeddings = torch.load(embedding_path, map_location=device)
    return embeddings.to(device)


def labels_to_targets(labels, embeddings, device):
    imagenet_indices = [PET_TO_IMAGENET_INDEX[int(label)] for label in labels]
    target_index_tensor = torch.tensor(imagenet_indices, device=device)
    return embeddings[target_index_tensor]


def save_overlay(image_rgb, heatmap, output_path: Path):
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)
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

    models_by_name = load_models(args, device)
    teacher_name, teacher_given_student, student_given_teacher = create_pdc_pairs(
        models_by_name,
        NormalizeLayer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device),
        NormalizeLayer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device),
        args.feature_depth,
        args.target_layer,
    )
    embeddings = load_label_embeddings(device)

    pdc_target_layers = {
        f"{teacher_name}_given_{name}": pair.target_layer_X
        for name, pair in teacher_given_student.items()
    }
    pdc_target_layers.update(
        {
            f"{name}_given_{teacher_name}": pair.target_layer_X
            for name, pair in student_given_teacher.items()
        }
    )

    for batch_index, (images, labels) in enumerate(test_loader):
        if batch_index >= args.max_batches:
            break
        if len(images) < 4:
            continue

        input_tensor = images.to(device)
        target_embeddings = labels_to_targets(labels, embeddings, device)

        cam_outputs = {}
        for name, pair in teacher_given_student.items():
            key = f"{teacher_name}_given_{name}"
            cam_outputs[key] = run_cam(
                pair,
                pdc_target_layers[key],
                input_tensor,
                target_embeddings,
                use_cuda,
            )

        for name, pair in student_given_teacher.items():
            key = f"{name}_given_{teacher_name}"
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

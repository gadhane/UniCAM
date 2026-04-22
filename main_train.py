import argparse
import warnings

import shutup
import torch
import torch.optim as optim

from KD_Lib.KD import VanillaKD
from common import (
    DEFAULT_CSV_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_MODELS_DIR,
    build_dataloader,
    build_eval_transform,
    build_resnet,
    build_train_transform,
    ensure_dir,
    load_split_dataset,
    resolve_dir,
    seed_everything,
)

shutup.please()
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher and student models on PetImages.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--teacher-model", default="resnet18")
    parser.add_argument("--student-model", default="resnet18")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOGS_DIR / "VanillaResNet18ResNet18"))
    return parser.parse_args()


def build_loaders(data_dir, batch_size, num_workers):
    pin_memory = torch.cuda.is_available()
    train_dataset = load_split_dataset(data_dir, "train", build_train_transform())
    test_dataset = load_split_dataset(data_dir, "test", build_eval_transform())

    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def main():
    args = parse_args()
    seed_everything(args.seed)

    models_dir = ensure_dir(resolve_dir(args.models_dir, DEFAULT_MODELS_DIR))
    csv_dir = ensure_dir(resolve_dir(args.csv_dir, DEFAULT_CSV_DIR))
    log_dir = ensure_dir(resolve_dir(args.log_dir, DEFAULT_LOGS_DIR / "VanillaResNet18ResNet18"))

    teacher_model = build_resnet(args.teacher_model, pretrained=True)
    student_model = build_resnet(args.student_model, pretrained=True)
    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers)

    teacher_optimizer = optim.SGD(
        teacher_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
    )
    student_optimizer = optim.SGD(
        student_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
    )

    student_model_path = models_dir / f"Vanilla_{args.student_model}.pt"
    teacher_model_path = models_dir / f"teacher_{args.teacher_model}.pt"
    metrics_path = csv_dir / f"Vanilla_{args.student_model}.csv"

    distiller = VanillaKD(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        teacher_optimizer,
        student_optimizer,
        logdir=str(log_dir),
    )
    distiller.train_teacher(
        epochs=args.teacher_epochs,
        plot_losses=False,
        save_model=True,
        save_model_pth=str(teacher_model_path),
        filename=str(metrics_path),
    )
    distiller.train_student(
        epochs=args.student_epochs,
        plot_losses=False,
        save_model=True,
        save_model_pth=str(student_model_path),
        filename=str(metrics_path),
    )


if __name__ == "__main__":
    main()

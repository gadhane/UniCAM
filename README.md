# UniCAM

UniCAM is a gradient-based explainability framework for knowledge distillation. It helps visualize what a teacher transfers to a student, what the student still misses, and how aligned their learned feature spaces are.

This repository contains two main workflows:
- explanation with UniCAM
- quantification with FSS and RS

## Workflow

1. Train or load your teacher and student checkpoints.
2. Generate UniCAM explanations.
3. Quantify transfer with raw-feature RS/FSS.

## Main Scripts

### `main_UniCAM.py`
Use this when your teacher and student models are from the ResNet family and you want explanations on your trained checkpoints.

### `main_UniCAM_mixed.py`
Use this when your teacher and student come from mixed supported CNN families such as `resnet` and `vgg`. This script is the mixed-backbone explainability entrypoint and is the place to extend if you add more model families later.

### `main_RSFSS_raw.py`
Use this for quantification. It computes FSS and RS directly from raw features with no CAM and no perturbation.

### `main_RSFSS_raw_mixed.py`
Use this for quantification when the teacher and students come from mixed supported families such as `resnet` and `vgg`. It computes FSS and RS from raw features and also writes the summary table.

### `main_RSFSS.py`
This is an explanation-conditioned quantification path. It is useful as an additional analysis script, but for the main quantification workflow use `main_RSFSS_raw.py`.

### `main_RS_FSS.py`
Teacher-only quantification baseline.

### `quantifyXAI.py`
Mixed-architecture quantification preset.


### `vanillaKd.py`
Legacy KD training script kept in the repository.

## How To Run

### 1. UniCAM for trained ResNet teacher and students

```powershell
python .\main_UniCAM.py
```

### 2. UniCAM for mixed `resnet` / `vgg` backbones

```powershell
python .\main_UniCAM_mixed.py --teacher-arch resnet152 --student-archs vgg11 vgg13 vgg16 --max-batches 4 --output-dir .\RemoveResults\UniCAM_mixed_test
```

Example with VGG teacher and ResNet students:

```powershell
python .\main_UniCAM_mixed.py --teacher-arch vgg16 --student-archs resnet18 resnet34 resnet50 --max-batches 4 --output-dir .\RemoveResults\UniCAM_mixed_test2
```

### 3. Quantification with raw features

```powershell
python .\main_RSFSS_raw.py --max-batches 10 --batch-size 8
```

### 4. Optional explanation-conditioned quantification

```powershell
python .\main_RSFSS.py --max-batches 10 --batch-size 8
```

### 5. Quantification with raw features for mixed `resnet` / `vgg` backbones

```powershell
python .\main_RSFSS_raw_mixed.py --teacher-arch resnet152 --student-archs vgg11 vgg13 vgg16 --max-batches 10 --batch-size 8
```

Example with VGG teacher and ResNet students:

```powershell
python .\main_RSFSS_raw_mixed.py --teacher-arch vgg16 --student-archs resnet18 resnet34 resnet50 --max-batches 10 --batch-size 8
```

### 6. Teacher-only baseline

```powershell
python .\main_RS_FSS.py --split test --max-batches 10 --batch-size 8
```

### 7. Mixed-architecture quantification preset

```powershell
python .\quantifyXAI.py --max-batches 10 --batch-size 8
```

## Quantification Meaning

### Feature Similarity Score (FSS)
Measures how well teacher and student feature geometries align at the same layer.

### Relevance Score (RS)
Measures whether the remaining or transferred features are aligned with the target semantic structure.

In this repository, we replaced BERT embeddings with CLIP embeddings because the target task is visual and CLIP provides a more suitable visual semantic space.

## How To Compare Models

When teacher and student are compared directly, the comparison is still reasonable and useful. It tells you how much the student aligns with the teacher and what useful teacher information may still remain outside the student.

If your goal is to understand what the student learned specifically because of knowledge distillation, the better comparison is usually:
- `student_kd`
- `base_model`

Here, `base_model` means the same student architecture trained with the same setup but without KD.

This makes the residual interpretation as follows:
- `student_kd_given_base_model` means the features learned because of KD
- `base_model_given_student_kd` means the features the KD student did not retain

This comparison is especially important when teacher and student have different architectures. If teacher and student have the same architecture, comparing the student directly with the teacher is also a reasonable way to understand  the knowledge transfer.

## Use With Your Own Models And Dataset

You can adapt the pipeline to your own teacher and student models and your own dataset.

### For explanation

- If all models are ResNet-family models, use `main_UniCAM.py`.
- If models come from mixed supported CNN families, use `main_UniCAM_mixed.py`.
- If you want to support a new model family, update the mixed script in three places:
  - model loading
  - feature extractor for residual computation
  - target-layer selection for UniCAM hooks

### For quantification

- Use `main_RSFSS_raw.py` as the main quantification script.
- Use `main_RSFSS_raw_mixed.py` when your teacher and students come from mixed supported families such as `resnet` and `vgg`.
- Update the model specifications to your teacher and student checkpoints.
- Point `--data-dir` to your dataset root.
- Make sure the dataset split structure matches the expected layout, such as `train/` and `test/`.
- If you change the label set, rebuild the CLIP prototypes on your dataset.

### Practical steps

1. Replace the default checkpoint names in the script you want to run.
2. Point `--data-dir` to your dataset.
3. Confirm the class mapping used by the dataset loader.
4. If needed, adjust the target layer or stage mapping for your model family.
5. Run UniCAM for explanation and `main_RSFSS_raw.py` or `main_RSFSS_raw_mixed.py` for quantification.

## Outputs

- UniCAM explanation images are written under `RemoveResults/`
- RS/FSS reports are written under `distance/`

## Notes

- `main_UniCAM.py` is the main explainability script used for ResNet teacher-student experiments.
- `main_UniCAM_mixed.py` is the maintained mixed-backbone explainability path for supported families such as `resnet` and `vgg`.
- For quantification, prefer `main_RSFSS_raw.py` for same-family setups and `main_RSFSS_raw_mixed.py` for mixed supported families.

<<<<<<< HEAD
# UniCAM

UniCAM is a visual explainability technique specifically designed to extract knowledge and residual features during knowledge distillation.

## Steps:
  TBC

## Installation

To install UniCAM, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/gadhane/UniCAM.git
    ```

2. Navigate to the project directory:
    ```bash
    cd UniCAM
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch-name`.
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please contact [gadhane](https://github.com/gadhane).

```

Feel free to customize it further based on the specific details and requirements of your project.
=======
# PetImages KD Explainability

This folder contains the maintained PetImages workflow for explaining and quantifying knowledge distillation (KD).

## Maintained Files

- `main_UniCAM.py`
  Real PetImages UniCAM on your trained teacher/student checkpoints.
- `main_UniCAM_mixed.py`
  Mixed-backbone UniCAM sanity-check on ImageNet-pretrained `resnet` and `vgg` families.
- `main_RSFSS.py`
  Explanation-conditioned RS/FSS (`HiResCAM -> ROAD -> re-feed`).
- `main_RSFSS_raw.py`
  Raw-feature RS/FSS baseline with no CAM and no perturbation.
- `main_RS_FSS.py`
  Teacher-only baseline quantification.
- `quantifyXAI.py`
  Mixed-architecture quantification preset.
- `main_train.py`
  Refactored training entrypoint.
- `vanillaKd.py`
  Legacy KD training script kept by request.

## Recommended Order

1. Train or load the teacher and student checkpoints.
2. Generate UniCAM explanations with `main_UniCAM.py`.
3. Quantify transfer with either:
   - `main_RSFSS.py` for explanation-conditioned features (`HiResCAM -> ROAD -> re-feed`)
   - `main_RSFSS_raw.py` for a raw feature baseline (no CAM, no ROAD)

## Metric Meaning

The reports contain four distinct types of scores.

### 1. Teacher vs Student

This is the feature similarity score (FSS-style alignment).

Question:
- Do the teacher and student encode similar feature geometry at the same layer?

Interpretation:
- High positive values mean the student is closely aligned with the teacher.

### 2. Model vs Ground Truth

This is direct relevance.

Question:
- Are the features of a model aligned with the target semantic geometry?

Current target:
- CLIP image prototypes built from real `Cat` and `Dog` images on the `train` split.

Interpretation:
- High positive values mean the feature geometry is aligned with the class prototype geometry.

### 3. Conditioning / Residual Relevance

Examples:
- `student_given_teacher`
- `teacher_given_student`

Question:
- After removing information shared by two models, is the remaining feature geometry still aligned with ground truth?

Interpretation:
- `student_given_teacher`: useful knowledge unique to the student beyond the teacher.
- `teacher_given_student`: useful residual knowledge still present in the teacher but not learned by the student.

### 4. Progression

This is an internal consistency measure inside a single model.

Question:
- How much does an earlier layer align with the final analyzed layer of the same model?

Interpretation:
- High progression means the earlier representation is refined into a coherent final representation.
- It is not directly a teacher-student transfer metric.
- It is not directly a residual relevance metric.

## Why We Use CLIP Prototypes

The original text target used BERT embeddings of `Cat` and `Dog`.

That was replaced with CLIP image prototypes because:
- the task is visual
- the target should live in a visual semantic space
- CLIP image features provide a richer class geometry than one-hot labels

The prototypes are built by:
1. loading a frozen CLIP image encoder
2. encoding real images from a prototype split, typically `train`
3. averaging features per class
4. normalizing the averaged class prototype vectors

## Stability Safeguards

Prototype-based RS/FSS is sensitive to degenerate batches, especially in a two-class setup.

The evaluator now:
- skips batches that do not contain enough distinct classes
- skips batches where a class appears too few times
- records processed label counts in the report metadata
- records how many batches were skipped
- treats `max-batches` as the number of valid processed batches, not merely the first shuffled batches seen

Important defaults:
- `--max-batches 10`
- `--min-distinct-classes-per-batch 2`
- `--min-samples-per-class 2`

## Explanation-Conditioned vs Raw Features

There are two maintained quantification modes.

### Explanation-conditioned

Run:

```powershell
python .\main_RSFSS.py --max-batches 10
```

Pipeline:
1. generate `HiResCAM`
2. apply `ROAD`
3. obtain the perturbed visualization image
4. normalize it
5. feed it back into the model
6. extract features and compute similarity / relevance / partial relevance

This measures the geometry of the explanation-conditioned feature space.

### Raw feature baseline

Run:

```powershell
python .\main_RSFSS_raw.py --max-batches 10
```

Pipeline:
1. use the original normalized image
2. extract features directly from each model
3. compute similarity / relevance / partial relevance

This is the best baseline for checking whether a failure comes from the model representations or from the explanation pipeline.

## Interpreting Negative Values

The current implementation uses a signed centered-distance correlation form.

So:
- positive means aligned geometry
- near zero means little measurable relation
- negative means anti-aligned geometry

Negative values are mathematically valid in this implementation. They do not automatically mean the code is wrong.

## Useful Commands

Real PetImages UniCAM:

```powershell
python .\main_UniCAM.py
```

Mixed-backbone UniCAM sanity check:

```powershell
python .\main_UniCAM_mixed.py --teacher-arch resnet152 --student-archs vgg11 vgg13 vgg16 --max-batches 4 --output-dir .\RemoveResults\UniCAM_mixed_test
```

Explanation-conditioned quantification:

```powershell
python .\main_RSFSS.py --max-batches 10 --batch-size 8
```

Raw-feature baseline:

```powershell
python .\main_RSFSS_raw.py --max-batches 10 --batch-size 8
```

Teacher-only baseline:

```powershell
python .\main_RS_FSS.py --split test --max-batches 10 --batch-size 8
```

Mixed-architecture quantification preset:

```powershell
python .\quantifyXAI.py --max-batches 10 --batch-size 8
```

## Notes

- `main_UniCAM.py` is the main PetImages explanation implementation.
- `main_UniCAM_mixed.py` is a model-family validation script. It is useful for sanity checks, but it is not the primary PetImages experiment.
- If you want RS/FSS from whole raw features only, use `main_RSFSS_raw.py`.
>>>>>>> 60c1059 (update: modified code and fix errors, updated BERT to CLIP embeddings.)

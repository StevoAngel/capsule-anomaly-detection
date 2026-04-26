# Capsule Anomaly Detection - Classical ML Benchmark

This project implements and compares several classical machine learning approaches for anomaly detection on the "Capsule" subset of the MVTec AD dataset.

Objective
- Provide a systematic benchmark of classical (non-deep) methods to evaluate their performance on the Capsule anomaly detection task.

Included approaches
- PCA reconstruction error (PCA-RE)
- Isolation Forest (tuned with threshold calibration)
- One-Class SVM (OC-SVM) with PCA preprocessing
- Additional baseline models and pipelines where applicable

Data
- Dataset: MVTec AD — Capsule category
- Expected layout: `data/raw/capsule/train/good`, `data/raw/capsule/test/*`, and `data/raw/capsule/ground_truth`.

Workflow
1. Preprocess images and extract HOG features.
2. Split evaluation pool (mixed good + anomaly) into stratified validation and test sets.
3. Train models using only defect-free training images.
4. Perform grid search and threshold calibration on the validation split.
5. Evaluate final models on the held-out test split and log metrics/artifacts (e.g., with MLflow).

Notes
- The repository intentionally keeps `data/`, `mlruns/`, and `reports/` out of version control via `.gitignore`.
- If you want to remove files from the repository history (destructive), use a history-rewriting tool (e.g., `git filter-repo` or BFG) with caution.

How to run
- Open the main notebook: `project/capsule_detection_project/anomaly_capsule_detection.ipynb`.
- Ensure the dataset is placed under `data/raw/capsule`.
- Run cells in order to reproduce preprocessing, tuning, and evaluation.

License
- Check the original MVTec dataset license at the MVTec website before redistributing data.


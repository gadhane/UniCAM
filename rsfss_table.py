from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


def _format_metric_value(metric_info: Dict[str, Any]) -> str:
    return f"{metric_info['signed']:.4f}"


def _display_name(model_name: str) -> str:
    return model_name.replace("_", " ").title()


def _build_explanation(
    teacher_residual_info: Dict[str, Any],
    student_residual_info: Dict[str, Any],
) -> str:
    teacher_residual = float(teacher_residual_info["signed"])
    student_residual = float(student_residual_info["signed"])
    delta = student_residual - teacher_residual

    if abs(delta) < 0.05:
        return "Knowledge learned and residual teacher features are similarly relevant at this layer."
    if student_residual > teacher_residual:
        return "The knowledge learned is more relevant than the teacher residual features at this layer."
    return "The residual teacher features are more relevant, so the student failed to learn the more relevant features at this layer."


def build_rsfss_summary_rows(report: Dict[str, Any]) -> List[Dict[str, str]]:
    meta = report.get("meta", {})
    model_specs = meta.get("models", [])
    if not model_specs:
        return []

    teacher_name = model_specs[0]["name"]
    student_names = [model_spec["name"] for model_spec in model_specs[1:]]
    similarity = report.get("similarity", {})
    partial = report.get("partial", {})

    teacher_relevance = similarity.get(f"{teacher_name}_vs_ground_truth", {})
    rows: List[Dict[str, str]] = []

    for student_name in student_names:
        student_relevance = similarity.get(f"{student_name}_vs_ground_truth", {})
        teacher_residual = partial.get(f"{teacher_name}_given_{student_name}", {})
        student_residual = partial.get(f"{student_name}_given_{teacher_name}", {})

        for layer_name, teacher_info in teacher_relevance.items():
            student_info = student_relevance.get(layer_name)
            if student_info is None:
                continue
            teacher_residual_info = teacher_residual.get(layer_name, teacher_info)
            student_residual_info = student_residual.get(layer_name, student_info)

            rows.append(
                {
                    "KD technique": _display_name(student_name),
                    "Layer": layer_name,
                    "Relevance of Teacher Model Features": _format_metric_value(teacher_info),
                    "Relevance of Student Model Features": _format_metric_value(student_info),
                    "Explanation": _build_explanation(teacher_residual_info, student_residual_info),
                }
            )

    return rows


def _markdown_table(rows: List[Dict[str, str]]) -> str:
    headers = [
        "KD technique",
        "Layer",
        "Relevance of Teacher Model Features",
        "Relevance of Student Model Features",
        "Explanation",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def write_rsfss_summary_table(report: Dict[str, Any], output_path: str | Path) -> Path:
    rows = build_rsfss_summary_rows(report)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_table(rows), encoding="utf-8")
    return output_path


def write_rsfss_summary_csv(report: Dict[str, Any], output_path: str | Path) -> Path:
    rows = build_rsfss_summary_rows(report)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "KD technique",
        "Layer",
        "Relevance of Teacher Model Features",
        "Relevance of Student Model Features",
        "Explanation",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path

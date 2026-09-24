#!/usr/bin/env python3
"""Evaluate all saved checkpoints on iSAID and rank by mIoU."""
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def main():
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to exactly one GPU.")
    dataset_root = os.environ.get("DATASET_ROOT", "/mnt/data6/yc/datasets/OVSISBenchDataset")
    os.environ["OVSISBENCH_DATASETS"] = dataset_root
    os.environ["DETECTRON2_DATASETS"] = dataset_root
    root = Path(os.environ.get("TRAIN_OUTPUT", str(project / "output/clip_vitb_384_isaid_attr64_dual_teacher_aff_residual_moe_2gpu_bs4"))).resolve()
    checkpoints = sorted(root.glob("*.pth"))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found: {root}")
    out = root / "eval_all_checkpoints_1gpu" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    import torch

    results, failures = [], []
    print(f"GPU={os.environ['CUDA_VISIBLE_DEVICES']}; checkpoints={len(checkpoints)}; results={out}", flush=True)
    for checkpoint in checkpoints:
        run = out / checkpoint.stem
        run.mkdir()
        print(f"\nEvaluating {checkpoint.name}; log={run / 'console.log'}", flush=True)
        cmd = [
            sys.executable, "-u", "train_net.py",
            "--config-file", "configs/clip_vitb_384_isaid_attr64_dual_teacher_aff_residual_moe.yaml",
            "--num-gpus", "1", "--eval-only",
            "MODEL.WEIGHTS", str(checkpoint), "OUTPUT_DIR", str(run),
            "MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED", "False",
            "MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED", "False",
            "MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.ENABLED", "False",
            "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON", "datasets/iSAID.json",
            "DATASETS.TEST", '("iSAID_all_sem_seg",)',
            "TEST.SLIDING_WINDOW", "False", "TEST.AUG.ENABLED", "False",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES", "[1,1]",
        ]
        try:
            with (run / "console.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            result_file = run / "inference/sem_seg_evaluation.pth"
            try:
                metrics = torch.load(result_file, map_location="cpu", weights_only=False)
            except TypeError:
                metrics = torch.load(result_file, map_location="cpu")
            score = float(metrics["mIoU"])
            if not math.isfinite(score):
                raise ValueError(f"Invalid mIoU: {score}")
            results.append((str(checkpoint), score))
            results.sort(key=lambda row: row[1], reverse=True)
            with (out / "ranking.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["checkpoint", "mIoU"])
                writer.writerows(results)
            (out / "best_checkpoint.txt").write_text(f"{results[0][0]}\nmIoU={results[0][1]:.6f}\n")
            print(f"mIoU={score:.6f}; current best={results[0]}", flush=True)
        except Exception as exc:
            failures.append(checkpoint.name)
            (out / "failures.txt").write_text("\n".join(failures) + "\n")
            print(f"FAILED {checkpoint.name}: {exc}; see {run / 'console.log'}", flush=True)
    print(f"\nResults: {out}", flush=True)
    if results:
        print(f"Best checkpoint: {results[0][0]}\nmIoU={results[0][1]:.6f}", flush=True)
    if failures:
        raise SystemExit(f"Ranking incomplete; failed checkpoints: {failures}")


if __name__ == "__main__":
    main()

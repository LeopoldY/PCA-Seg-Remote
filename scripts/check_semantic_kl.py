"""Numerical contract tests plus optional real-data calibration (zero updates).

Run from the project root with PYTHONPATH=. in the training environment.
"""
import argparse
import json
from pathlib import Path

import torch
from cat_seg.modeling.backbone.remote_clip import semantic_distribution_loss, same_grid_content_loss


def check_math():
    torch.manual_seed(12)
    features = torch.randn(8, 16)
    concepts = torch.randn(7, 16)
    rotation = torch.linalg.qr(torch.randn(16, 16)).Q
    student = features.clone().requires_grad_()
    # Same semantic judgments in independently rotated coordinates must match.
    loss = semantic_distribution_loss(student, features @ rotation,
                                      concepts, concepts @ rotation)
    assert abs(loss.item()) < 2e-6, loss
    teacher = torch.randn(8, 24, requires_grad=True)
    teacher_text = torch.randn(7, 24, requires_grad=True)
    student_text = concepts.clone().requires_grad_()
    loss = semantic_distribution_loss(student, teacher, student_text, teacher_text)
    loss.backward()
    assert torch.isfinite(student.grad).all() and student.grad.norm() > 0
    assert teacher.grad is None and teacher_text.grad is None and student_text.grad is None
    # Explicit definition checks the KL direction and T^2 / region reduction.
    import torch.nn.functional as F
    s = F.normalize(student.detach(), dim=-1) @ F.normalize(concepts, dim=-1).T * 5
    t = F.normalize(teacher.detach(), dim=-1) @ F.normalize(teacher_text.detach(), dim=-1).T * 5
    expected = (t.softmax(-1) * (t.log_softmax(-1) - s.log_softmax(-1))).sum(-1).mean() * 4
    assert torch.allclose(loss.detach(), expected, atol=2e-6)
    for temperature in (0, -1, float("nan")):
        try:
            semantic_distribution_loss(student, teacher, concepts, teacher_text, temperature)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid temperature accepted")
    class GridTeacher(torch.nn.Module):
        def __init__(self, feature):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 32, stride=32)
            self.register_buffer("feature", feature)

        def encode_dense(self, images, keep_shape, mode):
            assert keep_shape and mode == "maskclip"
            return self.feature[:len(images)]

    feature = torch.randn(1, 4, 24, 24)
    grid_teacher = GridTeacher(feature)
    tokens = feature.flatten(2).transpose(1, 2).clone().requires_grad_()
    images = torch.randn(1, 3, 384, 384)
    boxes = torch.tensor([[0, 0, 0, 384, 384], [0, 32, 64, 192, 320]], dtype=torch.float32)
    legacy = same_grid_content_loss(grid_teacher, images, tokens, boxes, (768, 768), 1)
    assert abs(legacy.item()) < 1e-6
    legacy.backward()
    assert tokens.grad is not None and all(p.grad is None for p in grid_teacher.parameters())
    try:
        same_grid_content_loss(grid_teacher, images, tokens, boxes, (384, 384), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Grid mismatch accepted")
    print("PASS: coordinate invariance, distinct dimensions, KL direction/reduction, gradient isolation")
    print("PASS: legacy same-grid cosine and mismatch rejection")


def calibrate(args):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.modeling import build_model
    from detectron2.utils.events import EventStorage
    from detectron2.utils.env import seed_all_rng
    from cat_seg import add_cat_seg_config
    from train_net import Trainer
    seed_all_rng(42)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.DATALOADER.NUM_WORKERS = 0
    model = build_model(cfg)
    model.train()
    loader = iter(Trainer.build_train_loader(cfg))
    rows = []
    with EventStorage():
        for index in range(args.batches):
            data = next(loader)
            model.zero_grad(set_to_none=True)
            losses = model(data)
            assert "loss_remote_clip_semantic" in losses and "loss_remote_clip_content" not in losses
            assert all(torch.isfinite(x) for x in losses.values())
            if index == 0:
                losses["loss_remote_clip_semantic"].backward()
                grads = [p.grad for n, p in model.named_parameters()
                         if "clip_model.visual" in n and p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                assert sum(g.abs().sum().item() for g in grads) > 0
            teacher = model._get_remote_clip_distill_teacher()
            assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            teacher_ids = {id(p) for p in teacher.parameters()}
            assert not teacher_ids.intersection(id(p) for p in model.parameters())
            assert not any("_remote_clip" in k or "_student_clip_text" in k for k in model.state_dict())
            row = {k: v.detach().item() for k, v in losses.items()}
            row["raw_semantic_kl"] = row["loss_remote_clip_semantic"] / cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.LOSS_WEIGHT
            row["raw_context"] = row["loss_rs_dino_context"] / cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.LOSS_WEIGHT
            row["files"] = [x["file_name"] for x in data]
            rows.append(row)
            print(json.dumps({"batch": index, **row}), flush=True)
            del losses
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "files"}
    summary.update(optimizer_steps=0, batches=args.batches, images=args.batches * 4,
                   config=args.config, student_gradient_finite_nonzero=True,
                   teacher_frozen_unregistered=True,
                   suggested_semantic_weight=1e-4 / summary["raw_semantic_kl"],
                   suggested_context_weight=1e-4 / summary["raw_context"])
    Path(args.output).write_text(json.dumps({"summary": summary, "batches": rows}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--output", default="semantic_kl_calibration.json")
    args = parser.parse_args()
    check_math()
    if args.config:
        calibrate(args)

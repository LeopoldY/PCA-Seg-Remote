"""Focused AFF/packed-expert checks; can also run under two-rank torchrun.

python tools/test_aff_residual_moe.py --device cuda
torchrun --standalone --nproc_per_node=2 tools/test_aff_residual_moe.py --distributed
"""

import argparse
import ast
import copy
import importlib.util
import io
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "cat_seg/modeling/transformer"
spec = importlib.util.spec_from_file_location(
    "aff_residual_moe", MODULE_DIR / "aff_residual_moe.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AFFResidualMoE = module.AFFResidualMoE


def legacy_type():
    # Use the actual legacy class, without importing CLIP/Detectron2 for a
    # standalone numerical test. Never maintain a second expert definition.
    tree = ast.parse((MODULE_DIR / "model.py").read_text())
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DualFeatureMoE"
    )
    scope = {"nn": nn, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "legacy", "exec"), scope)
    return scope["DualFeatureMoE"]


def check_packing(device, rank):
    torch.manual_seed(10)
    legacy = legacy_type()(16).to(device)
    packed = AFFResidualMoE(16).to(device).packed_experts
    for expert in legacy.expert_layers:
        expert[1].running_mean.uniform_(-0.3, 0.3)
        expert[1].running_var.uniform_(0.5, 1.5)
        with torch.no_grad():
            expert[1].weight.uniform_(0.5, 1.5)
            expert[1].bias.uniform_(-0.2, 0.2)
    state = packed.state_dict()
    for key in state:
        values = [e.state_dict()[key] for e in legacy.expert_layers]
        state[key] = values[0].clone() if values[0].ndim == 0 else torch.cat(values)
    packed.load_state_dict(state)

    for training in (False, True):
        legacy.train(training)
        packed.train(training)
        legacy.zero_grad(set_to_none=True)
        packed.zero_grad(set_to_none=True)
        torch.manual_seed(20 + rank)
        x = torch.randn(6, 32, 5, 7, device=device, requires_grad=True)
        xp = x.detach().clone().requires_grad_(True)
        expected = torch.cat([e(x) for e in legacy.expert_layers], dim=1)
        actual = packed(xp)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
        upstream = torch.randn_like(actual)
        (expected * upstream).mean().backward()
        (actual * upstream).mean().backward()
        torch.testing.assert_close(xp.grad, x.grad, atol=3e-6, rtol=3e-4)
        for name, parameter in packed.named_parameters():
            expected_grad = torch.cat([
                dict(e.named_parameters())[name].grad for e in legacy.expert_layers
            ])
            torch.testing.assert_close(
                parameter.grad, expected_grad, atol=3e-6, rtol=3e-4
            )
        for key in ("running_mean", "running_var"):
            torch.testing.assert_close(
                getattr(packed[1], key),
                torch.cat([getattr(e[1], key) for e in legacy.expert_layers]),
                atol=3e-6, rtol=3e-4,
            )
        for expert in legacy.expert_layers:
            assert packed[1].num_batches_tracked == expert[1].num_batches_tracked


def check_aff_selection(device):
    model = AFFResidualMoE(16).to(device).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for projection in (model.spatial_proj, model.class_proj):
            projection.weight[:, :, 0, 0].copy_(torch.eye(16, device=device))
        model.res_scale.fill_(0.5)
        spatial = torch.randn(1, 16, 3, 5, 7, device=device)
        semantic = torch.randn_like(spatial)
        for bias, expected in (
            (0, 0.25 * (spatial + semantic)),
            (30, 0.5 * spatial),
            (-30, 0.5 * semantic),
        ):
            model.local_att[-1].bias.fill_(bias)
            torch.testing.assert_close(model(spatial, semantic), expected)


def check_contract(device):
    torch.manual_seed(30)
    model = AFFResidualMoE(16).to(device).eval()
    spatial = torch.randn(2, 16, 3, 5, 7, device=device)
    semantic = torch.randn_like(spatial)
    with torch.no_grad():
        output = model(spatial, semantic)
        assert output.shape == spatial.shape
        permutation = torch.tensor([2, 0, 1], device=device)
        torch.testing.assert_close(
            model(spatial[:, :, permutation], semantic[:, :, permutation]),
            output[:, :, permutation],
        )
        # In evaluation, another class must not affect this class's pooling.
        changed = spatial.clone()
        changed[:, :, 0] += 10
        torch.testing.assert_close(model(changed, semantic)[:, :, 1:], output[:, :, 1:])
        assert model(spatial[:1, :, :1], semantic[:1, :, :1]).shape == (1, 16, 1, 5, 7)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = copy.deepcopy(model)
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
        torch.testing.assert_close(restored(spatial, semantic), output)
    for bad_s, bad_k in ((spatial[0], semantic[0]), (spatial, semantic[:, :, :2])):
        try:
            model(bad_s, bad_k)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted")
    assert sum(p.numel() for p in AFFResidualMoE(128).parameters()) == 333669


def check_training(device, rank, distributed):
    torch.manual_seed(40)
    model = AFFResidualMoE(16).to(device).train()
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(50 + rank)
    modes = (False, True) if device.type == "cuda" else (False,)
    for amp in modes:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            # Match the channel-last storage returned by the aggregators,
            # rather than testing only newly allocated contiguous inputs.
            spatial = torch.randn(2, 3, 5, 7, 16, device=device).permute(
                0, 4, 1, 2, 3
            ).requires_grad_(True)
            semantic = torch.randn_like(spatial, requires_grad=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(spatial, semantic)
                loss = output.float().square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, f"Missing gradient: {name}"
                assert torch.isfinite(parameter.grad).all(), f"Invalid gradient: {name}"
            assert spatial.grad is not None and torch.isfinite(spatial.grad).all()
            assert semantic.grad is not None and torch.isfinite(semantic.grad).all()
            optimizer.step()
    if distributed:
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference, atol=0, rtol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    rank = 0
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    try:
        for name, check in (
            ("legacy expert outputs/gradients/BN equivalence", lambda: check_packing(device, rank)),
            ("AFF midpoint and branch selection", lambda: check_aff_selection(device)),
            ("shape/class independence/checkpoint/parameter count", lambda: check_contract(device)),
            (f"training gradients (AMP={device.type == 'cuda'}, DDP={args.distributed})",
             lambda: check_training(device, rank, args.distributed)),
        ):
            check()
            if rank == 0:
                print(f"PASS: {name}", flush=True)
    finally:
        if args.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

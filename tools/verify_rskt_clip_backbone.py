"""Verify RSKT dense-feature parity and a fresh teacher-free checkpoint load."""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_seg.third_party import clip
from cat_seg import add_cat_seg_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="ViT-B/16")
    parser.add_argument("--resolution", type=int, default=384)
    args = parser.parse_args()
    torch.manual_seed(0)
    # Keep numerical parity independent of TF32 kernel selection on Ampere+.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location("rskt_reference_model", args.reference)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    original = torch.jit.load(args.clip_checkpoint, map_location="cpu")
    reference = reference_module.build_model(
        original.state_dict(), prompt_depth=0, prompt_length=0
    ).float().cuda().eval()
    del original
    student, _ = clip.load(args.model_name, device="cuda", jit=False,
                           checkpoint=args.clip_checkpoint)
    student = student.float().eval()
    with torch.no_grad():
        image = torch.randn(1, 3, args.resolution, args.resolution, device="cuda")
        expected = reference.encode_image(image, dense=True)
        actual = student.encode_image(image, dense=True)
        # RSKT requests attention maps (explicit attention); PCA uses SDPA.
        # Record native-kernel drift, then align kernel selection to verify
        # architecture/weights separately without loosening that comparison.
        parity_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        def sdpa_attention(block, x):
            mask = block.attn_mask
            if mask is not None:
                mask = mask.to(dtype=x.dtype, device=x.device)
            return block.attn(x, x, x, need_weights=False, attn_mask=mask)
        for block in reference.visual.transformer.resblocks:
            block.attention = types.MethodType(sdpa_attention, block)
        aligned = reference.encode_image(image, dense=True)
        torch.testing.assert_close(actual, aligned, atol=1e-4, rtol=1e-4)
        aligned_error = (actual - aligned).abs().max().item()
    del reference, student, expected, actual, aligned
    torch.cuda.empty_cache()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED = False
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.ENABLED = False
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.WEIGHTS = "/nonexistent/rsib.pth"
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.WEIGHTS = "/nonexistent/remoteclip.pt"
    model = build_model(cfg).eval()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert not any("teacher" in key for key in saved["model"])
    model.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        raw = torch.rand(3, 384, 384, device="cuda") * 255
        result = model([{"image": raw, "height": 384, "width": 384}])
    assert torch.isfinite(result[0]["sem_seg"]).all()
    assert model._rs_dino_distill_teacher is None
    assert model._remote_clip_distill_teacher is None
    print(json.dumps({
        "reference_max_abs_error": parity_error,
        "same_attention_kernel_max_abs_error": aligned_error,
        "checkpoint_iteration": saved["iteration"],
        "strict_checkpoint_load": True,
        "teachers_loaded": False,
        "inference_shape": list(result[0]["sem_seg"].shape),
    }))


if __name__ == "__main__":
    main()

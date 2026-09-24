"""Frozen RemoteCLIP image teacher used by DeCLIP-style distillation."""

from pathlib import Path

from torch import nn

import cat_seg.src.open_clip as open_clip


def build_remote_clip_visual(model_name: str, checkpoint_path: str) -> nn.Module:
    """Load an RSKT-Seg RemoteCLIP checkpoint and retain its image tower only."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"RemoteCLIP pretrained weights not found: {checkpoint_path}"
        )

    model = open_clip.create_model(
        model_name,
        pretrained=str(checkpoint),
        precision="fp32",
        device="cpu",
    )
    visual = model.visual
    visual.requires_grad_(False)
    visual.eval()
    return visual


def same_grid_regions(teacher, images, student_tokens, boxes, input_size,
                      teacher_batch_size):
    """Return pooled regions on native, co-registered teacher/student grids.

    Both towers see the same full-image field of view. Only image pixels are
    resized; dense features are never interpolated to hide a grid mismatch.
    """
    import math
    import torch
    from torch.nn import functional as F
    from torchvision.ops import roi_align

    batch, count, channels = student_tokens.shape
    grid = math.isqrt(count)
    if grid * grid != count or images.shape[0] != batch:
        raise ValueError("Student tokens must be a square grid with matching batch.")
    conv = teacher.conv1
    native_grid = tuple(
        (int(size) + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
        for size, pad, dilation, kernel, stride in zip(
            input_size, conv.padding, conv.dilation, conv.kernel_size, conv.stride)
    )
    if native_grid != (grid, grid):
        raise ValueError(
            f"RemoteCLIP native grid {native_grid} != student {(grid, grid)}; "
            "set INPUT_SIZE to student grid times teacher patch stride."
        )
    teacher.eval()
    with torch.no_grad():
        resized = F.interpolate(images.float(), size=input_size,
                                mode="bilinear", align_corners=False)
        maps = [teacher.encode_dense(x, keep_shape=True, mode="maskclip")
                for x in resized.split(teacher_batch_size)]
        teacher_map = torch.cat(maps).float().detach()
    student_map = student_tokens.transpose(1, 2).reshape(batch, channels, grid, grid).float()
    if teacher_map.shape[0] != batch or teacher_map.shape[-2:] != (grid, grid):
        raise ValueError(f"Dense feature shape mismatch: {teacher_map.shape} vs {student_map.shape}")
    # Match encode_dense's per-token normalization before identical ROI pooling.
    student_map = F.normalize(student_map, dim=1)
    teacher_map = F.normalize(teacher_map, dim=1)
    grid_boxes = boxes.to(device=student_map.device, dtype=torch.float32).clone()
    grid_boxes[:, [1, 3]] *= grid / images.shape[-1]
    grid_boxes[:, [2, 4]] *= grid / images.shape[-2]
    def pool(feature):
        return F.normalize(roi_align(feature, grid_boxes, (1, 1),
                                    spatial_scale=1.0, sampling_ratio=-1,
                                    aligned=True).flatten(1), dim=-1)
    student_regions, teacher_regions = pool(student_map), pool(teacher_map)
    return student_regions, teacher_regions


def same_grid_content_loss(teacher, images, student_tokens, boxes, input_size,
                           teacher_batch_size):
    student_regions, teacher_regions = same_grid_regions(
        teacher, images, student_tokens, boxes, input_size, teacher_batch_size)
    if student_regions.shape != teacher_regions.shape:
        raise ValueError("Cosine distillation requires matching feature dimensions.")
    return 1.0 - (student_regions * teacher_regions).sum(-1).mean()


def semantic_distribution_loss(student_regions, teacher_regions, student_text,
                               teacher_text, temperature=2.0, logit_scale=10.0):
    """KL(teacher || student) over identically ordered concepts in own spaces.

    Text anchors and teacher outputs are fixed. Only student image features
    receive gradients; feature dimensions may differ between the two spaces.
    KL is summed over concepts and averaged over regions, with T^2 scaling.
    """
    import math
    from torch.nn import functional as F
    if not (math.isfinite(temperature) and temperature > 0
            and math.isfinite(logit_scale) and logit_scale > 0):
        raise ValueError("Temperature and logit scale must be finite and positive.")
    if any(x.ndim != 2 for x in
           (student_regions, teacher_regions, student_text, teacher_text)):
        raise ValueError("Regions and concept prototypes must be matrices.")
    if (student_text.shape[0] != teacher_text.shape[0] or student_text.shape[0] < 2
            or student_regions.shape[0] != teacher_regions.shape[0]
            or student_regions.shape[0] == 0
            or student_regions.shape[1] != student_text.shape[1]
            or teacher_regions.shape[1] != teacher_text.shape[1]):
        raise ValueError("Concept counts, region counts, or own-space dimensions disagree.")
    student_logits = (F.normalize(student_regions.float(), dim=-1)
                      @ F.normalize(student_text.detach().float(), dim=-1).T)
    teacher_logits = (F.normalize(teacher_regions.detach().float(), dim=-1)
                      @ F.normalize(teacher_text.detach().float(), dim=-1).T)
    return F.kl_div(
        F.log_softmax(student_logits * (logit_scale / temperature), dim=-1),
        F.log_softmax(teacher_logits * (logit_scale / temperature), dim=-1),
        reduction="batchmean", log_target=True) * temperature ** 2


def build_remote_clip_semantic_teacher(model_name, checkpoint_path, concepts,
                                       templates, device):
    """Encode frozen RemoteCLIP anchors once, then retain only its visual tower."""
    import torch
    from torch.nn import functional as F
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)
    model = open_clip.create_model(model_name, pretrained=str(checkpoint_path),
                                   precision="fp32", device=device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(model_name)
    anchors = []
    with torch.no_grad():
        for concept in concepts:
            names = concept.split(", ")
            prompts = [template.format(name) for template in templates for name in names]
            features = F.normalize(model.encode_text(tokenizer(prompts).to(device)).float(), dim=-1)
            features = F.normalize(features.reshape(len(templates), len(names), -1).mean(1), dim=-1)
            anchors.append(F.normalize(features.mean(0), dim=-1))
    return model.visual, torch.stack(anchors).detach()

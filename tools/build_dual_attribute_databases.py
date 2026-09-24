#!/usr/bin/env python3
"""Build attribute-only and Level-2 phrase attribute databases.

The input JSONL files are produced by
``tools/extract_level2_attribute_candidates.py``. Attribute expressions use a
global weighted spherical K-means. Expanded phrases use noun-family-aware
weighted spherical K-means so high-frequency modifiers do not collapse
unrelated nouns into the same centers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--attribute-output", required=True)
    parser.add_argument("--phrase-output", required=True)
    parser.add_argument("--model-name", default="EVA02-CLIP-B-16")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--backend",
        choices=("eva", "open_clip", "openai"),
        default="eva",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-clusters", type=int, default=512)
    parser.add_argument("--noun-families", type=int, default=96)
    parser.add_argument("--cluster-iterations", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--fit-samples", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--synthetic-quantile",
        type=float,
        default=0.05,
        help="Observed noun-evidence similarity quantile used for filtering.",
    )
    parser.add_argument("--synthetic-margin", type=float, default=0.02)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("phrase"):
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_number}"
                )
            records.append(record)
    if not records:
        raise RuntimeError(f"No records loaded from {path}")
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_text_encoder(
    backend: str,
    model_name: str,
    checkpoint: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Callable[[List[str]], Tensor]]:
    if backend == "eva":
        import cat_seg.src.open_clip as open_clip
        import cat_seg.src.open_clip.eva_clip as eva_clip

        model = eva_clip.create_model(
            model_name=model_name,
            pretrained=checkpoint or None,
            force_custom_clip=True,
            precision="fp32",
            device=str(device),
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    elif backend == "open_clip":
        import cat_seg.src.open_clip as open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=checkpoint or None,
            device=str(device),
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        from cat_seg.third_party import clip

        model, _ = clip.load(model_name, device=str(device), jit=False)
        tokenizer = clip.tokenize
    return model.float().eval(), tokenizer


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    model: torch.nn.Module,
    tokenizer: Callable[[List[str]], Tensor],
    device: torch.device,
    batch_size: int,
) -> Tensor:
    batches = []
    for start in range(0, len(texts), max(1, batch_size)):
        tokens = tokenizer(list(texts[start:start + batch_size])).to(device)
        encoded = model.encode_text(tokens)
        batches.append(F.normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(batches, dim=0)


def project_embeddings(
    embeddings: Tensor,
    projection_dim: int,
    seed: int,
) -> Tensor:
    embeddings = F.normalize(embeddings.float().cpu(), dim=-1)
    if embeddings.shape[-1] <= projection_dim:
        return embeddings
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(
        embeddings.shape[-1],
        projection_dim,
        generator=generator,
    )
    projection = F.normalize(projection, dim=0)
    return F.normalize(embeddings @ projection, dim=-1)


def weighted_spherical_kmeans(
    embeddings: Tensor,
    weights: Tensor,
    num_clusters: int,
    iterations: int,
    projection_dim: int,
    fit_samples: int,
    chunk_size: int,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    embeddings = F.normalize(embeddings.float().cpu(), dim=-1)
    weights = weights.float().cpu().flatten().clamp_min(1e-6)
    if embeddings.shape[0] != weights.numel():
        raise ValueError("Weights must match embeddings.")
    num_items = embeddings.shape[0]
    num_clusters = max(1, min(int(num_clusters), num_items))
    if num_clusters == num_items:
        return embeddings, torch.arange(num_items, dtype=torch.long)

    projected = project_embeddings(
        embeddings,
        projection_dim=projection_dim,
        seed=seed,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    fit_count = min(num_items, max(num_clusters, int(fit_samples)))
    probabilities = weights / weights.sum()
    fit_indices = torch.multinomial(
        probabilities,
        fit_count,
        replacement=False,
        generator=generator,
    )
    fit = projected[fit_indices]
    fit_weights = weights[fit_indices]

    selected = [int(fit_weights.argmax())]
    maximum_similarity = fit @ fit[selected[0]]
    for _ in range(1, num_clusters):
        distance = (1.0 - maximum_similarity).clamp_min(0.0)
        initialization_score = distance * fit_weights.sqrt()
        initialization_score[selected] = -1
        next_index = int(initialization_score.argmax())
        selected.append(next_index)
        maximum_similarity = torch.maximum(
            maximum_similarity,
            fit @ fit[next_index],
        )
    centers = fit[selected].clone()

    for _ in range(max(1, int(iterations))):
        labels_fit = (fit @ centers.t()).argmax(dim=-1)
        weighted_fit = fit * fit_weights[:, None]
        sums = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters, dtype=torch.float32)
        sums.index_add_(0, labels_fit, weighted_fit)
        counts.index_add_(0, labels_fit, fit_weights)
        non_empty = counts > 0
        centers[non_empty] = F.normalize(
            sums[non_empty] / counts[non_empty, None],
            dim=-1,
        )

    labels = torch.empty(num_items, dtype=torch.long)
    for start in range(0, num_items, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), num_items)
        labels[start:end] = (
            projected[start:end] @ centers.t()
        ).argmax(dim=-1)

    full_sums = torch.zeros(
        num_clusters,
        embeddings.shape[-1],
        dtype=torch.float32,
    )
    full_counts = torch.zeros(num_clusters, dtype=torch.float32)
    full_sums.index_add_(0, labels, embeddings * weights[:, None])
    full_counts.index_add_(0, labels, weights)
    non_empty = full_counts > 0
    full_centers = torch.zeros_like(full_sums)
    full_centers[non_empty] = F.normalize(
        full_sums[non_empty] / full_counts[non_empty, None],
        dim=-1,
    )
    if (~non_empty).any():
        replacement = weights.argsort(descending=True)
        full_centers[~non_empty] = embeddings[
            replacement[:int((~non_empty).sum())]
        ]
    return F.normalize(full_centers, dim=-1), labels


def candidate_weights(records: Sequence[dict]) -> Tensor:
    values = []
    for record in records:
        support = max(1, int(record.get("support_count", 1)))
        score = max(
            0.05,
            float(record.get("compatibility_score", 0.5)),
        )
        values.append(math.sqrt(support) * score)
    return torch.tensor(values, dtype=torch.float32)


def filter_synthetic_phrases(
    records: Sequence[dict],
    embeddings: Tensor,
    quantile: float,
    margin: float,
) -> Tuple[List[dict], Tensor, List[dict]]:
    noun_to_indices: Dict[str, List[int]] = defaultdict(list)
    noun_to_observed: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        noun = str(record["noun"])
        noun_to_indices[noun].append(index)
        if record.get("provenance") == "observed":
            noun_to_observed[noun].append(index)

    noun_centers = {}
    observed_similarities = []
    noun_observed_similarities = {}
    for noun, indices in noun_to_observed.items():
        center = F.normalize(
            embeddings[indices].mean(dim=0),
            dim=-1,
        )
        noun_centers[noun] = center
        similarities = embeddings[indices] @ center
        noun_observed_similarities[noun] = similarities
        observed_similarities.append(similarities)

    if observed_similarities:
        global_observed = torch.cat(observed_similarities)
        global_threshold = float(torch.quantile(
            global_observed,
            min(max(float(quantile), 0.0), 1.0),
        ))
    else:
        global_threshold = -1.0

    kept_records = []
    kept_embeddings = []
    rejected = []
    for index, record in enumerate(records):
        if record.get("provenance") != "same_noun_recombined":
            kept_records.append(record)
            kept_embeddings.append(embeddings[index])
            continue
        noun = str(record["noun"])
        center = noun_centers.get(noun)
        if center is None:
            rejected_record = dict(record)
            rejected_record["rejection_reason"] = "no_observed_noun_center"
            rejected.append(rejected_record)
            continue
        similarities = noun_observed_similarities.get(noun)
        threshold = global_threshold
        if similarities is not None and similarities.numel() >= 4:
            threshold = float(torch.quantile(
                similarities,
                min(max(float(quantile), 0.0), 1.0),
            ))
        threshold -= float(margin)
        similarity = float(embeddings[index] @ center)
        if similarity >= threshold:
            accepted = dict(record)
            accepted["clip_evidence_similarity"] = similarity
            accepted["clip_filter_threshold"] = threshold
            kept_records.append(accepted)
            kept_embeddings.append(embeddings[index])
        else:
            rejected_record = dict(record)
            rejected_record["clip_evidence_similarity"] = similarity
            rejected_record["clip_filter_threshold"] = threshold
            rejected_record["rejection_reason"] = "low_clip_evidence_similarity"
            rejected.append(rejected_record)

    return (
        kept_records,
        torch.stack(kept_embeddings),
        rejected,
    )


def allocate_family_clusters(
    family_ids: Tensor,
    weights: Tensor,
    total_clusters: int,
) -> Dict[int, int]:
    families = sorted(set(family_ids.tolist()))
    counts = Counter(family_ids.tolist())
    if total_clusters < len(families):
        raise ValueError(
            "Total clusters must cover every non-empty noun family."
        )
    allocation = {family: 1 for family in families}
    family_mass = {
        family: float(weights[family_ids == family].sum())
        for family in families
    }
    remaining = total_clusters - len(families)
    for _ in range(remaining):
        eligible = [
            family
            for family in families
            if allocation[family] < counts[family]
        ]
        if not eligible:
            break
        family = max(
            eligible,
            key=lambda value: (
                math.sqrt(max(family_mass[value], 1e-8))
                / (allocation[value] + 1),
                counts[value],
                -value,
            ),
        )
        allocation[family] += 1
    if sum(allocation.values()) != total_clusters:
        raise RuntimeError("Unable to allocate the requested cluster count.")
    return allocation


def noun_aware_clustering(
    records: Sequence[dict],
    embeddings: Tensor,
    weights: Tensor,
    model: torch.nn.Module,
    tokenizer: Callable[[List[str]], Tensor],
    device: torch.device,
    batch_size: int,
    num_clusters: int,
    noun_families: int,
    iterations: int,
    projection_dim: int,
    fit_samples: int,
    chunk_size: int,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    nouns = sorted({str(record["noun"]) for record in records})
    noun_to_index = {noun: index for index, noun in enumerate(nouns)}
    noun_embeddings = encode_texts(
        nouns,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    noun_weights = torch.zeros(len(nouns), dtype=torch.float32)
    for record, weight in zip(records, weights):
        noun_weights[noun_to_index[str(record["noun"])]] += weight

    family_count = min(
        max(1, int(noun_families)),
        len(nouns),
        max(1, int(num_clusters)),
    )
    _, noun_family_labels = weighted_spherical_kmeans(
        noun_embeddings,
        noun_weights,
        num_clusters=family_count,
        iterations=iterations,
        projection_dim=projection_dim,
        fit_samples=fit_samples,
        chunk_size=chunk_size,
        seed=seed,
    )
    record_family_ids = torch.tensor([
        int(noun_family_labels[noun_to_index[str(record["noun"])]])
        for record in records
    ])
    allocation = allocate_family_clusters(
        record_family_ids,
        weights,
        total_clusters=num_clusters,
    )

    all_centers = []
    global_labels = torch.empty(len(records), dtype=torch.long)
    center_family_ids = []
    offset = 0
    for family in sorted(allocation):
        indices = torch.nonzero(
            record_family_ids == family,
            as_tuple=False,
        ).flatten()
        family_centers, family_labels = weighted_spherical_kmeans(
            embeddings[indices],
            weights[indices],
            num_clusters=allocation[family],
            iterations=iterations,
            projection_dim=projection_dim,
            fit_samples=fit_samples,
            chunk_size=chunk_size,
            seed=seed + family + 1,
        )
        all_centers.append(family_centers)
        global_labels[indices] = family_labels + offset
        center_family_ids.extend([family] * family_centers.shape[0])
        offset += family_centers.shape[0]
    centers = torch.cat(all_centers, dim=0)
    if centers.shape[0] != num_clusters:
        raise RuntimeError(
            f"Expected {num_clusters} centers, got {centers.shape[0]}."
        )
    return (
        centers,
        global_labels,
        torch.tensor(center_family_ids, dtype=torch.long),
    )


def cluster_metadata(
    records: Sequence[dict],
    embeddings: Tensor,
    centers: Tensor,
    labels: Tensor,
) -> Tuple[List[str], List[List[str]]]:
    representatives = []
    source_classes = []
    for cluster_index in range(centers.shape[0]):
        members = torch.nonzero(
            labels == cluster_index,
            as_tuple=False,
        ).flatten()
        if members.numel() == 0:
            representatives.append("")
            source_classes.append([])
            continue
        similarities = embeddings[members] @ centers[cluster_index]
        representative_index = int(members[similarities.argmax()])
        representatives.append(str(
            records[representative_index]["phrase"]
        ))
        classes = set()
        for member in members.tolist():
            classes.update(records[member].get("source_classes", []))
        source_classes.append(sorted(classes))
    return representatives, source_classes


def build_payload(
    kind: str,
    records: Sequence[dict],
    embeddings: Tensor,
    centers: Tensor,
    labels: Tensor,
    source_json: str,
    source_hash: str,
    candidate_path: Path,
    model_name: str,
    checkpoint_hash: str,
    center_family_ids: Tensor | None = None,
    rejected_count: int = 0,
) -> dict:
    representatives, cluster_classes = cluster_metadata(
        records,
        embeddings,
        centers,
        labels,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "embeddings": centers.cpu(),
        "attribute_type_ids": torch.full(
            (centers.shape[0],),
            7,
            dtype=torch.long,
        ),
        "representative_phrases": representatives,
        "cluster_source_classes": cluster_classes,
        "records": list(records),
        "cluster_labels": labels.cpu(),
        "meta": {
            "database_kind": kind,
            "expansion_level": 2,
            "source_json": str(Path(source_json)),
            "source_sha256": source_hash,
            "candidate_jsonl": str(candidate_path),
            "candidate_sha256": file_sha256(str(candidate_path)),
            "model_name": model_name,
            "clip_checkpoint_sha256": checkpoint_hash,
            "clip_output_dim": int(embeddings.shape[-1]),
            "include_interactions": False,
            "strict_train_only": True,
            "num_source_attributes": len(records),
            "num_clusters": int(centers.shape[0]),
            "rejected_synthetic_candidates": int(rejected_count),
            "clustering": (
                "noun_family_weighted_spherical_kmeans"
                if center_family_ids is not None
                else "weighted_spherical_kmeans"
            ),
        },
    }
    if center_family_ids is not None:
        payload["cluster_noun_family_ids"] = center_family_ids.cpu()
        payload["meta"]["num_noun_families"] = int(
            center_family_ids.unique().numel()
        )
    return payload


def atomic_save(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)


def main() -> None:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    attribute_path = candidate_dir / "attribute_candidates.jsonl"
    phrase_path = candidate_dir / "phrase_candidates_level2.jsonl"
    attribute_records = read_jsonl(attribute_path)
    attribute_records = [
        record
        for record in attribute_records
        if (
            record.get("phrase") not in {"other", "stuff"}
            and any(
                relation in {"amod", "acl"}
                for relation in record.get("relations", [])
            )
        )
    ]
    if len(attribute_records) < args.num_clusters:
        raise RuntimeError(
            "Attribute-only filtering left fewer records than clusters."
        )
    phrase_records = read_jsonl(phrase_path)

    device = torch.device(args.device)
    model, tokenizer = load_text_encoder(
        backend=args.backend,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        device=device,
    )

    attribute_embeddings = encode_texts(
        [str(record["phrase"]) for record in attribute_records],
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )
    phrase_embeddings = encode_texts(
        [str(record["phrase"]) for record in phrase_records],
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )
    (
        filtered_phrase_records,
        filtered_phrase_embeddings,
        rejected_records,
    ) = filter_synthetic_phrases(
        phrase_records,
        phrase_embeddings,
        quantile=args.synthetic_quantile,
        margin=args.synthetic_margin,
    )
    rejected_path = candidate_dir / "rejected_candidates_clip.jsonl"
    write_jsonl(rejected_path, rejected_records)

    attribute_weight_values = candidate_weights(attribute_records)
    phrase_weight_values = candidate_weights(filtered_phrase_records)
    attribute_centers, attribute_labels = weighted_spherical_kmeans(
        attribute_embeddings,
        attribute_weight_values,
        num_clusters=args.num_clusters,
        iterations=args.cluster_iterations,
        projection_dim=args.projection_dim,
        fit_samples=args.fit_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    (
        phrase_centers,
        phrase_labels,
        center_family_ids,
    ) = noun_aware_clustering(
        filtered_phrase_records,
        filtered_phrase_embeddings,
        phrase_weight_values,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        num_clusters=args.num_clusters,
        noun_families=args.noun_families,
        iterations=args.cluster_iterations,
        projection_dim=args.projection_dim,
        fit_samples=args.fit_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    source_hash = file_sha256(args.source_json)
    checkpoint_hash = file_sha256(args.checkpoint)
    attribute_payload = build_payload(
        kind="attribute_only",
        records=attribute_records,
        embeddings=attribute_embeddings,
        centers=attribute_centers,
        labels=attribute_labels,
        source_json=args.source_json,
        source_hash=source_hash,
        candidate_path=attribute_path,
        model_name=args.model_name,
        checkpoint_hash=checkpoint_hash,
    )
    phrase_payload = build_payload(
        kind="expanded_attribute_noun_phrase",
        records=filtered_phrase_records,
        embeddings=filtered_phrase_embeddings,
        centers=phrase_centers,
        labels=phrase_labels,
        source_json=args.source_json,
        source_hash=source_hash,
        candidate_path=phrase_path,
        model_name=args.model_name,
        checkpoint_hash=checkpoint_hash,
        center_family_ids=center_family_ids,
        rejected_count=len(rejected_records),
    )
    attribute_output = Path(args.attribute_output)
    phrase_output = Path(args.phrase_output)
    atomic_save(attribute_payload, attribute_output)
    atomic_save(phrase_payload, phrase_output)

    report = {
        "attribute_output": str(attribute_output),
        "phrase_output": str(phrase_output),
        "attribute_records": len(attribute_records),
        "phrase_records_before_clip_filter": len(phrase_records),
        "phrase_records_after_clip_filter": len(
            filtered_phrase_records
        ),
        "rejected_synthetic_candidates": len(rejected_records),
        "attribute_embedding_shape": list(
            attribute_payload["embeddings"].shape
        ),
        "phrase_embedding_shape": list(
            phrase_payload["embeddings"].shape
        ),
        "noun_families": int(center_family_ids.unique().numel()),
        "attribute_provenance": dict(Counter(
            record.get("provenance", "unknown")
            for record in attribute_records
        )),
        "phrase_provenance": dict(Counter(
            record.get("provenance", "unknown")
            for record in filtered_phrase_records
        )),
        "parameters": vars(args),
    }
    report_path = candidate_dir / "database_build_report.json"
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

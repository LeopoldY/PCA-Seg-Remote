#!/usr/bin/env python3
"""Extract and expand Level-2 attribute--noun candidates with spaCy.

This stage is intentionally independent of PyTorch/Detectron2. It parses the
training-only phrase JSON, builds an evidence graph, and writes auditable JSONL
inputs for the CLIP encoding/database stage.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


PROVENANCE_PRIORITY = {
    "observed": 3,
    "decomposed": 2,
    "same_noun_recombined": 1,
}
STRUCTURAL_PLACEHOLDERS = {"other", "stuff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--top-attributes-per-noun", type=int, default=8)
    parser.add_argument("--max-generated-per-noun", type=int, default=32)
    parser.add_argument("--min-edge-support", type=int, default=2)
    parser.add_argument("--min-global-pair-support", type=int, default=1)
    parser.add_argument("--productive-compound-min-support", type=int, default=3)
    parser.add_argument("--productive-compound-min-heads", type=int, default=3)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_lemma(token) -> str:
    lemma = token.lemma_.strip().lower()
    if not lemma or lemma == "-pron-":
        lemma = token.text.strip().lower()
    return normalize_text(lemma)


def canonical_category(category: str) -> str:
    parts = [
        part
        for part in normalize_text(category).replace("-", " ").split()
        if part not in {"other", "stuff"}
    ]
    return " ".join(parts)


def modifier_phrase(token) -> str:
    """Return a compact modifier expression rooted at ``token``."""
    members = [token]
    for child in token.children:
        if (
            child.i < token.i
            and child.dep_ in {"advmod", "neg"}
            and child.pos_ in {"ADV", "ADJ", "PART"}
        ):
            members.append(child)
    members.sort(key=lambda value: value.i)
    return normalize_text(" ".join(value.text for value in members))


def adjective_roots(chunk, root) -> List[Tuple[str, int, str]]:
    roots = []
    direct = {
        token.i
        for token in chunk
        if token.head == root and token.dep_ == "amod"
    }
    for token in chunk:
        is_direct = token.i in direct
        is_conjoined = (
            token.dep_ == "conj"
            and (
                token.head.i in direct
                or (
                    token.head.dep_ == "conj"
                    and token.head.head.i in direct
                )
            )
        )
        if is_direct or is_conjoined:
            phrase = modifier_phrase(token)
            if phrase:
                roots.append((phrase, token.i, "amod"))
    for token in root.children:
        if (
            token.dep_ == "acl"
            and token.i < root.i
            and token.pos_ in {"VERB", "ADJ"}
        ):
            phrase = modifier_phrase(token)
            if phrase:
                roots.append((phrase, token.i, "acl"))
    return roots


def match_categories(noun: str, categories: Sequence[str]) -> List[str]:
    noun_tokens = set(noun.replace("-", " ").split())
    scored = []
    for original in categories:
        canonical = canonical_category(original)
        category_tokens = set(canonical.split())
        if not category_tokens:
            continue
        overlap = len(noun_tokens & category_tokens)
        exact = int(canonical == noun)
        contained = int(canonical in noun or noun in canonical)
        score = 3 * exact + 2 * contained + overlap / len(category_tokens)
        if score > 0:
            scored.append((score, str(original)))
    if not scored:
        return []
    maximum = max(score for score, _ in scored)
    return sorted({
        category
        for score, category in scored
        if score == maximum
    })


def add_candidate(
    candidates: Dict[str, dict],
    phrase: str,
    noun: str,
    attributes: Sequence[str],
    provenance: str,
    score: float,
    image_ids: Iterable[int],
    source_phrases: Iterable[str],
    source_categories: Iterable[str],
) -> None:
    phrase = normalize_text(phrase)
    if not phrase:
        return
    incoming = {
        "phrase": phrase,
        "noun": noun,
        "attributes": list(attributes),
        "provenance": provenance,
        "compatibility_score": float(score),
        "source_image_ids": set(image_ids),
        "source_phrases": set(source_phrases),
        "source_classes": set(source_categories),
    }
    current = candidates.get(phrase)
    if current is None:
        candidates[phrase] = incoming
        return
    if (
        PROVENANCE_PRIORITY[provenance]
        > PROVENANCE_PRIORITY[current["provenance"]]
    ):
        current["provenance"] = provenance
        current["noun"] = noun
        current["attributes"] = list(attributes)
    current["compatibility_score"] = max(
        current["compatibility_score"],
        float(score),
    )
    current["source_image_ids"].update(incoming["source_image_ids"])
    current["source_phrases"].update(incoming["source_phrases"])
    current["source_classes"].update(incoming["source_classes"])


def render_candidate_phrase(
    attributes: Sequence[str],
    noun: str,
    pair_orders: Counter,
    conjunction_orders: Counter,
) -> str:
    if len(attributes) == 2:
        forward_key = (attributes[0], attributes[1])
        backward_key = (attributes[1], attributes[0])
        forward = conjunction_orders[forward_key]
        backward = conjunction_orders[backward_key]
        total = pair_orders[forward_key] + pair_orders[backward_key]
        if max(forward, backward) > 0 and 2 * (forward + backward) >= total:
            ordered = (
                attributes
                if forward >= backward
                else (attributes[1], attributes[0])
            )
            return f"{ordered[0]} and {ordered[1]} {noun}"
    return " ".join((*attributes, noun))


def serializable_candidate(record: dict) -> dict:
    output = dict(record)
    for key in ("source_image_ids", "source_phrases", "source_classes"):
        output[key] = sorted(output[key])
    output["support_count"] = len(output["source_image_ids"])
    return output


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    import spacy

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as stream:
        source_data = json.load(stream)
    if not isinstance(source_data, dict):
        raise ValueError("Input JSON must map image ids to records.")

    phrase_instances = []
    for item in source_data.values():
        image_id = int(item.get("image_id", -1))
        categories = [str(value) for value in item.get(
            "original_categories", []
        )]
        for phrase in item.get("specified", []):
            normalized = normalize_text(phrase)
            if normalized:
                phrase_instances.append(
                    (normalized, {
                        "image_id": image_id,
                        "categories": categories,
                        "source_phrase": str(phrase).strip(),
                    })
                )

    nlp = spacy.load(args.spacy_model, disable=["ner"])
    raw_chunks = []
    compound_heads: Dict[str, Counter] = defaultdict(Counter)
    compound_totals = Counter()

    parsed = nlp.pipe(
        phrase_instances,
        as_tuples=True,
        batch_size=max(1, args.batch_size),
        n_process=max(1, args.n_process),
    )
    for doc, context in parsed:
        for chunk in doc.noun_chunks:
            root = chunk.root
            if root.pos_ not in {"NOUN", "PROPN"}:
                continue
            root_lemma = normalize_lemma(root)
            if not root_lemma:
                continue
            compounds = []
            for token in chunk:
                if (
                    token.head == root
                    and token.dep_ == "compound"
                    and token.pos_ in {"NOUN", "PROPN", "ADJ"}
                ):
                    lemma = normalize_lemma(token)
                    if lemma:
                        compounds.append((lemma, token.i))
                        compound_heads[lemma][root_lemma] += 1
                        compound_totals[lemma] += 1
            raw_chunks.append({
                "image_id": context["image_id"],
                "categories": context["categories"],
                "source_phrase": context["source_phrase"],
                "root": root_lemma,
                "root_index": root.i,
                "compounds": compounds,
                "modifiers": adjective_roots(chunk, root),
            })

    productive_compounds = {
        modifier
        for modifier, count in compound_totals.items()
        if (
            count >= args.productive_compound_min_support
            and len(compound_heads[modifier])
            >= args.productive_compound_min_heads
        )
    }

    edges: Dict[Tuple[str, str], dict] = {}
    observed_combinations: Dict[Tuple[Tuple[str, ...], str], dict] = {}
    global_pair_images: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    global_pair_order = Counter()
    global_pair_conjunction_order = Counter()

    for raw in raw_chunks:
        if raw["root"] in STRUCTURAL_PLACEHOLDERS:
            continue
        noun_parts = [
            (compound, index)
            for compound, index in raw["compounds"]
            if (
                compound not in productive_compounds
                and compound not in STRUCTURAL_PLACEHOLDERS
            )
        ]
        noun_parts.append((raw["root"], raw["root_index"]))
        noun_parts.sort(key=lambda value: value[1])
        noun = normalize_text(" ".join(value for value, _ in noun_parts))
        if not noun:
            continue

        modifiers = list(raw["modifiers"])
        for compound, index in raw["compounds"]:
            if compound in productive_compounds:
                modifiers.append((compound, index, "compound_modifier"))
        modifiers.sort(key=lambda value: value[1])

        ordered_attributes = []
        relations = {}
        for attribute, _, relation in modifiers:
            if (
                attribute
                and attribute != noun
                and attribute not in STRUCTURAL_PLACEHOLDERS
                and attribute not in ordered_attributes
            ):
                ordered_attributes.append(attribute)
                relations[attribute] = relation
        if not ordered_attributes:
            continue

        matched_categories = match_categories(noun, raw["categories"])
        for attribute in ordered_attributes:
            key = (attribute, noun)
            edge = edges.setdefault(key, {
                "attribute": attribute,
                "noun": noun,
                "relations": set(),
                "image_ids": set(),
                "source_phrases": set(),
                "source_classes": set(),
            })
            edge["relations"].add(relations[attribute])
            if raw["image_id"] >= 0:
                edge["image_ids"].add(raw["image_id"])
            edge["source_phrases"].add(raw["source_phrase"])
            edge["source_classes"].update(matched_categories)

        combo_key = (tuple(ordered_attributes), noun)
        combo = observed_combinations.setdefault(combo_key, {
            "image_ids": set(),
            "source_phrases": set(),
            "source_classes": set(),
        })
        if raw["image_id"] >= 0:
            combo["image_ids"].add(raw["image_id"])
        combo["source_phrases"].add(raw["source_phrase"])
        combo["source_classes"].update(matched_categories)

        for first, second in itertools.combinations(
            ordered_attributes, 2
        ):
            pair = tuple(sorted((first, second)))
            if raw["image_id"] >= 0:
                global_pair_images[pair].add(raw["image_id"])
            global_pair_order[(first, second)] += 1
            normalized_source = normalize_text(raw["source_phrase"])
            if f"{first} and {second}" in normalized_source:
                global_pair_conjunction_order[(first, second)] += 1
            elif f"{second} and {first}" in normalized_source:
                global_pair_conjunction_order[(second, first)] += 1

    attribute_images: Dict[str, Set[int]] = defaultdict(set)
    noun_images: Dict[str, Set[int]] = defaultdict(set)
    noun_edge_total = Counter()
    for edge in edges.values():
        attribute_images[edge["attribute"]].update(edge["image_ids"])
        noun_images[edge["noun"]].update(edge["image_ids"])
        noun_edge_total[edge["noun"]] += len(edge["image_ids"])
    total_events = max(
        1,
        sum(len(edge["image_ids"]) for edge in edges.values()),
    )

    for edge in edges.values():
        support = max(1, len(edge["image_ids"]))
        attr_support = max(
            1,
            len(attribute_images[edge["attribute"]]),
        )
        noun_support = max(1, len(noun_images[edge["noun"]]))
        p_pair = support / total_events
        p_attr = attr_support / total_events
        p_noun = noun_support / total_events
        pmi = math.log(max(p_pair / max(p_attr * p_noun, 1e-12), 1e-12))
        npmi = pmi / max(-math.log(max(p_pair, 1e-12)), 1e-12)
        support_norm = math.log1p(support) / math.log1p(
            max(1, noun_support)
        )
        conditional = support / max(1, noun_edge_total[edge["noun"]])
        edge["npmi"] = float(npmi)
        edge["conditional_probability"] = float(conditional)
        edge["score"] = float(
            0.45 * support_norm
            + 0.35 * max(0.0, npmi)
            + 0.20 * conditional
        )

    phrase_candidates: Dict[str, dict] = {}

    for (attributes, noun), combo in observed_combinations.items():
        edge_scores = [
            edges[(attribute, noun)]["score"]
            for attribute in attributes
        ]
        add_candidate(
            phrase_candidates,
            phrase=render_candidate_phrase(
                attributes,
                noun,
                global_pair_order,
                global_pair_conjunction_order,
            ),
            noun=noun,
            attributes=attributes,
            provenance="observed",
            score=sum(edge_scores) / len(edge_scores),
            image_ids=combo["image_ids"],
            source_phrases=combo["source_phrases"],
            source_categories=combo["source_classes"],
        )

    noun_to_edges: Dict[str, List[dict]] = defaultdict(list)
    for edge in edges.values():
        noun_to_edges[edge["noun"]].append(edge)
        add_candidate(
            phrase_candidates,
            phrase=f"{edge['attribute']} {edge['noun']}",
            noun=edge["noun"],
            attributes=[edge["attribute"]],
            provenance="decomposed",
            score=edge["score"],
            image_ids=edge["image_ids"],
            source_phrases=edge["source_phrases"],
            source_categories=edge["source_classes"],
        )

    generated_per_noun = Counter()
    for noun, noun_edges in noun_to_edges.items():
        eligible = [
            edge
            for edge in noun_edges
            if len(edge["image_ids"]) >= args.min_edge_support
        ]
        eligible.sort(
            key=lambda edge: (
                edge["score"],
                len(edge["image_ids"]),
                edge["attribute"],
            ),
            reverse=True,
        )
        eligible = eligible[:max(1, args.top_attributes_per_noun)]
        generated = []
        for first, second in itertools.combinations(eligible, 2):
            pair = tuple(sorted((
                first["attribute"],
                second["attribute"],
            )))
            pair_support = len(global_pair_images.get(pair, set()))
            if pair_support < args.min_global_pair_support:
                continue
            forward = global_pair_order[(
                first["attribute"],
                second["attribute"],
            )]
            backward = global_pair_order[(
                second["attribute"],
                first["attribute"],
            )]
            ordered = (
                (first["attribute"], second["attribute"])
                if forward >= backward
                else (second["attribute"], first["attribute"])
            )
            pair_score = math.log1p(pair_support) / math.log(10.0)
            score = (
                0.4 * first["score"]
                + 0.4 * second["score"]
                + 0.2 * min(1.0, pair_score)
            )
            generated.append((score, ordered, first, second))
        generated.sort(key=lambda value: value[0], reverse=True)

        for score, attributes, first, second in generated[
            :max(0, args.max_generated_per_noun)
        ]:
            image_ids = (
                first["image_ids"]
                | second["image_ids"]
                | global_pair_images[tuple(sorted(attributes))]
            )
            source_phrases = (
                first["source_phrases"]
                | second["source_phrases"]
            )
            source_classes = (
                first["source_classes"]
                | second["source_classes"]
            )
            before = len(phrase_candidates)
            add_candidate(
                phrase_candidates,
                phrase=render_candidate_phrase(
                    attributes,
                    noun,
                    global_pair_order,
                    global_pair_conjunction_order,
                ),
                noun=noun,
                attributes=attributes,
                provenance="same_noun_recombined",
                score=score,
                image_ids=image_ids,
                source_phrases=source_phrases,
                source_categories=source_classes,
            )
            if len(phrase_candidates) > before:
                generated_per_noun[noun] += 1

    attribute_candidates = []
    attribute_to_edges: Dict[str, List[dict]] = defaultdict(list)
    for edge in edges.values():
        attribute_to_edges[edge["attribute"]].append(edge)
    for attribute, image_ids in attribute_images.items():
        related_edges = attribute_to_edges[attribute]
        source_phrases = set()
        source_classes = set()
        nouns = set()
        relations = set()
        for edge in related_edges:
            source_phrases.update(edge["source_phrases"])
            source_classes.update(edge["source_classes"])
            nouns.add(edge["noun"])
            relations.update(edge["relations"])
        score = max(edge["score"] for edge in related_edges)
        attribute_candidates.append({
            "phrase": attribute,
            "attribute": attribute,
            "nouns": sorted(nouns),
            "relations": sorted(relations),
            "provenance": "observed_attribute",
            "compatibility_score": float(score),
            "support_count": len(image_ids),
            "source_image_ids": sorted(image_ids),
            "source_phrases": sorted(source_phrases),
            "source_classes": sorted(source_classes),
        })

    serialized_phrases = [
        serializable_candidate(record)
        for record in phrase_candidates.values()
    ]
    serialized_phrases.sort(
        key=lambda record: (
            record["noun"],
            -record["compatibility_score"],
            record["phrase"],
        )
    )
    attribute_candidates.sort(
        key=lambda record: (
            -record["compatibility_score"],
            -record["support_count"],
            record["phrase"],
        )
    )

    edge_records = []
    for edge in edges.values():
        edge_records.append({
            "attribute": edge["attribute"],
            "noun": edge["noun"],
            "relations": sorted(edge["relations"]),
            "support_count": len(edge["image_ids"]),
            "source_image_ids": sorted(edge["image_ids"]),
            "source_phrases": sorted(edge["source_phrases"]),
            "source_classes": sorted(edge["source_classes"]),
            "npmi": edge["npmi"],
            "conditional_probability": edge[
                "conditional_probability"
            ],
            "compatibility_score": edge["score"],
        })
    edge_records.sort(
        key=lambda record: (
            record["noun"],
            -record["compatibility_score"],
            record["attribute"],
        )
    )

    edge_count = write_jsonl(
        output_dir / "observed_relations.jsonl",
        edge_records,
    )
    attribute_count = write_jsonl(
        output_dir / "attribute_candidates.jsonl",
        attribute_candidates,
    )
    phrase_count = write_jsonl(
        output_dir / "phrase_candidates_level2.jsonl",
        serialized_phrases,
    )

    provenance_counts = Counter(
        record["provenance"]
        for record in serialized_phrases
    )
    report = {
        "input_json": str(input_path),
        "spacy_model": args.spacy_model,
        "source_images": len(source_data),
        "specified_instances": len(phrase_instances),
        "noun_chunks_with_attributes": len(raw_chunks),
        "productive_compounds": len(productive_compounds),
        "observed_relation_edges": edge_count,
        "unique_attributes": attribute_count,
        "unique_nouns": len(noun_to_edges),
        "phrase_candidates": phrase_count,
        "phrase_provenance": dict(provenance_counts),
        "nouns_with_level2_expansion": sum(
            value > 0 for value in generated_per_noun.values()
        ),
        "level2_generated_total": sum(generated_per_noun.values()),
        "parameters": vars(args),
        "samples": {
            "attributes": attribute_candidates[:20],
            "phrases": serialized_phrases[:20],
        },
    }
    with open(
        output_dir / "candidate_report.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)

    print(json.dumps(
        {
            key: value
            for key, value in report.items()
            if key not in {"samples", "parameters"}
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()

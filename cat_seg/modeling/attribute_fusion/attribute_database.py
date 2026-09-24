"""Utilities for ExCEL-format attribute databases.

ExCEL stores a two-item list: a ``[D, M]`` KMeans center bank followed by a
``[C, M]`` binary matrix that records which clusters occur in each source
class.  PCA-Seg uses the center bank exactly as ExCEL does; the class matrix is
retained so rebuilt databases stay format-compatible with the official files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def load_excel_attribute_database(
    database_path: str,
    expected_dim: Optional[int] = None,
    expected_num_clusters: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"ExCEL attribute database not found: {path}")

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError(
            "An ExCEL attribute database must be [cluster_bank, class_flags]."
        )
    cluster_bank, class_flags = payload
    if not isinstance(cluster_bank, Tensor) or cluster_bank.ndim != 2:
        raise ValueError("ExCEL cluster_bank must have shape [D, M].")
    if not isinstance(class_flags, Tensor) or class_flags.ndim != 2:
        raise ValueError("ExCEL class_flags must have shape [C, M].")
    if class_flags.shape[1] != cluster_bank.shape[1]:
        raise ValueError("ExCEL cluster_bank and class_flags disagree on M.")
    if expected_dim is not None and cluster_bank.shape[0] != expected_dim:
        raise ValueError(
            f"Attribute dimension {cluster_bank.shape[0]} does not match "
            f"the PCA-Seg text dimension {expected_dim}. Re-encode the "
            "official ExCEL descriptor JSON with PCA-Seg's text encoder."
        )
    if (
        expected_num_clusters is not None
        and cluster_bank.shape[1] != expected_num_clusters
    ):
        raise ValueError(
            f"Attribute cluster count {cluster_bank.shape[1]} does not match "
            f"the configured value {expected_num_clusters}."
        )
    return cluster_bank.float(), class_flags.float()


def build_excel_cluster_bank(
    class_embeddings: Sequence[Tensor],
    num_clusters: int,
) -> Tuple[Tensor, Tensor]:
    """Reproduce ExCEL's ``attr_clustering`` database construction."""
    if not class_embeddings:
        raise ValueError("No ExCEL descriptor embeddings were provided.")
    if any(item.ndim != 2 or item.shape[0] == 0 for item in class_embeddings):
        raise ValueError("Each class embedding tensor must be non-empty [N, D].")

    embeddings = torch.cat([item.float().cpu() for item in class_embeddings])
    if num_clusters <= 0 or num_clusters > embeddings.shape[0]:
        raise ValueError("num_clusters must be in [1, number of descriptors].")

    # These are deliberately the same estimator and arguments as ExCEL.
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
        embeddings.numpy()
    )
    class_indices = torch.cat([
        torch.full((item.shape[0],), index, dtype=torch.long)
        for index, item in enumerate(class_embeddings)
    ])
    class_flags = []
    for class_index in range(len(class_embeddings)):
        labels = kmeans.labels_[class_indices.numpy() == class_index]
        active_clusters = np.unique(labels)
        flags = np.zeros(num_clusters, dtype=np.float32)
        flags[active_clusters] = 1.0
        class_flags.append(flags)

    cluster_bank = torch.from_numpy(
        kmeans.cluster_centers_.transpose(1, 0)
    )
    return cluster_bank, torch.from_numpy(np.stack(class_flags))

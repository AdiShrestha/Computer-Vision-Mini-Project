"""PyTorch dataset and data loaders for multi-modal time-series panels.

Enforces zero future-data leakage, split boundary isolation, deterministic
collation, worker seed initialization, and M3 memory budget batch bounds [8, 16].
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.utils.data

from .features import MultiModalPanel
from .splits import SplitManifest, _availability_record, parse_utc_timestamp


def seed_worker(worker_id: int) -> None:
    """Initialize worker seed deterministically for PyTorch DataLoader workers."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def deterministic_collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Collate panels into batch tensors, rejecting unobserved batches."""
    x_list, mask_list, meta_list = zip(*batch)
    batch_x = torch.stack(x_list, dim=0)
    batch_mask = torch.stack(mask_list, dim=0)

    # Reject batches with zero observed elements as unestimable
    if batch_mask.sum().item() == 0:
        raise ValueError("Batch contains zero valid observations; loss is unestimable")

    return batch_x, batch_mask, list(meta_list)


class PanelDataset(torch.utils.data.Dataset):
    """Dataset producing schema-sized contexts for one explicit fitting role."""

    def __init__(
        self,
        panels: Sequence[MultiModalPanel],
        split_manifest: SplitManifest,
        role: str,
        *, mode: str = "observational",
        normalizer: Optional[Any] = None,
    ):
        if mode != "fixture" or any(p.origin != "fixture" for p in panels):
            from .admission import blocked
            blocked("research panel loading")
        if role not in {"fitting", "development_validation", "threshold_calibration", "evaluation"}:
            raise ValueError("dataset requires one of the four explicit fitting roles")

        self.role = role
        self.split_manifest = split_manifest
        self.normalizer = normalizer
        physical_graph = split_manifest.physical_graph
        if physical_graph.get("cross_role_edges"):
            raise ValueError("fixture G_S contains cross-role physical edges")
        if physical_graph.get("unresolved_units") or physical_graph.get("unresolved_pairs"):
            raise ValueError("fixture G_S has unresolved physical or coordinate membership")
        if physical_graph.get("role_isolation_status") != "FIXTURE_ONLY_NOT_RESEARCH_VERIFIED":
            raise ValueError("fixture G_S must be explicitly fixture-only and isolated")
        graph_nodes = {node.get("lake_id"): node for node in physical_graph.get("nodes", ())}
        role_by_lake = {lake_id: role for role, member in split_manifest.scope_membership.items()
                        for lake_id in member["lake_ids"]}
        if len(graph_nodes) != len(physical_graph.get("nodes", ())) or set(graph_nodes) != set(role_by_lake):
            raise ValueError("fixture G_S nodes differ from explicit four-role lake membership")
        for lake_id, node in graph_nodes.items():
            if (node.get("scope") != role_by_lake[lake_id] or
                    node.get("physical_support_status") != "FIXTURE_VERIFIED" or
                    node.get("coordinate_status") != "FIXTURE_VERIFIED" or
                    not isinstance(node.get("physical_support_evidence"), Mapping) or
                    not isinstance(node.get("coordinate_evidence"), Mapping)):
                raise ValueError("fixture G_S lacks bound support/coordinate evidence for an assigned lake")

        membership = split_manifest.scope_membership[role]
        allowed_lakes = set(membership["lake_ids"])
        allowed_windows = set(membership["window_ids"])
        scheduled = {row.window_id: row for row in split_manifest.schedule_rows}

        # Filter and validate panels
        self.valid_panels: List[MultiModalPanel] = []
        for panel in panels:
            # A panel must bind to the exact, independently scheduled lake/window row.
            if panel.lake_id not in allowed_lakes:
                raise ValueError(
                    f"Panel lake {panel.lake_id} is not in allowed {role} lakes: {allowed_lakes}"
                )
            if panel.window_id not in allowed_windows or panel.window_id not in scheduled:
                raise ValueError("panel window is absent from explicit role schedule membership")
            window = scheduled[panel.window_id]
            if window.scope != role or window.lake_id != panel.lake_id:
                raise ValueError("panel identity differs from scheduled role/window membership")
            if not window.is_eligible or window.status != "ELIGIBLE":
                raise ValueError("panel cannot load from an ineligible or unassessed schedule row")
            if panel.start_date < window.context_start[:10]:
                raise ValueError("panel support starts before its declared context")
            if panel.end_date > window.decision_timestamp[:10]:
                raise ValueError("panel support ends after its decision time")
            decision = parse_utc_timestamp(window.decision_timestamp)
            context_start = parse_utc_timestamp(window.context_start)
            context_end = parse_utc_timestamp(window.context_end)
            supports = {source["record_id"]: source for source in window.metadata.get("source_supports", ())}
            if set(window.observation_ids) != set(supports):
                raise ValueError("scheduled source IDs and support rows differ")
            for source in supports.values():
                availability = _availability_record(source)
                support_start = parse_utc_timestamp(source["support_start"])
                support_end = parse_utc_timestamp(source["support_end"])
                available_at = parse_utc_timestamp(availability["at"]) if availability["at"] is not None else None
                if support_end < support_start:
                    raise ValueError("scheduled source support interval is reversed")
                if support_start < context_start or support_end > context_end:
                    raise ValueError("scheduled source support falls outside the closed panel context")
                if support_end >= decision:
                    raise ValueError("schedule contains future or open source support")
                if availability["status"] not in {"PROVIDER_EVIDENCE", "DECLARED_APPROXIMATION"} or available_at is None:
                    raise ValueError("schedule source availability is unresolved")
                if available_at < support_end:
                    raise ValueError("schedule source availability precedes its support end")
                if available_at > decision:
                    raise ValueError("schedule contains future or late-available support")
            if panel.mask.any() and not panel.lineage:
                raise ValueError("observed panel cells require source lineage")
            lineage_cells = set()
            for cell, source in panel.lineage.items():
                if not isinstance(cell, str) or "|" not in cell or not isinstance(source, Mapping):
                    raise ValueError("panel lineage requires explicit schedule-cell identities")
                day, channel = cell.split("|", 1)
                if day not in panel.dates or channel not in panel.channel_names:
                    raise ValueError("panel lineage cell is outside its declared schedule/schema")
                source_id = source.get("measurement_id")
                scheduled_source = supports.get(source_id)
                if scheduled_source is None:
                    raise ValueError("panel source is absent from the independently scheduled membership")
                scheduled_availability = _availability_record(scheduled_source)
                lineage_available = parse_utc_timestamp(source.get("available_at", ""))
                if (parse_utc_timestamp(source.get("support_start", "")) !=
                        parse_utc_timestamp(scheduled_source["support_start"]) or
                        parse_utc_timestamp(source.get("support_end", "")) !=
                        parse_utc_timestamp(scheduled_source["support_end"]) or
                        lineage_available != parse_utc_timestamp(scheduled_availability["at"])):
                        raise ValueError("panel source support differs from its scheduled evidence")
                if (parse_utc_timestamp(source["support_end"]) >= decision or
                        lineage_available > decision):
                    raise ValueError("panel lineage contains future or late-available support")
                lineage_cells.add((panel.dates.index(day), panel.channel_names.index(channel)))
            observed_cells = set(map(tuple, np.argwhere(panel.mask).tolist()))
            if not observed_cells <= lineage_cells:
                raise ValueError("observed panel cells lack declared source lineage")

            self.valid_panels.append(panel)

        if not self.valid_panels:
            raise ValueError(f"No valid panels found for role '{role}'")

    def __len__(self) -> int:
        return len(self.valid_panels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        panel = self.valid_panels[idx]

        if self.normalizer is not None:
            values_norm, mask = self.normalizer.transform(panel.values, panel.mask)
        else:
            values_norm = np.where(panel.mask, panel.values, 0.0).astype(np.float32)
            mask = panel.mask.copy()

        x_tensor = torch.from_numpy(values_norm.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(bool))
        meta = panel.to_dict()

        return x_tensor, mask_tensor, meta


def create_panel_dataloader(
    dataset: PanelDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Create a deterministic DataLoader with memory bounds [8, 16]."""
    # Enforce batch size bounds [8, 16] unless dataset is smaller than 8 in small tests
    if (batch_size < 8 or batch_size > 16) and len(dataset) >= 8:
        raise ValueError(
            f"batch_size must be between 8 and 16 to respect memory bounds, got {batch_size}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
        collate_fn=deterministic_collate,
        num_workers=num_workers,
    )

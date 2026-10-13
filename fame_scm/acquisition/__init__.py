"""Budgeted competence-evidence acquisition."""

from .scores import acquisition_scores, partition_remainder, select_batch

__all__ = ["acquisition_scores", "partition_remainder", "select_batch"]

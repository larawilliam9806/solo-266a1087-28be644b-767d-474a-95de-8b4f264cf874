"""Mutually-exclusive partition base table for one statistical period.

Stored cells overlap (ancestors and descendants), so summing them directly
double-counts.  This module derives, for every stored cell, its *nearest
stored descendants* — stored cells below it with no other stored cell in
between — and the residual population left after deducting them::

    remaining(cell) = pop(cell) - sum(pop(d) for d in nearest descendants)

The residual regions are mutually exclusive and together cover exactly the
union of all stored cells, so a statistical tool can sum the CSV rows
verbatim without double-counting.

Rules
-----
* one row per stored cell: ``cell, deducted ids (JSON array), remaining``;
  even if the residual region falls apart into several pieces it stays a
  single row and the population is never split;
* when the nearest descendants tile the cell completely, their populations
  must add up to the cell's population exactly, and the (now empty)
  residual row is omitted from the CSV;
* when they leave gaps, a negative remainder is a conflict;
* any conflict names its cell and aborts the whole export: no partial CSV.

Everything streams.  The finest cells are never enumerated (coverage is
checked arithmetically with ``4**k`` leaf counts) and neither the period's
populations nor the CSV text are materialised in memory: every cursor is
consumed row by row and each CSV row is emitted in small pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

from .archive import ArchiveError, BatchSnapshot, MAX_DEPTH

CSV_HEADER = "cell,deducted,remaining\r\n"


class PartitionConflict(ArchiveError):
    """One stored cell contradicts its nearest stored descendants."""

    def __init__(
        self,
        period: str,
        cell: str,
        population: int,
        deducted_population: int,
        fully_tiled: bool,
    ) -> None:
        self.period = period
        self.cell = cell
        self.population = population
        self.deducted_population = deducted_population
        self.remaining = population - deducted_population
        self.fully_tiled = fully_tiled
        if fully_tiled:
            detail = (
                f"cell {cell!r} is fully tiled by its nearest stored "
                f"descendants, but their populations sum to "
                f"{deducted_population}, not {population}"
            )
        else:
            detail = (
                f"cell {cell!r} has population {population}, less than the "
                f"{deducted_population} held by its nearest stored descendants"
            )
        super().__init__(f"period {period!r}: {detail}")


def _leaves(cell: str) -> int:
    """Number of finest cells below ``cell`` (counted, never enumerated)."""
    return 1 << (2 * (MAX_DEPTH - len(cell)))


def _nearest_descendants(
    snapshot: BatchSnapshot, cell: str
) -> Iterator[Tuple[str, int]]:
    """Yield the nearest stored descendants of ``cell`` in id order.

    A stored descendant is *nearest* when no other stored cell sits between
    it and ``cell``.  Descendants arrive in lexicographic order, so the
    scan only keeps the current chain of already reported ancestors: memory
    stays O(grid depth) no matter how wide the export is.
    """
    chain: List[str] = []
    for desc, pop in snapshot.iter_descendants(cell):
        while chain and not desc.startswith(chain[-1]):
            chain.pop()
        if chain:
            continue  # below an already reported nearest descendant
        yield desc, pop
        chain.append(desc)


def _deduction(snapshot: BatchSnapshot, cell: str) -> Tuple[int, int, int]:
    """Aggregate the nearest descendants: (population sum, leaf cover, count)."""
    pop_sum = 0
    leaf_sum = 0
    count = 0
    for desc, pop in _nearest_descendants(snapshot, cell):
        pop_sum += pop
        leaf_sum += _leaves(desc)
        count += 1
    return pop_sum, leaf_sum, count


@dataclass(frozen=True)
class PartitionStats:
    cell_count: int
    row_count: int
    total_population: int


def validate_partition(snapshot: BatchSnapshot) -> PartitionStats:
    """Check one batch and return the export's shape, streaming row by row.

    Raises :class:`PartitionConflict` naming the first inconsistent cell.
    Nothing is emitted before the whole batch is known to be consistent, so
    a rejected export produces no CSV at all.
    """
    cells = 0
    rows = 0
    total = 0
    for cell, population in snapshot.iter_cells():
        cells += 1
        pop_sum, leaf_sum, _count = _deduction(snapshot, cell)
        remaining = population - pop_sum
        if leaf_sum == _leaves(cell):
            # Fully tiled: the parts must account for the whole exactly, and
            # the empty residual row is omitted from the export.
            if remaining != 0:
                raise PartitionConflict(
                    snapshot.period, cell, population, pop_sum, fully_tiled=True
                )
            continue
        if remaining < 0:
            raise PartitionConflict(
                snapshot.period, cell, population, pop_sum, fully_tiled=False
            )
        rows += 1
        total += remaining
    return PartitionStats(cells, rows, total)


def emit_partition_csv(snapshot: BatchSnapshot) -> Iterator[str]:
    """Stream the partition CSV; call only after validation passes.

    Each row is re-derived from the same snapshot (so check and download
    always describe the same batch) and written in pieces, keeping memory
    bounded by the grid depth rather than by the batch size.
    """
    yield CSV_HEADER
    for cell, population in snapshot.iter_cells():
        pop_sum, leaf_sum, count = _deduction(snapshot, cell)
        if leaf_sum == _leaves(cell):
            continue  # empty residual region: row omitted
        remaining = population - pop_sum
        if count == 0:
            # No descendants: the whole row is one small chunk and no second
            # scan is needed.
            yield f"{cell},[],{remaining}\r\n"
            continue
        yield cell
        yield ',"['
        first = True
        for desc, _pop in _nearest_descendants(snapshot, cell):
            if first:
                first = False
            else:
                yield ","
            # Cell ids are digits 0-3 only, so doubling the CSV quotes is an
            # exact encoding of this JSON string array.
            yield '""'
            yield desc
            yield '""'
        yield ']",'
        yield str(remaining)
        yield "\r\n"

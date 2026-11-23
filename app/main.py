"""FastAPI service exposing the census archive.

Endpoints
---------
``POST /archive``                          submit (or atomically replace) one period's batch
``GET  /archive/{period}``                 batch metadata
``POST /census/query``                     minimal-read expression + union population
``GET  /archive/{period}/partition/check`` validate the mutually-exclusive partition table
``GET  /archive/{period}/partition.csv``   download the partition base table as CSV

The census endpoint works in two clearly separated phases inside one SQLite
read snapshot: first derive the expression from the period's catalog only, then
read headcounts for exactly the cells that appear in the expression.

The partition export validates the whole batch and streams the CSV inside one
SQLite read snapshot, so the check and the download always describe the same
batch even while another client replaces the period wholesale.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .archive import (
    Archive,
    BatchConflict,
    InvalidBatch,
    MissingCells,
    UnknownPeriod,
    validate_cell,
)
from .partition import PartitionConflict, emit_partition_csv, validate_partition
from .solver import MAX_DEPTH, analyze

MAX_CELLS_PER_REQUEST = 100_000
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "archive.sqlite3"


def default_database_path() -> str:
    return os.environ.get("CENSUS_DB", str(DEFAULT_DB))


def get_archive() -> Archive:
    return Archive(default_database_path())


def _check_cell(cell: str) -> str:
    try:
        return validate_cell(cell)
    except InvalidBatch as exc:
        # Re-raise as ValueError so pydantic reports it as a validation error.
        raise ValueError(str(exc)) from exc


class CellEntry(BaseModel):
    cell: str = Field(..., description='quadtree cell id; the root is ""')
    population: int = Field(..., ge=0)


class BatchSubmission(BaseModel):
    period: str = Field(..., min_length=1, description="statistical period label")
    replace: bool = Field(
        False,
        description="replace an existing batch atomically when true",
    )
    cells: List[CellEntry] = Field(
        default_factory=list, max_length=MAX_CELLS_PER_REQUEST
    )

    @field_validator("period")
    @classmethod
    def _period_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("period must not be blank")
        return value

    @field_validator("cells")
    @classmethod
    def _cells_valid(cls, value: List[CellEntry]) -> List[CellEntry]:
        seen = set()
        for entry in value:
            _check_cell(entry.cell)
            if entry.cell in seen:
                raise ValueError(f"duplicate cell id {entry.cell!r} in batch")
            seen.add(entry.cell)
        return value


class BatchResponse(BaseModel):
    period: str
    cell_count: int
    total_population: int
    created_at: str
    replaced: bool


class QueryRequest(BaseModel):
    period: str = Field(..., min_length=1)
    cells: List[str] = Field(..., max_length=MAX_CELLS_PER_REQUEST)

    @field_validator("cells")
    @classmethod
    def _cells_valid(cls, value: List[str]) -> List[str]:
        return [_check_cell(cell) for cell in value]


class ExpressionTermModel(BaseModel):
    cell: str
    coefficient: int


class CensusResponse(BaseModel):
    period: str
    cells: List[str]
    expression: str
    terms: List[ExpressionTermModel]
    retrieved_cells: int
    total_population: int


class MissingArchiveResponse(BaseModel):
    detail: str
    period: str
    missing_blocks: List[str]
    truncated: bool = False


class PartitionCheckResponse(BaseModel):
    period: str
    ok: bool
    cell_count: int
    row_count: int
    total_population: int
    created_at: str


app = FastAPI(
    title="Census Archive Service",
    description=(
        "Quadtree census archive. Submit whole statistical-period batches "
        "atomically and compute union populations with minimal-read expressions."
    ),
    version="1.0.0",
)


@app.exception_handler(UnknownPeriod)
async def _unknown_period_handler(request: Request, exc: UnknownPeriod) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "period": exc.period},
    )


@app.exception_handler(MissingCells)
async def _missing_cells_handler(request: Request, exc: MissingCells) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "period": exc.period,
            "missing_blocks": list(exc.cells),
            "truncated": False,
        },
    )


@app.exception_handler(PartitionConflict)
async def _partition_conflict_handler(
    request: Request, exc: PartitionConflict
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "period": exc.period,
            "cell": exc.cell,
            "population": exc.population,
            "deducted_population": exc.deducted_population,
            "remaining": exc.remaining,
            "fully_tiled": exc.fully_tiled,
        },
    )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/archive", response_model=BatchResponse, status_code=201)
def submit_batch(
    submission: BatchSubmission,
    archive: Archive = Depends(get_archive),
) -> BatchResponse:
    try:
        info = archive.replace_batch(
            submission.period,
            [(entry.cell, entry.population) for entry in submission.cells],
            allow_replace=submission.replace,
        )
    except BatchConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=f"{exc}; pass replace=true to replace it atomically",
        ) from exc
    except InvalidBatch as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchResponse(
        period=info.period,
        cell_count=info.cell_count,
        total_population=info.total_population,
        created_at=info.created_at,
        replaced=submission.replace,
    )


@app.get("/archive/{period}", response_model=BatchResponse)
def read_batch(period: str, archive: Archive = Depends(get_archive)) -> BatchResponse:
    info = archive.batch_info(period)
    return BatchResponse(
        period=info.period,
        cell_count=info.cell_count,
        total_population=info.total_population,
        created_at=info.created_at,
        replaced=False,
    )


@app.get("/archive/{period}/partition/check", response_model=PartitionCheckResponse)
def check_partition(
    period: str, archive: Archive = Depends(get_archive)
) -> PartitionCheckResponse:
    """Validate the partition table without downloading it.

    ``total_population`` is the sum of the ``remaining`` column the CSV
    would carry, i.e. the population of the union of all stored cells.
    """
    with archive.snapshot(period) as snapshot:
        stats = validate_partition(snapshot)
        return PartitionCheckResponse(
            period=period,
            ok=True,
            cell_count=stats.cell_count,
            row_count=stats.row_count,
            total_population=stats.total_population,
            created_at=snapshot.created_at,
        )


@app.get("/archive/{period}/partition.csv")
def download_partition(
    period: str, archive: Archive = Depends(get_archive)
) -> StreamingResponse:
    """Stream the mutually-exclusive partition base table as CSV.

    Validation and streaming share one SQLite read snapshot, so the CSV is
    checked against exactly the batch it is generated from; a conflicting
    batch yields ``409`` and no CSV at all.  The snapshot never blocks a
    concurrent wholesale replacement of the period.
    """
    stack = ExitStack()
    snapshot = stack.enter_context(archive.snapshot(period))
    try:
        validate_partition(snapshot)
    except BaseException:
        stack.close()
        raise

    def stream():
        try:
            yield from emit_partition_csv(snapshot)
        finally:
            stack.close()

    filename = quote(f"partition-{period}.csv")
    headers = {
        "Content-Disposition": (
            f"attachment; filename=\"partition.csv\"; filename*=UTF-8''{filename}"
        ),
        # Batch version, identical to the check endpoint's created_at.
        "X-Batch-Created-At": snapshot.created_at,
    }
    return StreamingResponse(stream(), media_type="text/csv", headers=headers)


@app.post("/census/query", response_model=CensusResponse)
def query_census(
    request: QueryRequest,
    archive: Archive = Depends(get_archive),
) -> CensusResponse:
    with archive.snapshot(request.period) as snapshot:
        # Phase 1: derive the minimal expression from the catalog alone.  No
        # headcount is read and no finest cell is enumerated.
        try:
            analysis = analyze(snapshot.catalog(), request.cells, depth=MAX_DEPTH)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not analysis.feasible:
            return _missing_response(request.period, analysis)
        solution = analysis.solution

        # Phase 2: read only the cells that occur in the expression.
        populations = snapshot.populations(term.cell for term in solution.terms)
        total = solution.evaluate(populations.__getitem__)

    return CensusResponse(
        period=request.period,
        cells=list(dict.fromkeys(request.cells)),
        expression=solution.render(),
        terms=[
            ExpressionTermModel(cell=term.cell, coefficient=term.coefficient)
            for term in solution.terms
        ],
        retrieved_cells=solution.retrieved_cells,
        total_population=total,
    )


def _missing_response(period: str, analysis) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=MissingArchiveResponse(
            detail=(
                "the archive for this period cannot support an exact "
                "expression for the requested cells; archive the listed "
                "missing blocks and retry"
            ),
            period=period,
            missing_blocks=list(analysis.missing_blocks),
            truncated=analysis.missing_truncated,
        ).model_dump(),
    )

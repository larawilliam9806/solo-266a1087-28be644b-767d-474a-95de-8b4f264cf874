import csv
import io
import itertools
import json
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.archive import Archive
from app.main import app, get_archive

MAX_DEPTH = 12


@pytest.fixture()
def archive(tmp_path):
    return Archive(tmp_path / "archive.sqlite3")


@pytest.fixture()
def client(archive):
    app.dependency_overrides[get_archive] = lambda: archive
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def submit(client, period, cells, replace=False):
    return client.post(
        "/archive",
        json={
            "period": period,
            "replace": replace,
            "cells": [{"cell": c, "population": p} for c, p in cells],
        },
    )


def parse_csv(text):
    """Parse the exported CSV into [(cell, deducted list, remaining)]."""
    reader = csv.reader(io.StringIO(text, newline=""))
    rows = list(reader)
    assert rows, "CSV must at least carry a header"
    assert rows[0] == ["cell", "deducted", "remaining"]
    return [
        (cell, json.loads(deducted), int(remaining))
        for cell, deducted, remaining in rows[1:]
    ]


def reference_partition(cells, depth=MAX_DEPTH):
    """Independent in-memory reference implementation.

    Returns ("ok", rows) with rows = [(cell, deducted, remaining)] in id
    order, or ("conflict", cell) for the first conflicting cell in id order.
    """

    def leaves(cell):
        return 1 << (2 * (depth - len(cell)))

    ids = sorted(cells)
    rows = []
    for cell in ids:
        descendants = [d for d in ids if len(d) > len(cell) and d.startswith(cell)]
        nearest = [
            d
            for d in descendants
            if not any(d != other and d.startswith(other) for other in descendants)
        ]
        deducted = sum(cells[d] for d in nearest)
        cover = sum(leaves(d) for d in nearest)
        remaining = cells[cell] - deducted
        if cover == leaves(cell):
            if remaining != 0:
                return ("conflict", cell)
            continue  # fully tiled: empty residual row omitted
        if remaining < 0:
            return ("conflict", cell)
        rows.append((cell, nearest, remaining))
    return ("ok", rows)


def test_fully_tiled_parent_row_is_omitted(client):
    submit(client, "tiled", [("", 100), ("0", 25), ("1", 25), ("2", 25), ("3", 25)])
    resp = client.get("/archive/tiled/partition.csv")
    assert resp.status_code == 200, resp.text
    rows = parse_csv(resp.text)
    # The root's residual region is empty: its row is omitted entirely.
    assert rows == [("0", [], 25), ("1", [], 25), ("2", [], 25), ("3", [], 25)]
    assert sum(remaining for _, _, remaining in rows) == 100


def test_nearest_descendants_skip_intermediate_stored_cells(client):
    submit(client, "nest", [("", 100), ("0", 40), ("00", 10)])
    resp = client.get("/archive/nest/partition.csv")
    assert resp.status_code == 200, resp.text
    # "00" is deducted from "0", not from "": only the nearest stored
    # descendants (no other stored cell in between) appear in each row.
    assert parse_csv(resp.text) == [
        ("", ["0"], 60),
        ("0", ["00"], 30),
        ("00", [], 10),
    ]


def test_gap_with_zero_remaining_still_emits_row(client):
    submit(client, "gap0", [("", 100), ("0", 60), ("1", 40)])
    resp = client.get("/archive/gap0/partition.csv")
    assert resp.status_code == 200, resp.text
    # Not fully tiled (only two of four quarters covered), so the row stays
    # even though the residual population happens to be zero.
    assert parse_csv(resp.text)[0] == ("", ["0", "1"], 0)


def test_split_residual_region_is_one_row(client):
    # The residual of "" is everything except "00" and "01": several
    # disconnected pieces, but exactly one row and the population is not split.
    submit(client, "split", [("", 100), ("00", 10), ("01", 20)])
    rows = parse_csv(client.get("/archive/split/partition.csv").text)
    assert rows == [("", ["00", "01"], 70), ("00", [], 10), ("01", [], 20)]


def test_root_cell_only_batch(client):
    submit(client, "root", [("", 1234)])
    rows = parse_csv(client.get("/archive/root/partition.csv").text)
    assert rows == [("", [], 1234)]


def test_empty_batch_exports_header_only(client):
    submit(client, "empty", [])
    resp = client.get("/archive/empty/partition.csv")
    assert resp.status_code == 200, resp.text
    assert parse_csv(resp.text) == []
    check = client.get("/archive/empty/partition/check").json()
    assert check["row_count"] == 0
    assert check["total_population"] == 0
    assert check["cell_count"] == 0


def test_tiled_sum_mismatch_is_conflict(client):
    submit(client, "mis", [("", 100), ("0", 25), ("1", 25), ("2", 25), ("3", 24)])
    resp = client.get("/archive/mis/partition.csv")
    assert resp.status_code == 409
    body = resp.json()
    assert body["cell"] == ""
    assert body["fully_tiled"] is True
    assert body["deducted_population"] == 99


def test_gap_with_negative_remaining_is_conflict(client):
    submit(client, "neg", [("", 100), ("0", 60), ("1", 60)])
    resp = client.get("/archive/neg/partition.csv")
    assert resp.status_code == 409
    body = resp.json()
    assert body["cell"] == ""
    assert body["fully_tiled"] is False
    assert body["remaining"] == -20


def test_conflict_points_at_the_deeper_cell(client):
    submit(
        client,
        "deep-bad",
        [("", 200), ("0", 50), ("1", 50), ("2", 50), ("3", 50), ("00", 30), ("01", 30)],
    )
    resp = client.get("/archive/deep-bad/partition.csv")
    assert resp.status_code == 409
    assert resp.json()["cell"] == "0"


def test_conflict_emits_no_csv(client):
    submit(client, "bad", [("", 100), ("0", 60), ("1", 60)])
    resp = client.get("/archive/bad/partition.csv")
    assert resp.status_code == 409
    # The whole CSV is withheld: the response is the JSON error, not a
    # partial stream.
    assert resp.headers["content-type"].startswith("application/json")
    check = client.get("/archive/bad/partition/check")
    assert check.status_code == 409
    assert check.json()["cell"] == ""


def test_unknown_period(client):
    assert client.get("/archive/nope/partition.csv").status_code == 404
    assert client.get("/archive/nope/partition/check").status_code == 404


def test_check_endpoint_reports_shape(client):
    submit(client, "chk", [("", 100), ("0", 40), ("00", 10)])
    resp = client.get("/archive/chk/partition/check")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["cell_count"] == 3
    assert body["row_count"] == 3
    assert body["total_population"] == 100
    assert body["created_at"]


def test_csv_response_headers(client):
    submit(client, "hdr", [("0", 1)])
    created_at = client.get("/archive/hdr/partition/check").json()["created_at"]
    resp = client.get("/archive/hdr/partition.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    # Check and download describe the same batch version.
    assert resp.headers["x-batch-created-at"] == created_at


def test_deep_cells_export_without_expansion(client):
    deep = "012301230123"
    parent = "0123"
    submit(client, "deep", [(parent, 50), (deep, 5)])
    resp = client.get("/archive/deep/partition.csv")
    assert resp.status_code == 200, resp.text
    assert parse_csv(resp.text) == [(parent, [deep], 45), (deep, [], 5)]


def test_mixed_depth_tiling(client):
    # "01" is refined into its four children: together with "00", "02", "03"
    # they still tile "0" exactly, so "0"'s row is omitted.
    submit(
        client,
        "mixed",
        [("0", 100), ("00", 25), ("010", 6), ("011", 6), ("012", 6), ("013", 7), ("02", 25), ("03", 25)],
    )
    rows = parse_csv(client.get("/archive/mixed/partition.csv").text)
    assert ("0", ["00", "010", "011", "012", "013", "02", "03"], 0) not in rows
    assert [cell for cell, _, _ in rows] == ["00", "010", "011", "012", "013", "02", "03"]
    assert sum(remaining for _, _, remaining in rows) == 100


def test_zero_population_cell_still_emits_row(client):
    submit(client, "zero", [("0", 0), ("1", 5)])
    rows = parse_csv(client.get("/archive/zero/partition.csv").text)
    assert rows == [("0", [], 0), ("1", [], 5)]


def test_random_batches_match_reference(client):
    rng = random.Random(20261123)
    for i in range(150):
        cells = {}
        for _ in range(rng.randint(0, 20)):
            cell = "".join(rng.choice("0123") for _ in range(rng.randint(0, 4)))
            cells[cell] = rng.randint(0, 600)
        period = f"prop-{i}"
        assert submit(client, period, sorted(cells.items())).status_code == 201

        outcome = reference_partition(cells)
        resp = client.get(f"/archive/{period}/partition.csv")
        if outcome[0] == "conflict":
            assert resp.status_code == 409, (cells, resp.text)
            assert resp.json()["cell"] == outcome[1]
            continue

        assert resp.status_code == 200, (cells, resp.text)
        rows = parse_csv(resp.text)
        assert rows == outcome[1]

        check = client.get(f"/archive/{period}/partition/check").json()
        assert check["cell_count"] == len(cells)
        assert check["row_count"] == len(rows)
        assert check["total_population"] == sum(remaining for _, _, remaining in rows)


def test_leaf_consistent_batches_sum_to_union_population(client):
    """Batches derived from finest-cell populations never conflict, and the
    exported rows add up to the population of the union of all stored cells."""
    rng = random.Random(7)
    depth = 3
    leaves = ["".join(t) for t in itertools.product("0123", repeat=depth)]
    all_cells = [""]
    for length in range(1, depth + 1):
        all_cells += ["".join(t) for t in itertools.product("0123", repeat=length)]

    for i in range(40):
        leaf_pops = {leaf: rng.randint(0, 9) for leaf in leaves}

        def pop_of(cell):
            return sum(p for leaf, p in leaf_pops.items() if leaf.startswith(cell))

        chosen = [c for c in all_cells if rng.random() < 0.35]
        cells = {c: pop_of(c) for c in chosen}
        period = f"leaf-{i}"
        assert submit(client, period, sorted(cells.items())).status_code == 201

        resp = client.get(f"/archive/{period}/partition.csv")
        assert resp.status_code == 200, resp.text
        rows = parse_csv(resp.text)
        covered = [
            leaf for leaf in leaves if any(leaf.startswith(c) for c in chosen)
        ]
        expected = sum(leaf_pops[leaf] for leaf in covered)
        assert sum(remaining for _, _, remaining in rows) == expected
        # Residual regions are disjoint: no deducted cell may also appear as
        # the residual of another row's region.  Every stored cell appears as
        # a row at most once by construction; deducted ids must be exactly the
        # stored cells whose row region sits below another stored cell.


def test_replace_during_export_keeps_consistent_snapshot(client, archive):
    """A wholesale replacement mid-export neither blocks nor mixes batches."""
    submit(client, "swap", [("0", 10), ("1", 20), ("2", 30), ("3", 40)])
    with client.stream("GET", "/archive/swap/partition.csv") as resp:
        assert resp.status_code == 200
        lines = resp.iter_lines()
        assert next(lines) == "cell,deducted,remaining"
        # Replace the whole period while the export is streaming.
        r = submit(client, "swap", [("", 999)], replace=True)
        assert r.status_code == 201, r.text
        body = "cell,deducted,remaining\n" + "\n".join(lines)
    # The stream still describes the batch it was validated against.
    assert parse_csv(body) == [
        ("0", [], 10),
        ("1", [], 20),
        ("2", [], 30),
        ("3", [], 40),
    ]
    # The replacement did go through for later readers.
    assert client.get("/archive/swap").json()["cell_count"] == 1
    assert parse_csv(client.get("/archive/swap/partition.csv").text) == [("", [], 999)]


def test_replacement_changes_export(client):
    submit(client, "rep", [("0", 10), ("1", 20)])
    assert parse_csv(client.get("/archive/rep/partition.csv").text) == [
        ("0", [], 10),
        ("1", [], 20),
    ]
    submit(client, "rep", [("", 70), ("0", 10)], replace=True)
    assert parse_csv(client.get("/archive/rep/partition.csv").text) == [
        ("", ["0"], 60),
        ("0", [], 10),
    ]

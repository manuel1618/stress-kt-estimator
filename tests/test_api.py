from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_LOAD_CASES = [
    {"case_name": "LC1", "fx": 100, "fy": 0, "fz": 0, "mx": 0, "my": 0, "mz": 0, "stress": 100},
    {"case_name": "LC2", "fx": 0, "fy": 100, "fz": 0, "mx": 0, "my": 0, "mz": 0, "stress": 100},
    {"case_name": "LC3", "fx": 40, "fy": 40, "fz": 0, "mx": 0, "my": 0, "mz": 0, "stress": 90},
    {"case_name": "LC4", "fx": -100, "fy": 0, "fz": 0, "mx": 0, "my": 0, "mz": 0, "stress": -80},
]


class TestHealth:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestSolve:
    def test_solve_basic(self):
        resp = client.post("/api/solve", json={
            "load_cases": SAMPLE_LOAD_CASES,
            "settings": {"use_separate_sign": True},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["kt_names"]) == 12
        assert len(data["kt_values"]) == 12
        assert len(data["per_case"]) == len(SAMPLE_LOAD_CASES)

    def test_solve_linked(self):
        resp = client.post("/api/solve", json={
            "load_cases": SAMPLE_LOAD_CASES,
            "settings": {"use_separate_sign": False},
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_solve_empty_load_cases_rejected(self):
        resp = client.post("/api/solve", json={
            "load_cases": [],
        })
        assert resp.status_code == 422

    def test_solve_with_safety_factor(self):
        resp = client.post("/api/solve", json={
            "load_cases": SAMPLE_LOAD_CASES,
            "settings": {"safety_factor": 1.5},
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_solve_invalid_safety_factor(self):
        resp = client.post("/api/solve", json={
            "load_cases": SAMPLE_LOAD_CASES,
            "settings": {"safety_factor": 0},
        })
        assert resp.status_code == 422


class TestRecalc:
    def test_recalc_basic(self):
        kt_values = [1.0, 0.8, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        resp = client.post("/api/recalc", json={
            "load_cases": SAMPLE_LOAD_CASES,
            "settings": {"use_separate_sign": True},
            "kt_values": kt_values,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["kt_values"]) == 12

    def test_recalc_wrong_kt_count(self):
        resp = client.post("/api/recalc", json={
            "load_cases": SAMPLE_LOAD_CASES,
            "settings": {},
            "kt_values": [1.0, 2.0],
        })
        assert resp.status_code == 422


class TestSuggestUnlink:
    def test_suggest_unlink(self):
        cases = SAMPLE_LOAD_CASES + [
            {"case_name": "LC5", "fx": 0, "fy": -100, "fz": 0, "mx": 0, "my": 0, "mz": 0, "stress": 60},
        ]
        resp = client.post("/api/suggest-unlink", json={"load_cases": cases})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["suggested_components"], list)

    def test_suggest_unlink_too_few_cases(self):
        resp = client.post("/api/suggest-unlink", json={
            "load_cases": [SAMPLE_LOAD_CASES[0]],
        })
        assert resp.status_code == 422


class TestFindMinimalUnlink:
    def test_find_minimal_unlink(self):
        resp = client.post("/api/find-minimal-unlink", json={
            "load_cases": SAMPLE_LOAD_CASES,
            "settings": {"use_separate_sign": True},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["sign_modes"], list)
        assert len(data["sign_modes"]) == 6
        if data["result"] is not None:
            assert "kt_names" in data["result"]

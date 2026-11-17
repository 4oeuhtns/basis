"""The tearsheet is a self-contained page built only from saved results."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from qflab.ui.tearsheet import page_data, scientific_checks, write_tearsheet

MANIFEST = {
    "run_id": "test-run",
    "status": "completed",
    "seed": 1,
    "quantum_target": "qpp-cpu",
    "workers": 1,
    "config_sha256": "0" * 64,
    "started": "2026-01-01T00:00:00+00:00",
    "environment": {
        "packages": {"cudaq": "0.17.0", "numpy": "2", "scipy": "1"},
        "python": "3.12",
        "platform": "test",
        "cpu_count": 1,
        "git_commit": None,
        "git_dirty": False,
    },
    "sections": {"risk_lab": {"status": "failed", "error": "boom", "traceback": "secret path"}},
}


def risk_lab(coverage=0.95, matches=True):
    return {
        "reference": {"probability_mass": 1.0, "scaling_round_trip_error": 0.0},
        "circuit_verification": [{"matches": matches}, {"matches": True}],
        "grid_sweep": [{"circuit_matches_discrete_sum": True}],
        "benchmark": [
            {"method": "iqae", "coverage": coverage, "seeds": 30},
            {"method": "monte_carlo", "coverage": 0.1, "seeds": 30},
        ],
        "settings": {"confidence": 0.95},
    }


def portfolio(reconciles=True, differences=("optimizer.solver",)):
    run = {
        "cash_reconciliation": {"reconciles": reconciles, "largest_difference": 0.0},
        "decisions": [{"status": "feasible", "selected": ["A", "B", "C"]}],
        "fallbacks": 0,
        "config": {"big": True},
    }
    return {
        "runs": {"exact": run, "qaoa": dict(run), "benchmark": dict(run)},
        "pairing": {
            "differs_only_in_method": differences == ("optimizer.solver",),
            "same_snapshot": True,
            "config_differences": list(differences),
        },
    }


def status(checks, fragment):
    return next(c["status"] for c in checks if fragment in c["name"])


def test_checks_report_what_this_run_measured_and_what_it_did_not():
    checks = scientific_checks({"risk_lab": risk_lab(), "portfolio": portfolio()})
    assert status(checks, "probability mass") == "pass"
    assert status(checks, "Indicator probability") == "pass"
    assert status(checks, "IQAE confidence") == "pass"
    assert status(checks, "self-financing") == "pass"
    assert status(checks, "intended method only") == "pass"
    # Honest about what is not re-measured here, and about what is not satisfied at all.
    assert status(checks, "Mutating future data") == "tests"
    assert status(checks, "Final evaluation dates") == "open"


def test_checks_fail_or_go_missing_instead_of_passing_by_default():
    broken = scientific_checks(
        {
            "risk_lab": risk_lab(coverage=0.5, matches=False),
            "portfolio": portfolio(reconciles=False, differences=("fee_bps", "optimizer.solver")),
        }
    )
    for fragment in (
        "Indicator probability",
        "IQAE confidence",
        "self-financing",
        "intended method only",
    ):
        assert status(broken, fragment) == "fail", fragment
    # Coverage slightly under nominal is within sampling error for 30 seeds.
    assert status(scientific_checks({"risk_lab": risk_lab(coverage=0.9)}), "IQAE") == "pass"
    absent = scientific_checks({"risk_lab": {"status": "failed", "error": "x"}})
    assert status(absent, "probability mass") == "missing"
    assert status(absent, "self-financing") == "missing"


def test_page_is_self_contained_and_keeps_embedded_data_inert(tmp_path):
    results = {
        "risk_lab": {"status": "failed", "error": "</script><script>alert(1)</script>"},
        "portfolio": portfolio(),
    }
    data = page_data(results, MANIFEST, {"name": "x"})
    assert data["available"] == {
        "risk_lab": False,
        "portfolio": True,
        "qaoa_study": False,
        "risk_forecast": False,
    }
    assert "traceback" not in data["manifest"]["sections"]["risk_lab"]
    assert "config" not in data["results"]["portfolio"]["runs"]["exact"]
    assert "config" in results["portfolio"]["runs"]["exact"]  # the saved results are untouched
    path = write_tearsheet(tmp_path / "tearsheet.html", results, MANIFEST, {"name": "x"})
    html = path.read_text()
    assert "/*__DATA__*/null" not in html and "test-run" in html
    assert html.count("</script>") == 2  # only the page's own two script elements close
    assert not re.search(r'(src|href)="https?://', html)  # nothing is fetched
    embedded = html.split("const DATA = ", 1)[1].split(";</script>", 1)[0]
    assert json.loads(embedded.replace("<\\/", "</"))["manifest"]["run_id"] == "test-run"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to parse the script")
def test_page_script_parses(tmp_path):
    path = write_tearsheet(tmp_path / "t.html", {}, MANIFEST, {})
    scripts = re.findall(r"<script>(.*?)</script>", path.read_text(), flags=re.S)
    assert len(scripts) == 2
    source = tmp_path / "page.js"
    source.write_text("\n".join(scripts))
    done = subprocess.run(["node", "--check", str(source)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert Path(path).stat().st_size > 20_000

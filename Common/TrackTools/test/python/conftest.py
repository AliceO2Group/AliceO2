"""Standalone O2 TrackTools PyROOT test setup; no dEdxAI/ADF imports."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def cpp():
    try:
        import ROOT
    except ImportError:
        pytest.fail("PyROOT is required: run within an O2/ROOT environment", pytrace=False)

    include = Path(os.environ.get("TRACKTOOLS_INCLUDE", Path(__file__).resolve().parents[2] / "include"))
    if not (include / "TrackGeometry.h").is_file():
        pytest.fail(f"Missing TrackGeometry.h under {include}", pytrace=False)
    ROOT.gInterpreter.AddIncludePath(str(include))
    try:
        ok = ROOT.gInterpreter.Declare('#include "TrackGeometry.h"')
    except Exception as exc:
        pytest.fail(f"Cannot compile TrackGeometry.h with current O2 environment: {exc}", pytrace=False)
    if not ok:
        pytest.fail("Cling rejected TrackGeometry.h; verify O2 include paths and dependencies", pytrace=False)
    return ROOT.o2

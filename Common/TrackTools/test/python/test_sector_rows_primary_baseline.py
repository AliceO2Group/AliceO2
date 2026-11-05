"""Current TPC H/L row-accounting baseline, explicitly REDUCED PRIMARY ONLY.

No ADF, dfdraw, local dEdxAI Python package or NumPy required. This test does
NOT exercise the missing full-5D secondary-track TPC row evaluator.
"""

import math

import pytest


@pytest.fixture
def tpc(cpp):
    return cpp.tpc.recohelpers


def _seed(ns, phi=2.15374, tgl=-0.814225, qpt=0.896099, bz=5.):
    s = ns.TPCPrimaryTrackSeed()
    s.phi, s.tgl, s.qpt, s.vz, s.bzKG = phi, tgl, qpt, 0., bz
    return s


def _assert_partition(result):
    crossed = int(result.nRowsCrossed)
    assert crossed == int(result.nRowsDead) + int(result.nRowsExcluded) + int(result.nRowsAvailable)
    assert 0 <= crossed <= 152
    # These are bitsets over exactly the 152 global TPC rows.
    counts = []
    for mask in (result.crossedMask, result.deadMask, result.excludedMask, result.availableMask):
        rows = [i for i in range(152) if (int(mask[i >> 5]) >> (i & 31)) & 1]
        counts.append(set(rows))
    crossed_set, dead_set, excluded_set, available_set = counts
    assert len(crossed_set) == crossed
    assert (dead_set | excluded_set | available_set) == crossed_set
    assert not (dead_set & excluded_set or dead_set & available_set or excluded_set & available_set)
    assert [len(x) for x in counts] == [int(result.nRowsCrossed), int(result.nRowsDead),
                                        int(result.nRowsExcluded), int(result.nRowsAvailable)]
    length_sum = float(result.lengthDeadCm + result.lengthExcludedCm + result.lengthAvailableCm)
    assert float(result.lengthCrossedCm) == pytest.approx(length_sum, abs=2e-4)


def test_mapper_has_152_rows(cpp):
    assert int(cpp.tpc.Mapper.instance().getNumberOfRows()) == 152


@pytest.mark.parametrize("method", ("sectorEdgeHelixRows", "sectorEdgeStackLinear"))
def test_primary_h_l_partition(tpc, method):
    cfg = tpc.TPCSectorEdgeConfig()
    cfg.guardLowerCm, cfg.guardUpperCm = 1., 1.
    result = getattr(tpc, method)(_seed(tpc), cfg)
    assert int(result.status) == 0, f"{method} status={int(result.status)}"
    assert int(result.nRowsCrossed) > 0
    _assert_partition(result)


def test_primary_guard_monotonicity(tpc):
    results = []
    for guard in (0., 0.5, 1.):
        cfg = tpc.TPCSectorEdgeConfig()
        cfg.guardLowerCm = cfg.guardUpperCm = guard
        result = tpc.sectorEdgeHelixRows(_seed(tpc), cfg)
        assert int(result.status) == 0
        _assert_partition(result)
        results.append(result)
    assert len({int(r.nRowsCrossed) for r in results}) == 1
    assert len({int(r.nRowsDead) for r in results}) == 1
    assert [int(r.nRowsExcluded) for r in results] == sorted(int(r.nRowsExcluded) for r in results)
    assert [int(r.nRowsAvailable) for r in results] == sorted(
        (int(r.nRowsAvailable) for r in results), reverse=True)


@pytest.mark.xfail(strict=True, reason="F6: reviewed rowRadius uses inner edge; pad-centre oracle not yet certified")
def test_f6_row_centre_expected_but_not_yet_implemented(cpp):
    """Known geometry blocker; expected failure until physical pad centres are verified."""
    tpc = cpp.tpc.recohelpers
    mapper = cpp.tpc.Mapper.instance()
    r0 = mapper.getPadRegionInfo(0)
    expected = float(r0.getRadiusFirstRow()) + 0.5 * float(r0.getPadHeight())
    assert float(tpc.rowRadius(mapper, 0)) == pytest.approx(expected, abs=1e-5)

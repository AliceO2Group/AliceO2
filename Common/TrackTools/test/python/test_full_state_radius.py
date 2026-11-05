"""Full O2 TrackPar (x, alpha, y, z, snp, tgl, qpt) state-at-radius tests.

Source API exists in the 2026-10-22 TrackGeometry.h snapshot. These are
regression checks, not certification of the TPC full-state row API.
"""

import math

import pytest


@pytest.mark.parametrize("method", ("H", "K0"))
def test_full_5d_reaches_radius(cpp, method):
    reco = cpp.recohelpers
    track = reco.makeTrackParFromLocalParameters(85., 0.10, 2.0, 17., 0.04, 0.10, 0.50)
    state = getattr(reco, f"getParametersAtRadius{method}")(track, 100., 5.)
    assert state.valid(), f"{method} status={int(state.status)}"
    assert math.hypot(float(state.gx), float(state.gy)) == pytest.approx(100., abs=0.02)
    d_sector = float(reco.stateDSector(state))
    assert 0 <= d_sector < 1


@pytest.mark.parametrize("x,alpha,y,z,snp,tgl,qpt,radius", [
    (80., 0.20, 4., 17., 0.12, 0.23, 0.40, 100.),
    (140., -0.40, 12., 68., -0.15, 0.40, -0.60, 180.),
    (205., 0.35, -8., 190., 0.08, -0.20, 0.25, 230.),
])
def test_displaced_secondary_full_state_zero_field_analytic_oracle(
        cpp, x, alpha, y, z, snp, tgl, qpt, radius):
    """A non-vertex track keeps y,z,alpha and has the correct straight-line intersection.

    At zero Bz the analytic H transport has an independent geometry oracle.
    The test will fail if the implementation silently resets to a primary seed.
    """
    reco = cpp.recohelpers
    track = reco.makeTrackParFromLocalParameters(x, alpha, y, z, snp, tgl, qpt)
    state = reco.getParametersAtRadiusH(track, radius, 0.)
    assert state.valid(), f"secondary H status={int(state.status)}"

    # Global position from the local O2 reference plane and momentum direction.
    x0 = x * math.cos(alpha) - y * math.sin(alpha)
    y0 = x * math.sin(alpha) + y * math.cos(alpha)
    theta = alpha + math.asin(snp)
    ux, uy = math.cos(theta), math.sin(theta)
    projection = x0 * ux + y0 * uy
    sxy = -projection + math.sqrt(projection**2 + radius**2 - x0**2 - y0**2)
    expected = (x0 + sxy * ux, y0 + sxy * uy, z + tgl * sxy)
    for actual, target in zip((state.gx, state.gy, state.gz), expected):
        assert float(actual) == pytest.approx(target, abs=2e-3)
    assert 0 <= float(reco.stateDSector(state)) < 1

    # Explicitly demonstrate that resetting this secondary to a primary changes the state.
    primary = reco.makePrimaryTrackPar(theta, tgl, qpt, 0.)
    primary_state = reco.getParametersAtRadiusH(primary, radius, 0.)
    assert primary_state.valid()
    delta = math.sqrt(sum((float(a) - float(b))**2 for a, b in zip(
        (state.gx, state.gy, state.gz),
        (primary_state.gx, primary_state.gy, primary_state.gz))))
    assert delta > 1.0  # cm, intentionally separated reference states


@pytest.mark.parametrize("qpt,bz", ((0.40, 5.), (-0.40, -5.)))
def test_signed_field_and_charge_explicit(cpp, qpt, bz):
    reco = cpp.recohelpers
    trk = reco.makeTrackParFromLocalParameters(85., 0.12, 1.2, 7., 0.05, 0.15, qpt)
    out = reco.getParametersAtRadiusH(trk, 100., bz)
    assert out.valid(), f"signed-field H status={int(out.status)}"
    assert math.hypot(float(out.gx), float(out.gy)) == pytest.approx(100., abs=0.02)


def test_invalid_radius_status(cpp):
    reco = cpp.recohelpers
    track = reco.makeTrackParFromLocalParameters(85., 0., 0., 0., 0.02, 0.1, 0.5)
    assert not reco.getParametersAtRadiusH(track, -1., 5.).valid()

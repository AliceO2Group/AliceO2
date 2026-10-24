# O2 TrackTools — Step 0 portable PyROOT tests

These tests execute the **existing** C++ `TrackGeometry.h` in a ROOT/O2 environment. They deliberately do not import `AliasDataFrame`, `dfdraw`, `gbregression`, or the development-repository `o2reco` package. Python dependencies: `pytest` and `ROOT` (PyROOT supplied by the matching O2/ROOT environment).

Run from `O2/Common/TrackTools` after configuring the O2 environment:

```sh
python -m pytest -q -ra test/python
```

If the source header is elsewhere, set `TRACKTOOLS_INCLUDE` to the include-root directory containing `TrackTools/TrackGeometry.h`. The test fixture **fails** if ROOT or the header cannot be loaded; no environment failure is counted as PASS.

Current cases:

- Full 5D `TrackPar` scalar H/K0 reaches the requested radius and yields a defined `dSector`.
- **Nonprimary/displaced 5D H** at `Bz=0` is independently checked against an analytic straight-line/cylinder intersection at three radii, including an outer-TPC secondary. Detects silent resetting to a primary vertex.
- Signed field/charge, explicit field argument, invalid-radius status.
- Current **reduced-primary-only** TPC H/L row classification: 152-bit partition identities, guard monotonicity and basic Mapper cardinality.
- F6 physical row centre is an **expected failure (XFAIL)** pending the Mapper oracle and code fix; it is not treated as a successful geometry test.

## Outstanding—not certified by Step 0

- Extended full-5D **TPC H/L/K** row/sector-edge tests for displaced secondary tracks are not present in this portable O2 test subset. Add and validate the separate full-state tests before claiming complete coverage.
- K field-map / high-curvature regressions (six previous red cases); do not widen tolerances or convert them to PASS.
- Full 152-row Mapper pad-centre oracle, native C++ unit test, O2 CTest registration, real-data parity.

**Never claim that these tests alone validate production secondary-track TPC geometry or all K propagation.** Record the current header SHA256, O2 commit, ROOT/O2 environment and full pytest transcript for the PR.

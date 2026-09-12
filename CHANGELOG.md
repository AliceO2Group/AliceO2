# Changes since 2026-08-29

## Changes in Analysis

- [#15735](https://github.com/AliceO2Group/AliceO2/pull/15735) 2026-08-29: Track skipped read timeframes in file statistics & dumping by [@autumn-mck](https://github.com/autumn-mck)
- [#15742](https://github.com/AliceO2Group/AliceO2/pull/15742) 2026-09-01: Analysis CCDB: add ability to have a uniformity column in a CCDB table by [@ktf](https://github.com/ktf)
- [#15744](https://github.com/AliceO2Group/AliceO2/pull/15744) 2026-09-03: DPL Analysis: allow lookup of paths based on the run / uniformity by [@ktf](https://github.com/ktf)
- [#15757](https://github.com/AliceO2Group/AliceO2/pull/15757) 2026-09-04: DPL CCDB Analysis: add ability to specify run dependent queries by [@ktf](https://github.com/ktf)
- [#15758](https://github.com/AliceO2Group/AliceO2/pull/15758) 2026-09-04: DPL: allow plugins to account for bytes by [@ktf](https://github.com/ktf)
## Changes in Common

- [#15784](https://github.com/AliceO2Group/AliceO2/pull/15784) 2026-09-11: Regularization fixes + switch for the pre-PR-15610 compatibility mode by [@shahor02](https://github.com/shahor02)
## Changes in DataFormats

- [#15705](https://github.com/AliceO2Group/AliceO2/pull/15705) 2026-08-29: MC: Ability to process empty timeframes (Part 1) by [@sawenzel](https://github.com/sawenzel)
- [#15762](https://github.com/AliceO2Group/AliceO2/pull/15762) 2026-09-04: Fix final so that new clang does not complain by [@ktf](https://github.com/ktf)
- [#15775](https://github.com/AliceO2Group/AliceO2/pull/15775) 2026-09-08: Remove hardcoded CCDB path by [@ktf](https://github.com/ktf)
- [#15777](https://github.com/AliceO2Group/AliceO2/pull/15777) 2026-09-09: [ALICE3] TF3: store BC and TDC in digits by [@maciacco](https://github.com/maciacco)
- [#15785](https://github.com/AliceO2Group/AliceO2/pull/15785) 2026-09-11: [MUON] Fix assignment of GlobalFwdTrack from base class by [@aferrero2707](https://github.com/aferrero2707)
## Changes in Detectors

- [#15714](https://github.com/AliceO2Group/AliceO2/pull/15714) 2026-08-29: MC-RECO: Ability to process empty timeframes (Part 2) by [@sawenzel](https://github.com/sawenzel)
- [#15705](https://github.com/AliceO2Group/AliceO2/pull/15705) 2026-08-29: MC: Ability to process empty timeframes (Part 1) by [@sawenzel](https://github.com/sawenzel)
- [#15740](https://github.com/AliceO2Group/AliceO2/pull/15740) 2026-08-31: Add more info on MC vertex by [@shahor02](https://github.com/shahor02)
- [#15739](https://github.com/AliceO2Group/AliceO2/pull/15739) 2026-08-31: Fix a misplaced prepreg strip in the TPC inner field cage by [@sawenzel](https://github.com/sawenzel)
- [#15716](https://github.com/AliceO2Group/AliceO2/pull/15716) 2026-08-31: Replace the TPC half-space cuts by bounded boxes by [@sawenzel](https://github.com/sawenzel)
- [#15746](https://github.com/AliceO2Group/AliceO2/pull/15746) 2026-09-01: MFT: Fix negative capacity tori by [@sawenzel](https://github.com/sawenzel)
- [#15738](https://github.com/AliceO2Group/AliceO2/pull/15738) 2026-09-01: TOF: Remove the MANY TGeo placements by [@sawenzel](https://github.com/sawenzel)
- [#15750](https://github.com/AliceO2Group/AliceO2/pull/15750) 2026-09-02: Add a reachability check to the geometry doctor and restore the ZEM geometry by [@sawenzel](https://github.com/sawenzel)
- [#15747](https://github.com/AliceO2Group/AliceO2/pull/15747) 2026-09-02: Fix out-of-range hit access in the TOF hit merging by [@sawenzel](https://github.com/sawenzel)
- [#15730](https://github.com/AliceO2Group/AliceO2/pull/15730) 2026-09-02: ITSMFT: share tracking slab allocation primitives by [@mpuccio](https://github.com/mpuccio)
- [#15751](https://github.com/AliceO2Group/AliceO2/pull/15751) 2026-09-02: TRD geometry simplification by [@sawenzel](https://github.com/sawenzel)
- [#15736](https://github.com/AliceO2Group/AliceO2/pull/15736) 2026-09-03: Deduplicate the MFT flex and the shared ALPIDE metal stack by [@sawenzel](https://github.com/sawenzel)
- [#15743](https://github.com/AliceO2Group/AliceO2/pull/15743) 2026-09-03: PIPE: fix duplicated RB26/2 bellow and restore the empty RB26/3 bellow by [@sawenzel](https://github.com/sawenzel)
- [#15754](https://github.com/AliceO2Group/AliceO2/pull/15754) 2026-09-03: TPC: move disable-IDC-scalers to CorrectionMapsOptions by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15753](https://github.com/AliceO2Group/AliceO2/pull/15753) 2026-09-04: [ALICE3] FT3: fix kapton fractional Z by [@rliotino99](https://github.com/rliotino99)
- [#15762](https://github.com/AliceO2Group/AliceO2/pull/15762) 2026-09-04: Fix final so that new clang does not complain by [@ktf](https://github.com/ktf)
- [#15764](https://github.com/AliceO2Group/AliceO2/pull/15764) 2026-09-04: Fix UB when constructing string by [@ktf](https://github.com/ktf)
- [#15763](https://github.com/AliceO2Group/AliceO2/pull/15763) 2026-09-04: Improve the geometry-doctor reachability audit by [@sawenzel](https://github.com/sawenzel)
- [#15759](https://github.com/AliceO2Group/AliceO2/pull/15759) 2026-09-07: [ALICE3] IOTOF: update id number to current number of chips by [@maciacco](https://github.com/maciacco)
- [#15752](https://github.com/AliceO2Group/AliceO2/pull/15752) 2026-09-07: Let the ZEM calorimeters be built without the far beam line by [@sawenzel](https://github.com/sawenzel)
- [#15767](https://github.com/AliceO2Group/AliceO2/pull/15767) 2026-09-07: Place the L3 coil turns explicitly instead of dividing the polyhedra by [@sawenzel](https://github.com/sawenzel)
- [#15773](https://github.com/AliceO2Group/AliceO2/pull/15773) 2026-09-09: [ALICE3] IOTOF: Improve propagation of hit in digitizer stepping by [@Marcellocosti](https://github.com/Marcellocosti)
- [#15777](https://github.com/AliceO2Group/AliceO2/pull/15777) 2026-09-09: [ALICE3] TF3: store BC and TDC in digits by [@maciacco](https://github.com/maciacco)
- [#15737](https://github.com/AliceO2Group/AliceO2/pull/15737) 2026-09-10: Add support for VecGeom v2 by [@ktf](https://github.com/ktf)
- [#15782](https://github.com/AliceO2Group/AliceO2/pull/15782) 2026-09-10: Clusterer will prioritize labels with lower trackIndex by [@shahor02](https://github.com/shahor02)
- [#15760](https://github.com/AliceO2Group/AliceO2/pull/15760) 2026-09-10: Fix geometry overlaps found by geometry-doctor by [@sawenzel](https://github.com/sawenzel)
- [#15780](https://github.com/AliceO2Group/AliceO2/pull/15780) 2026-09-10: Fix material bug in FT3 by [@JustusRudolph](https://github.com/JustusRudolph)
- [#15778](https://github.com/AliceO2Group/AliceO2/pull/15778) 2026-09-10: small changes in handling empty TF by [@pillot](https://github.com/pillot)
- [#15787](https://github.com/AliceO2Group/AliceO2/pull/15787) 2026-09-11: [ALICE3] Fix magnet radius by [@njacazio](https://github.com/njacazio)
- [#15784](https://github.com/AliceO2Group/AliceO2/pull/15784) 2026-09-11: Regularization fixes + switch for the pre-PR-15610 compatibility mode by [@shahor02](https://github.com/shahor02)
- [#15783](https://github.com/AliceO2Group/AliceO2/pull/15783) 2026-09-11: Upgrades: clusterer will prioritize labels with lower trackIndex by [@shahor02](https://github.com/shahor02)
- [#15789](https://github.com/AliceO2Group/AliceO2/pull/15789) 2026-09-12: Add the CADsupport module: CAD geometries as exact TGeo solids by [@sawenzel](https://github.com/sawenzel)
- [#15790](https://github.com/AliceO2Group/AliceO2/pull/15790) 2026-09-12: CAD tutorial : MkDocs sources plus an ITS round-trip example by [@sawenzel](https://github.com/sawenzel)
## Changes in Examples

- [#15741](https://github.com/AliceO2Group/AliceO2/pull/15741) 2026-09-01: Simple event pool merger by [@jackal1-66](https://github.com/jackal1-66)
- [#15789](https://github.com/AliceO2Group/AliceO2/pull/15789) 2026-09-12: Add the CADsupport module: CAD geometries as exact TGeo solids by [@sawenzel](https://github.com/sawenzel)
## Changes in Framework

- [#15714](https://github.com/AliceO2Group/AliceO2/pull/15714) 2026-08-29: MC-RECO: Ability to process empty timeframes (Part 2) by [@sawenzel](https://github.com/sawenzel)
- [#15735](https://github.com/AliceO2Group/AliceO2/pull/15735) 2026-08-29: Track skipped read timeframes in file statistics & dumping by [@autumn-mck](https://github.com/autumn-mck)
- [#15732](https://github.com/AliceO2Group/AliceO2/pull/15732) 2026-08-31: DPL Analysis: allow finalising callback also for CCDB columns by [@ktf](https://github.com/ktf)
- [#15742](https://github.com/AliceO2Group/AliceO2/pull/15742) 2026-09-01: Analysis CCDB: add ability to have a uniformity column in a CCDB table by [@ktf](https://github.com/ktf)
- [#15744](https://github.com/AliceO2Group/AliceO2/pull/15744) 2026-09-03: DPL Analysis: allow lookup of paths based on the run / uniformity by [@ktf](https://github.com/ktf)
- [#15757](https://github.com/AliceO2Group/AliceO2/pull/15757) 2026-09-04: DPL CCDB Analysis: add ability to specify run dependent queries by [@ktf](https://github.com/ktf)
- [#15758](https://github.com/AliceO2Group/AliceO2/pull/15758) 2026-09-04: DPL: allow plugins to account for bytes by [@ktf](https://github.com/ktf)
- [#15761](https://github.com/AliceO2Group/AliceO2/pull/15761) 2026-09-04: Fix clang 21 issue by [@ktf](https://github.com/ktf)
## Changes in Generators

- [#15705](https://github.com/AliceO2Group/AliceO2/pull/15705) 2026-08-29: MC: Ability to process empty timeframes (Part 1) by [@sawenzel](https://github.com/sawenzel)
- [#15741](https://github.com/AliceO2Group/AliceO2/pull/15741) 2026-09-01: Simple event pool merger by [@jackal1-66](https://github.com/jackal1-66)
- [#15755](https://github.com/AliceO2Group/AliceO2/pull/15755) 2026-09-04: Add protection against skipped loopers by [@jackal1-66](https://github.com/jackal1-66)
- [#15774](https://github.com/AliceO2Group/AliceO2/pull/15774) 2026-09-09: Implement HepMC reading randomisation by [@jackal1-66](https://github.com/jackal1-66)
## Changes in Steer

- [#15705](https://github.com/AliceO2Group/AliceO2/pull/15705) 2026-08-29: MC: Ability to process empty timeframes (Part 1) by [@sawenzel](https://github.com/sawenzel)
- [#15778](https://github.com/AliceO2Group/AliceO2/pull/15778) 2026-09-10: small changes in handling empty TF by [@pillot](https://github.com/pillot)

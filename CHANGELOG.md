# Changes since 2026-10-03

## Changes in Algorithm

- [#15895](https://github.com/AliceO2Group/AliceO2/pull/15895) 2026-10-10: GPU: make the kernel entry-point signature work on Metal by [@ktf](https://github.com/ktf)
- [#15920](https://github.com/AliceO2Group/AliceO2/pull/15920) 2026-10-10: GPU: pass the work-item indices to the helpers that lack them by [@ktf](https://github.com/ktf)
- [#15933](https://github.com/AliceO2Group/AliceO2/pull/15933) 2026-10-11: GPU: Faster QuickSort implementation for large data, and some cleanup of option names and help / debug messages by [@davidrohr](https://github.com/davidrohr)
- [#15929](https://github.com/AliceO2Group/AliceO2/pull/15929) 2026-10-12: GPU: Add CPUDevice backend to emulate device-like processing on host by [@davidrohr](https://github.com/davidrohr)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
## Changes in Analysis

- [#15952](https://github.com/AliceO2Group/AliceO2/pull/15952) 2026-10-13: DPL Analysis: keep reading after a skipped timeframe by [@autumn-mck](https://github.com/autumn-mck)
- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15976](https://github.com/AliceO2Group/AliceO2/pull/15976) 2026-10-17: DPL: hoist std::views::filter to speedup compilation by [@ktf](https://github.com/ktf)
## Changes in Common

- [#15891](https://github.com/AliceO2Group/AliceO2/pull/15891) 2026-10-05: GPU: refuse deterministic mode on Metal by [@ktf](https://github.com/ktf)
- [#15887](https://github.com/AliceO2Group/AliceO2/pull/15887) 2026-10-06: Make the VecGeom navigation mode of o2-sim faster with safety bounds and MultiUnions by [@sawenzel](https://github.com/sawenzel)
- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)
- [#15895](https://github.com/AliceO2Group/AliceO2/pull/15895) 2026-10-10: GPU: make the kernel entry-point signature work on Metal by [@ktf](https://github.com/ktf)
- [#15920](https://github.com/AliceO2Group/AliceO2/pull/15920) 2026-10-10: GPU: pass the work-item indices to the helpers that lack them by [@ktf](https://github.com/ktf)
- [#15923](https://github.com/AliceO2Group/AliceO2/pull/15923) 2026-10-10: MathUtils: extend the OpenCL sincos workaround to Metal by [@ktf](https://github.com/ktf)
- [#15933](https://github.com/AliceO2Group/AliceO2/pull/15933) 2026-10-11: GPU: Faster QuickSort implementation for large data, and some cleanup of option names and help / debug messages by [@davidrohr](https://github.com/davidrohr)
- [#15932](https://github.com/AliceO2Group/AliceO2/pull/15932) 2026-10-11: GPU: give Metal its own spellings for two math helpers by [@ktf](https://github.com/ktf)
- [#15934](https://github.com/AliceO2Group/AliceO2/pull/15934) 2026-10-12: Fix missing include by [@davidrohr](https://github.com/davidrohr)
- [#15929](https://github.com/AliceO2Group/AliceO2/pull/15929) 2026-10-12: GPU: Add CPUDevice backend to emulate device-like processing on host by [@davidrohr](https://github.com/davidrohr)
- [#15936](https://github.com/AliceO2Group/AliceO2/pull/15936) 2026-10-12: GPU: reach global memory through the generic address space on Metal by [@ktf](https://github.com/ktf)
- [#15946](https://github.com/AliceO2Group/AliceO2/pull/15946) 2026-10-12: MathUtils: let bringTo* deduce the type of the call they forward to by [@ktf](https://github.com/ktf)
- [#15947](https://github.com/AliceO2Group/AliceO2/pull/15947) 2026-10-12: MathUtils: make getMean()'s ternary unambiguous for an emulated double by [@ktf](https://github.com/ktf)
- [#15944](https://github.com/AliceO2Group/AliceO2/pull/15944) 2026-10-14: GPU: make the Metal atomics operate on plain counters by [@ktf](https://github.com/ktf)
- [#15949](https://github.com/AliceO2Group/AliceO2/pull/15949) 2026-10-14: MathUtils: declare fastATan2's Pi where its lambdas can use it by [@ktf](https://github.com/ktf)
- [#15963](https://github.com/AliceO2Group/AliceO2/pull/15963) 2026-10-15: GPU: let the constant-address-space constants be built on Metal by [@ktf](https://github.com/ktf)
- [#15962](https://github.com/AliceO2Group/AliceO2/pull/15962) 2026-10-15: GPU: make std::is_pointer address-space aware on Metal by [@ktf](https://github.com/ktf)
- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
## Changes in DataFormats

- [#15897](https://github.com/AliceO2Group/AliceO2/pull/15897) 2026-10-05: SOR adjustment for some MC productions with ITS ramp-up offset by [@altsybee](https://github.com/altsybee)
- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)
- [#15907](https://github.com/AliceO2Group/AliceO2/pull/15907) 2026-10-08: [ALICE3] TF3: group pixel columns into readout columns in digitizer by [@maciacco](https://github.com/maciacco)
- [#15882](https://github.com/AliceO2Group/AliceO2/pull/15882) 2026-10-08: AFIT-81: Implementation of configurable parameters in FDD reco by [@wpierozak](https://github.com/wpierozak)
- [#15915](https://github.com/AliceO2Group/AliceO2/pull/15915) 2026-10-11: MatchCosmics: cosmics in collision data + raw-cluster writer with TOF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15919](https://github.com/AliceO2Group/AliceO2/pull/15919) 2026-10-12: adjustments for orbit shifts for other MCs done with MC-prod-v12 by [@altsybee](https://github.com/altsybee)
- [#15963](https://github.com/AliceO2Group/AliceO2/pull/15963) 2026-10-15: GPU: let the constant-address-space constants be built on Metal by [@ktf](https://github.com/ktf)
## Changes in Detectors

- [#15887](https://github.com/AliceO2Group/AliceO2/pull/15887) 2026-10-06: Make the VecGeom navigation mode of o2-sim faster with safety bounds and MultiUnions by [@sawenzel](https://github.com/sawenzel)
- [#15894](https://github.com/AliceO2Group/AliceO2/pull/15894) 2026-10-06: MatchCosmics: propagate the seed covariance to the DCA by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)
- [#15902](https://github.com/AliceO2Group/AliceO2/pull/15902) 2026-10-06: simple end-of-stave cards for ML barrel by [@altsybee](https://github.com/altsybee)
- [#15906](https://github.com/AliceO2Group/AliceO2/pull/15906) 2026-10-07: [ALICE3] TF3: fix conversion factor in absolute time computation by [@maciacco](https://github.com/maciacco)
- [#15905](https://github.com/AliceO2Group/AliceO2/pull/15905) 2026-10-07: simple EoS cards for disks by [@altsybee](https://github.com/altsybee)
- [#15893](https://github.com/AliceO2Group/AliceO2/pull/15893) 2026-10-07: TPC SCD: add option to keep all cluster and add MC to unbinned residuals by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15907](https://github.com/AliceO2Group/AliceO2/pull/15907) 2026-10-08: [ALICE3] TF3: group pixel columns into readout columns in digitizer by [@maciacco](https://github.com/maciacco)
- [#15882](https://github.com/AliceO2Group/AliceO2/pull/15882) 2026-10-08: AFIT-81: Implementation of configurable parameters in FDD reco by [@wpierozak](https://github.com/wpierozak)
- [#15914](https://github.com/AliceO2Group/AliceO2/pull/15914) 2026-10-08: Read MC kinematics per event in MatchITSTPCQC and fix a reader leak by [@sawenzel](https://github.com/sawenzel)
- [#15910](https://github.com/AliceO2Group/AliceO2/pull/15910) 2026-10-08: TPC track reader: do not accumulate MC labels over entries by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15883](https://github.com/AliceO2Group/AliceO2/pull/15883) 2026-10-09: [TF3] Add efficiency, time resolution and ToA maps from ccdb by [@GiorgioAlbertoLucia](https://github.com/GiorgioAlbertoLucia)
- [#15921](https://github.com/AliceO2Group/AliceO2/pull/15921) 2026-10-09: Option to provide marker/color per histo in residuals plot by [@shahor02](https://github.com/shahor02)
- [#15915](https://github.com/AliceO2Group/AliceO2/pull/15915) 2026-10-11: MatchCosmics: cosmics in collision data + raw-cluster writer with TOF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15954](https://github.com/AliceO2Group/AliceO2/pull/15954) 2026-10-13: Add mean res. and sigmas histo per h-barrel by [@shahor02](https://github.com/shahor02)
- [#15948](https://github.com/AliceO2Group/AliceO2/pull/15948) 2026-10-13: Add per HB histos with 1 channel/chip by [@shahor02](https://github.com/shahor02)
- [#15953](https://github.com/AliceO2Group/AliceO2/pull/15953) 2026-10-13: GRPGeomHelper optionally provides aligned geom + alignments by [@shahor02](https://github.com/shahor02)
- [#15918](https://github.com/AliceO2Group/AliceO2/pull/15918) 2026-10-13: ITS GPU seeding vertexer: add clamps to adaptive low mult threshold, and propate cells towards beamline by [@cima22](https://github.com/cima22)
- [#15943](https://github.com/AliceO2Group/AliceO2/pull/15943) 2026-10-13: TPC SCD: write and read the MC truth of the residual tracks by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15956](https://github.com/AliceO2Group/AliceO2/pull/15956) 2026-10-14: Fix misplaced default in the function by [@shahor02](https://github.com/shahor02)
- [#15912](https://github.com/AliceO2Group/AliceO2/pull/15912) 2026-10-16: TRK: IRIS service aluminium from the MaterialManager by [@f3sch](https://github.com/f3sch)
- [#15959](https://github.com/AliceO2Group/AliceO2/pull/15959) 2026-10-17: [ITS] Track Follower: use getPhiPos() for the phi search window by [@AizatDaribayeva](https://github.com/AizatDaribayeva)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
- [#15957](https://github.com/AliceO2Group/AliceO2/pull/15957) 2026-10-17: ITS2 - anomalous cluster task improvement  by [@nicolovalle](https://github.com/nicolovalle)
## Changes in Framework

- [#15911](https://github.com/AliceO2Group/AliceO2/pull/15911) 2026-10-07: Avoid noise from misc-include-cleaner by [@ktf](https://github.com/ktf)
- [#15922](https://github.com/AliceO2Group/AliceO2/pull/15922) 2026-10-12: DPL: fix DataDescriptorMatcher and the associated query builder by [@ktf](https://github.com/ktf)
- [#15940](https://github.com/AliceO2Group/AliceO2/pull/15940) 2026-10-13: [Framework] Add `is_not_filtered_iterator` concept by [@mhemmer-cern](https://github.com/mhemmer-cern)
- [#15942](https://github.com/AliceO2Group/AliceO2/pull/15942) 2026-10-13: DPL Analysis: fix incompatible Join check by [@aalkin](https://github.com/aalkin)
- [#15952](https://github.com/AliceO2Group/AliceO2/pull/15952) 2026-10-13: DPL Analysis: keep reading after a skipped timeframe by [@autumn-mck](https://github.com/autumn-mck)
- [#15938](https://github.com/AliceO2Group/AliceO2/pull/15938) 2026-10-13: DPL: Fix leaks and self-assignment in Array2D by [@aalkin](https://github.com/aalkin)
- [#15941](https://github.com/AliceO2Group/AliceO2/pull/15941) 2026-10-13: DPL: reduce verbosity of the missing resources warning by [@ktf](https://github.com/ktf)
- [#15939](https://github.com/AliceO2Group/AliceO2/pull/15939) 2026-10-14: DPL: Fix ownership of the content of Variant by [@aalkin](https://github.com/aalkin)
- [#15975](https://github.com/AliceO2Group/AliceO2/pull/15975) 2026-10-16: DPL Analysis: out-of-line HistogramSpec related construction by [@ktf](https://github.com/ktf)
- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15976](https://github.com/AliceO2Group/AliceO2/pull/15976) 2026-10-17: DPL: hoist std::views::filter to speedup compilation by [@ktf](https://github.com/ktf)
- [#15980](https://github.com/AliceO2Group/AliceO2/pull/15980) 2026-10-17: DPL: out of line HistogramSpec related destructors by [@ktf](https://github.com/ktf)
## Changes in Steer

- [#15893](https://github.com/AliceO2Group/AliceO2/pull/15893) 2026-10-07: TPC SCD: add option to keep all cluster and add MC to unbinned residuals by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15914](https://github.com/AliceO2Group/AliceO2/pull/15914) 2026-10-08: Read MC kinematics per event in MatchITSTPCQC and fix a reader leak by [@sawenzel](https://github.com/sawenzel)
- [#15883](https://github.com/AliceO2Group/AliceO2/pull/15883) 2026-10-09: [TF3] Add efficiency, time resolution and ToA maps from ccdb by [@GiorgioAlbertoLucia](https://github.com/GiorgioAlbertoLucia)
## Changes in Utilities

- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)

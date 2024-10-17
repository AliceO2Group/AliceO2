# Changes since 2024-10-03

## Changes in Analysis

- [#13528](https://github.com/AliceO2Group/AliceO2/pull/13528) 2024-10-08: [DM] Add expected time computation in Framework by [@njacazio](https://github.com/njacazio)
## Changes in Common

- [#13556](https://github.com/AliceO2Group/AliceO2/pull/13556) 2024-10-03: DCAFitterGPU: reduce I/O overhead by copying elements using a kernel by [@mconcas](https://github.com/mconcas)
- [#13516](https://github.com/AliceO2Group/AliceO2/pull/13516) 2024-10-05: Prototype for reader-driver ability to skim MC input from files by [@shahor02](https://github.com/shahor02)
- [#13574](https://github.com/AliceO2Group/AliceO2/pull/13574) 2024-10-09: GPU: Switch integer types to <cstdint> types + some related / minor changes by [@davidrohr](https://github.com/davidrohr)
- [#13583](https://github.com/AliceO2Group/AliceO2/pull/13583) 2024-10-10: DPL: allow customising DataProcessingStats intervals by [@ktf](https://github.com/ktf)
- [#13585](https://github.com/AliceO2Group/AliceO2/pull/13585) 2024-10-11: Add Ds* (433) to physics constants by [@fcatalan92](https://github.com/fcatalan92)
## Changes in DataFormats

- [#13555](https://github.com/AliceO2Group/AliceO2/pull/13555) 2024-10-03: AggregatedRunInfo struct proposal by [@sawenzel](https://github.com/sawenzel)
- [#13560](https://github.com/AliceO2Group/AliceO2/pull/13560) 2024-10-03: Revert "AggregatedRunInfo struct proposal" by [@sawenzel](https://github.com/sawenzel)
- [#13561](https://github.com/AliceO2Group/AliceO2/pull/13561) 2024-10-03: Unconditionally load PVs if requested by [@shahor02](https://github.com/shahor02)
- [#13516](https://github.com/AliceO2Group/AliceO2/pull/13516) 2024-10-05: Prototype for reader-driver ability to skim MC input from files by [@shahor02](https://github.com/shahor02)
- [#13564](https://github.com/AliceO2Group/AliceO2/pull/13564) 2024-10-07: Fix compilation issue in AggregatedRunInfo by [@sawenzel](https://github.com/sawenzel)
- [#13570](https://github.com/AliceO2Group/AliceO2/pull/13570) 2024-10-09: ctpdev: macro for creating CTP BK counters entry from CCDB. by [@lietava](https://github.com/lietava)
- [#13580](https://github.com/AliceO2Group/AliceO2/pull/13580) 2024-10-10: Improve AggregatedRunInfo: prioritize use of CTP/Calib/FirstRunOrbit by [@sawenzel](https://github.com/sawenzel)
- [#13584](https://github.com/AliceO2Group/AliceO2/pull/13584) 2024-10-11: Ship GRPECS as part of AggregatedRunInfo; cleanup by [@sawenzel](https://github.com/sawenzel)
- [#13586](https://github.com/AliceO2Group/AliceO2/pull/13586) 2024-10-12: AggregatedRunInfo can be requested via GRPGeomHelper + related fixes by [@shahor02](https://github.com/shahor02)
- [#13590](https://github.com/AliceO2Group/AliceO2/pull/13590) 2024-10-15: Custom orbit shifts for runs <=LHC22m by [@ekryshen](https://github.com/ekryshen)
## Changes in Detectors

- [#13561](https://github.com/AliceO2Group/AliceO2/pull/13561) 2024-10-03: Unconditionally load PVs if requested by [@shahor02](https://github.com/shahor02)
- [#13516](https://github.com/AliceO2Group/AliceO2/pull/13516) 2024-10-05: Prototype for reader-driver ability to skim MC input from files by [@shahor02](https://github.com/shahor02)
- [#13578](https://github.com/AliceO2Group/AliceO2/pull/13578) 2024-10-08: Protection against ITS/MFT GBTTrailer packet status corruption by [@shahor02](https://github.com/shahor02)
- [#13572](https://github.com/AliceO2Group/AliceO2/pull/13572) 2024-10-09: TOF digitizer: Ability to process events happening before timeframe s… by [@sawenzel](https://github.com/sawenzel)
- [#13562](https://github.com/AliceO2Group/AliceO2/pull/13562) 2024-10-09: TPC digi: Cut digits arriving before timeframe/readout start by [@sawenzel](https://github.com/sawenzel)
- [#13570](https://github.com/AliceO2Group/AliceO2/pull/13570) 2024-10-09: ctpdev: macro for creating CTP BK counters entry from CCDB. by [@lietava](https://github.com/lietava)
- [#13581](https://github.com/AliceO2Group/AliceO2/pull/13581) 2024-10-10: More informaritive verbose ITS/MFT raw data fetching logging by [@shahor02](https://github.com/shahor02)
- [#13582](https://github.com/AliceO2Group/AliceO2/pull/13582) 2024-10-10: assure TOF code uses std::abs everywhere by [@noferini](https://github.com/noferini)
- [#13586](https://github.com/AliceO2Group/AliceO2/pull/13586) 2024-10-12: AggregatedRunInfo can be requested via GRPGeomHelper + related fixes by [@shahor02](https://github.com/shahor02)
- [#13589](https://github.com/AliceO2Group/AliceO2/pull/13589) 2024-10-14: [MCH] skip digits produced before the beginning of the TF by [@pillot](https://github.com/pillot)
- [#13596](https://github.com/AliceO2Group/AliceO2/pull/13596) 2024-10-15: ITSGPU: Make threads and blocks configurable from CLI by [@mconcas](https://github.com/mconcas)
- [#13606](https://github.com/AliceO2Group/AliceO2/pull/13606) 2024-10-17: Fix broken --max-tf option of ReaderDriver by [@shahor02](https://github.com/shahor02)
- [#13594](https://github.com/AliceO2Group/AliceO2/pull/13594) 2024-10-17: TPC: adding check for SACs at endOfStream by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#13601](https://github.com/AliceO2Group/AliceO2/pull/13601) 2024-10-17: Update LZEROElectronics.h by [@sawenzel](https://github.com/sawenzel)
- [#13558](https://github.com/AliceO2Group/AliceO2/pull/13558) 2024-10-17: adding extra margin for sync TOF dia calibs by [@noferini](https://github.com/noferini)
## Changes in Framework

- [#13559](https://github.com/AliceO2Group/AliceO2/pull/13559) 2024-10-03: DPL Analysis: re-enable prefetching by [@ktf](https://github.com/ktf)
- [#13566](https://github.com/AliceO2Group/AliceO2/pull/13566) 2024-10-04: Revert "DPL Analysis: re-enable prefetching" by [@ktf](https://github.com/ktf)
- [#13576](https://github.com/AliceO2Group/AliceO2/pull/13576) 2024-10-08: Drop non C++20 code by [@ktf](https://github.com/ktf)
- [#13528](https://github.com/AliceO2Group/AliceO2/pull/13528) 2024-10-08: [DM] Add expected time computation in Framework by [@njacazio](https://github.com/njacazio)
- [#13575](https://github.com/AliceO2Group/AliceO2/pull/13575) 2024-10-09: Pass CCDB Headers together with binary blob by [@shahor02](https://github.com/shahor02)
- [#13573](https://github.com/AliceO2Group/AliceO2/pull/13573) 2024-10-10: DPL: add helper method to retrieve and cache CCDB metadata by [@ktf](https://github.com/ktf)
- [#13583](https://github.com/AliceO2Group/AliceO2/pull/13583) 2024-10-10: DPL: allow customising DataProcessingStats intervals by [@ktf](https://github.com/ktf)
- [#13586](https://github.com/AliceO2Group/AliceO2/pull/13586) 2024-10-12: AggregatedRunInfo can be requested via GRPGeomHelper + related fixes by [@shahor02](https://github.com/shahor02)
- [#13592](https://github.com/AliceO2Group/AliceO2/pull/13592) 2024-10-14: Modify CCDB headers check to account for CCDBSerialized<> access of non-CCDB objects by [@shahor02](https://github.com/shahor02)
- [#13598](https://github.com/AliceO2Group/AliceO2/pull/13598) 2024-10-17: DPL: speedup homogeneous_apply_ref by [@ktf](https://github.com/ktf)

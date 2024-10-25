# Changes since 2024-10-11

## Changes in Analysis

- [#13605](https://github.com/AliceO2Group/AliceO2/pull/13605) 2024-10-18: DPL analysis: hide internal symbols by [@ktf](https://github.com/ktf)
## Changes in Common

- [#13585](https://github.com/AliceO2Group/AliceO2/pull/13585) 2024-10-11: Add Ds* (433) to physics constants by [@fcatalan92](https://github.com/fcatalan92)
## Changes in DataFormats

- [#13584](https://github.com/AliceO2Group/AliceO2/pull/13584) 2024-10-11: Ship GRPECS as part of AggregatedRunInfo; cleanup by [@sawenzel](https://github.com/sawenzel)
- [#13586](https://github.com/AliceO2Group/AliceO2/pull/13586) 2024-10-12: AggregatedRunInfo can be requested via GRPGeomHelper + related fixes by [@shahor02](https://github.com/shahor02)
- [#13590](https://github.com/AliceO2Group/AliceO2/pull/13590) 2024-10-15: Custom orbit shifts for runs <=LHC22m by [@ekryshen](https://github.com/ekryshen)
- [#13604](https://github.com/AliceO2Group/AliceO2/pull/13604) 2024-10-19: calculate run 1st orbit only when not available from CCDB by [@shahor02](https://github.com/shahor02)
## Changes in Detectors

- [#13586](https://github.com/AliceO2Group/AliceO2/pull/13586) 2024-10-12: AggregatedRunInfo can be requested via GRPGeomHelper + related fixes by [@shahor02](https://github.com/shahor02)
- [#13589](https://github.com/AliceO2Group/AliceO2/pull/13589) 2024-10-14: [MCH] skip digits produced before the beginning of the TF by [@pillot](https://github.com/pillot)
- [#13596](https://github.com/AliceO2Group/AliceO2/pull/13596) 2024-10-15: ITSGPU: Make threads and blocks configurable from CLI by [@mconcas](https://github.com/mconcas)
- [#13606](https://github.com/AliceO2Group/AliceO2/pull/13606) 2024-10-17: Fix broken --max-tf option of ReaderDriver by [@shahor02](https://github.com/shahor02)
- [#13594](https://github.com/AliceO2Group/AliceO2/pull/13594) 2024-10-17: TPC: adding check for SACs at endOfStream by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#13601](https://github.com/AliceO2Group/AliceO2/pull/13601) 2024-10-17: Update LZEROElectronics.h by [@sawenzel](https://github.com/sawenzel)
- [#13558](https://github.com/AliceO2Group/AliceO2/pull/13558) 2024-10-17: adding extra margin for sync TOF dia calibs by [@noferini](https://github.com/noferini)
- [#13603](https://github.com/AliceO2Group/AliceO2/pull/13603) 2024-10-18: TRD digi / O2-5395: Trivial collision cut by [@sawenzel](https://github.com/sawenzel)
- [#13613](https://github.com/AliceO2Group/AliceO2/pull/13613) 2024-10-19: Fix ITS L2G matrix generation by [@shahor02](https://github.com/shahor02)
- [#13552](https://github.com/AliceO2Group/AliceO2/pull/13552) 2024-10-21: Adding plots vs TPC occupancy by [@chiarazampolli](https://github.com/chiarazampolli)
- [#13620](https://github.com/AliceO2Group/AliceO2/pull/13620) 2024-10-23: Optionally extract TPC clusters MC truth resolution by [@shahor02](https://github.com/shahor02)
- [#13565](https://github.com/AliceO2Group/AliceO2/pull/13565) 2024-10-24: [TPC-QC] Add DCAr selection to Tracks.cxx by [@makor](https://github.com/makor)
## Changes in Framework

- [#13586](https://github.com/AliceO2Group/AliceO2/pull/13586) 2024-10-12: AggregatedRunInfo can be requested via GRPGeomHelper + related fixes by [@shahor02](https://github.com/shahor02)
- [#13592](https://github.com/AliceO2Group/AliceO2/pull/13592) 2024-10-14: Modify CCDB headers check to account for CCDBSerialized<> access of non-CCDB objects by [@shahor02](https://github.com/shahor02)
- [#13598](https://github.com/AliceO2Group/AliceO2/pull/13598) 2024-10-17: DPL: speedup homogeneous_apply_ref by [@ktf](https://github.com/ktf)
- [#13605](https://github.com/AliceO2Group/AliceO2/pull/13605) 2024-10-18: DPL analysis: hide internal symbols by [@ktf](https://github.com/ktf)
- [#13609](https://github.com/AliceO2Group/AliceO2/pull/13609) 2024-10-18: DPL: move byteswapping helpers to Endian.h by [@ktf](https://github.com/ktf)
- [#13614](https://github.com/AliceO2Group/AliceO2/pull/13614) 2024-10-20: DPL: use constraint rather than static_assert by [@ktf](https://github.com/ktf)
- [#13588](https://github.com/AliceO2Group/AliceO2/pull/13588) 2024-10-21: DPL Analysis: enable cache without prefetching by [@ktf](https://github.com/ktf)
## Changes in Steer

- [#13599](https://github.com/AliceO2Group/AliceO2/pull/13599) 2024-10-18: MID: skip digits produced before the beginning of the TF by [@dstocco](https://github.com/dstocco)
- [#13615](https://github.com/AliceO2Group/AliceO2/pull/13615) 2024-10-21: Try to fix CI by adding MIDRaw lib to digitization by [@shahor02](https://github.com/shahor02)
- [#13619](https://github.com/AliceO2Group/AliceO2/pull/13619) 2024-10-22: Suppress if (ENABLE_UPGRADES) in the CMakefile by [@shahor02](https://github.com/shahor02)

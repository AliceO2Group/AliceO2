# Changes since 2026-09-12

## Changes in Algorithm

- [#15837](https://github.com/AliceO2Group/AliceO2/pull/15837) 2026-09-24: GPU: route noexcept through GPUnoexcept() for Metal by [@ktf](https://github.com/ktf)
- [#15835](https://github.com/AliceO2Group/AliceO2/pull/15835) 2026-09-25: GPU: three additional Metal adaptations by [@ktf](https://github.com/ktf)
## Changes in Analysis

- [#15795](https://github.com/AliceO2Group/AliceO2/pull/15795) 2026-09-17: Harden bulk TTree reads against corrupted baskets by [@autumn-mck](https://github.com/autumn-mck)
## Changes in Common

- [#15772](https://github.com/AliceO2Group/AliceO2/pull/15772) 2026-09-15: GPU: Metal branches in the common definition macros by [@ktf](https://github.com/ktf)
- [#15779](https://github.com/AliceO2Group/AliceO2/pull/15779) 2026-09-15: Handle new CCDB setup by avoiding hardcoding the CCDB url by [@ktf](https://github.com/ktf)
- [#15801](https://github.com/AliceO2Group/AliceO2/pull/15801) 2026-09-16: GPU: Metal branches in the common array and math helpers by [@ktf](https://github.com/ktf)
- [#15794](https://github.com/AliceO2Group/AliceO2/pull/15794) 2026-09-16: GPU: provide the std type_traits subset used on Metal by [@ktf](https://github.com/ktf)
- [#15802](https://github.com/AliceO2Group/AliceO2/pull/15802) 2026-09-17: TPC: place shared constants in the Metal constant address space by [@ktf](https://github.com/ktf)
- [#15812](https://github.com/AliceO2Group/AliceO2/pull/15812) 2026-09-18: Parallel Geant4 scoring and NIEL clamp by [@sawenzel](https://github.com/sawenzel)
- [#15819](https://github.com/AliceO2Group/AliceO2/pull/15819) 2026-09-19: fix int/uint comparison by [@shahor02](https://github.com/shahor02)
- [#15818](https://github.com/AliceO2Group/AliceO2/pull/15818) 2026-09-21: GPUTracking: place the remaining cluster-finder constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15825](https://github.com/AliceO2Group/AliceO2/pull/15825) 2026-09-22: Common: put the namespace-scope constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15826](https://github.com/AliceO2Group/AliceO2/pull/15826) 2026-09-23: MathUtils: make SMatrixGPU compile as MSL by [@ktf](https://github.com/ktf)
- [#15833](https://github.com/AliceO2Group/AliceO2/pull/15833) 2026-09-24: GPU: keep the constant memory block out of Metal's constant address space by [@ktf](https://github.com/ktf)
- [#15837](https://github.com/AliceO2Group/AliceO2/pull/15837) 2026-09-24: GPU: route noexcept through GPUnoexcept() for Metal by [@ktf](https://github.com/ktf)
- [#15850](https://github.com/AliceO2Group/AliceO2/pull/15850) 2026-09-25: GPU: extend two existing OpenCL device workarounds to Metal by [@ktf](https://github.com/ktf)
- [#15835](https://github.com/AliceO2Group/AliceO2/pull/15835) 2026-09-25: GPU: three additional Metal adaptations by [@ktf](https://github.com/ktf)
- [#15781](https://github.com/AliceO2Group/AliceO2/pull/15781) 2026-09-25: ITSMFT: unify cellular automaton tracking for ITS and MFT by [@mpuccio](https://github.com/mpuccio)
- [#15843](https://github.com/AliceO2Group/AliceO2/pull/15843) 2026-09-25: o2-sim: Fixes, code refactor, simplification and performance enhancement by [@sawenzel](https://github.com/sawenzel)
## Changes in DataFormats

- [#15779](https://github.com/AliceO2Group/AliceO2/pull/15779) 2026-09-15: Handle new CCDB setup by avoiding hardcoding the CCDB url by [@ktf](https://github.com/ktf)
- [#15802](https://github.com/AliceO2Group/AliceO2/pull/15802) 2026-09-17: TPC: place shared constants in the Metal constant address space by [@ktf](https://github.com/ktf)
- [#15819](https://github.com/AliceO2Group/AliceO2/pull/15819) 2026-09-19: fix int/uint comparison by [@shahor02](https://github.com/shahor02)
- [#15818](https://github.com/AliceO2Group/AliceO2/pull/15818) 2026-09-21: GPUTracking: place the remaining cluster-finder constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15825](https://github.com/AliceO2Group/AliceO2/pull/15825) 2026-09-22: Common: put the namespace-scope constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15831](https://github.com/AliceO2Group/AliceO2/pull/15831) 2026-09-23: Use the LHC orbit duration for CTP scaler rates by [@sawenzel](https://github.com/sawenzel)
- [#15844](https://github.com/AliceO2Group/AliceO2/pull/15844) 2026-09-25: Fix compiler warnings and errors related to dictionaries by [@sawenzel](https://github.com/sawenzel)
- [#15853](https://github.com/AliceO2Group/AliceO2/pull/15853) 2026-09-26: Fix couple of invalid debug print arguments by [@davidrohr](https://github.com/davidrohr)
## Changes in Detectors

- [#15789](https://github.com/AliceO2Group/AliceO2/pull/15789) 2026-09-12: Add the CADsupport module: CAD geometries as exact TGeo solids by [@sawenzel](https://github.com/sawenzel)
- [#15790](https://github.com/AliceO2Group/AliceO2/pull/15790) 2026-09-12: CAD tutorial : MkDocs sources plus an ITS round-trip example by [@sawenzel](https://github.com/sawenzel)
- [#15786](https://github.com/AliceO2Group/AliceO2/pull/15786) 2026-09-13: IOTOF: add in-pixel efficiency by [@GiorgioAlbertoLucia](https://github.com/GiorgioAlbertoLucia)
- [#15793](https://github.com/AliceO2Group/AliceO2/pull/15793) 2026-09-15: [ALICE3] Fix some overlaps between services and supports by [@marcovanleeuwen](https://github.com/marcovanleeuwen)
- [#15779](https://github.com/AliceO2Group/AliceO2/pull/15779) 2026-09-15: Handle new CCDB setup by avoiding hardcoding the CCDB url by [@ktf](https://github.com/ktf)
- [#15800](https://github.com/AliceO2Group/AliceO2/pull/15800) 2026-09-16: Fix codechecker violations by [@davidrohr](https://github.com/davidrohr)
- [#15788](https://github.com/AliceO2Group/AliceO2/pull/15788) 2026-09-17: [EMCAL-688] Improve ClusterFactory `evalDispersion` function and fix bug in `buildCluster` by [@mhemmer-cern](https://github.com/mhemmer-cern)
- [#15804](https://github.com/AliceO2Group/AliceO2/pull/15804) 2026-09-17: [TF3] Improve digit efficiency in stepping by [@Marcellocosti](https://github.com/Marcellocosti)
- [#15809](https://github.com/AliceO2Group/AliceO2/pull/15809) 2026-09-17: Place the space-frame sectors explicitly instead of dividing BBMO by [@sawenzel](https://github.com/sawenzel)
- [#15802](https://github.com/AliceO2Group/AliceO2/pull/15802) 2026-09-17: TPC: place shared constants in the Metal constant address space by [@ktf](https://github.com/ktf)
- [#15812](https://github.com/AliceO2Group/AliceO2/pull/15812) 2026-09-18: Parallel Geant4 scoring and NIEL clamp by [@sawenzel](https://github.com/sawenzel)
- [#15796](https://github.com/AliceO2Group/AliceO2/pull/15796) 2026-09-21: [MUON] fix computation of delta phi in MFT-MCH matching by [@aferrero2707](https://github.com/aferrero2707)
- [#15818](https://github.com/AliceO2Group/AliceO2/pull/15818) 2026-09-21: GPUTracking: place the remaining cluster-finder constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15821](https://github.com/AliceO2Group/AliceO2/pull/15821) 2026-09-21: More VecGeom v2.x compatibility by [@ktf](https://github.com/ktf)
- [#15825](https://github.com/AliceO2Group/AliceO2/pull/15825) 2026-09-22: Common: put the namespace-scope constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15830](https://github.com/AliceO2Group/AliceO2/pull/15830) 2026-09-22: GPU: more constexpr cleanups to support Metal by [@ktf](https://github.com/ktf)
- [#15816](https://github.com/AliceO2Group/AliceO2/pull/15816) 2026-09-22: Prepare for new ROOT by [@aalkin](https://github.com/aalkin)
- [#15828](https://github.com/AliceO2Group/AliceO2/pull/15828) 2026-09-23: [ALICE 3] FT3 fix magnetic field silently disabled when FT3 is active by [@bulukutlu](https://github.com/bulukutlu)
- [#15832](https://github.com/AliceO2Group/AliceO2/pull/15832) 2026-09-23: Fix out-of-range BC slice for ambiguous tracks past the last BC by [@sawenzel](https://github.com/sawenzel)
- [#15799](https://github.com/AliceO2Group/AliceO2/pull/15799) 2026-09-23: TRD: using slope to correct y position and reject more fakes by [@glegras](https://github.com/glegras)
- [#15831](https://github.com/AliceO2Group/AliceO2/pull/15831) 2026-09-23: Use the LHC orbit duration for CTP scaler rates by [@sawenzel](https://github.com/sawenzel)
- [#15827](https://github.com/AliceO2Group/AliceO2/pull/15827) 2026-09-24: [ALICE 3] FT3 digitization: Change axis convention in disc sensors by [@marcovanleeuwen](https://github.com/marcovanleeuwen)
- [#15841](https://github.com/AliceO2Group/AliceO2/pull/15841) 2026-09-24: Ship the NIEL damage weights as CSV by [@sawenzel](https://github.com/sawenzel)
- [#15838](https://github.com/AliceO2Group/AliceO2/pull/15838) 2026-09-24: Write tracked V0s, cascades and 3-bodies in collision order by [@sawenzel](https://github.com/sawenzel)
- [#15844](https://github.com/AliceO2Group/AliceO2/pull/15844) 2026-09-25: Fix compiler warnings and errors related to dictionaries by [@sawenzel](https://github.com/sawenzel)
- [#15840](https://github.com/AliceO2Group/AliceO2/pull/15840) 2026-09-25: Give the MFT support volume a name of its own by [@sawenzel](https://github.com/sawenzel)
- [#15781](https://github.com/AliceO2Group/AliceO2/pull/15781) 2026-09-25: ITSMFT: unify cellular automaton tracking for ITS and MFT by [@mpuccio](https://github.com/mpuccio)
- [#15843](https://github.com/AliceO2Group/AliceO2/pull/15843) 2026-09-25: o2-sim: Fixes, code refactor, simplification and performance enhancement by [@sawenzel](https://github.com/sawenzel)
## Changes in Examples

- [#15789](https://github.com/AliceO2Group/AliceO2/pull/15789) 2026-09-12: Add the CADsupport module: CAD geometries as exact TGeo solids by [@sawenzel](https://github.com/sawenzel)
## Changes in Framework

- [#15795](https://github.com/AliceO2Group/AliceO2/pull/15795) 2026-09-17: Harden bulk TTree reads against corrupted baskets by [@autumn-mck](https://github.com/autumn-mck)
- [#15829](https://github.com/AliceO2Group/AliceO2/pull/15829) 2026-09-22: Improve ability to sync analysis wagons options with the current release values by [@ktf](https://github.com/ktf)
## Changes in Generators

- [#15808](https://github.com/AliceO2Group/AliceO2/pull/15808) 2026-09-17: Give box-gun primaries weight 1 in o2-sim by [@sawenzel](https://github.com/sawenzel)
- [#15834](https://github.com/AliceO2Group/AliceO2/pull/15834) 2026-09-24: BoxGenerator: enable sampling of pT and rapidity instead of p and eta by [@fmazzasc](https://github.com/fmazzasc)
## Changes in Steer

- [#15843](https://github.com/AliceO2Group/AliceO2/pull/15843) 2026-09-25: o2-sim: Fixes, code refactor, simplification and performance enhancement by [@sawenzel](https://github.com/sawenzel)

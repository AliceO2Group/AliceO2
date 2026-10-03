# Changes since 2026-09-19

## Changes in Algorithm

- [#15837](https://github.com/AliceO2Group/AliceO2/pull/15837) 2026-09-24: GPU: route noexcept through GPUnoexcept() for Metal by [@ktf](https://github.com/ktf)
- [#15835](https://github.com/AliceO2Group/AliceO2/pull/15835) 2026-09-25: GPU: three additional Metal adaptations by [@ktf](https://github.com/ktf)
## Changes in Analysis

- [#15797](https://github.com/AliceO2Group/AliceO2/pull/15797) 2026-09-28: Support skipping invalid timeframes across parent files by [@autumn-mck](https://github.com/autumn-mck)
## Changes in Common

- [#15819](https://github.com/AliceO2Group/AliceO2/pull/15819) 2026-09-19: fix int/uint comparison by [@shahor02](https://github.com/shahor02)
- [#15818](https://github.com/AliceO2Group/AliceO2/pull/15818) 2026-09-21: GPUTracking: place the remaining cluster-finder constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15825](https://github.com/AliceO2Group/AliceO2/pull/15825) 2026-09-22: Common: put the namespace-scope constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15826](https://github.com/AliceO2Group/AliceO2/pull/15826) 2026-09-23: MathUtils: make SMatrixGPU compile as MSL by [@ktf](https://github.com/ktf)
- [#15833](https://github.com/AliceO2Group/AliceO2/pull/15833) 2026-09-24: GPU: keep the constant memory block out of Metal's constant address space by [@ktf](https://github.com/ktf)
- [#15837](https://github.com/AliceO2Group/AliceO2/pull/15837) 2026-09-24: GPU: route noexcept through GPUnoexcept() for Metal by [@ktf](https://github.com/ktf)
- [#15850](https://github.com/AliceO2Group/AliceO2/pull/15850) 2026-09-25: GPU: extend two existing OpenCL device workarounds to Metal by [@ktf](https://github.com/ktf)
- [#15835](https://github.com/AliceO2Group/AliceO2/pull/15835) 2026-09-25: GPU: three additional Metal adaptations by [@ktf](https://github.com/ktf)
- [#15843](https://github.com/AliceO2Group/AliceO2/pull/15843) 2026-09-25: o2-sim: Fixes, code refactor, simplification and performance enhancement by [@sawenzel](https://github.com/sawenzel)
- [#15820](https://github.com/AliceO2Group/AliceO2/pull/15820) 2026-09-28: Add missing resonances PDG codes in the contraints files by [@BongHwi](https://github.com/BongHwi)
- [#15855](https://github.com/AliceO2Group/AliceO2/pull/15855) 2026-09-28: Fix memory deletion for bufferptr in SerializedInfo/RootSerializableK… by [@f3sch](https://github.com/f3sch)
- [#15792](https://github.com/AliceO2Group/AliceO2/pull/15792) 2026-09-29: ORT CI tests by [@ChSonnabend](https://github.com/ChSonnabend)
- [#15870](https://github.com/AliceO2Group/AliceO2/pull/15870) 2026-09-30: Fix ORT CI test script by [@ChSonnabend](https://github.com/ChSonnabend)
- [#15866](https://github.com/AliceO2Group/AliceO2/pull/15866) 2026-10-01: GPU: give Metal an IEEE-754 binary64 in software by [@ktf](https://github.com/ktf)
- [#15865](https://github.com/AliceO2Group/AliceO2/pull/15865) 2026-10-01: o2-sim: VecGeom navigation mode for Geant4 by [@sawenzel](https://github.com/sawenzel)
## Changes in DataFormats

- [#15819](https://github.com/AliceO2Group/AliceO2/pull/15819) 2026-09-19: fix int/uint comparison by [@shahor02](https://github.com/shahor02)
- [#15818](https://github.com/AliceO2Group/AliceO2/pull/15818) 2026-09-21: GPUTracking: place the remaining cluster-finder constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15825](https://github.com/AliceO2Group/AliceO2/pull/15825) 2026-09-22: Common: put the namespace-scope constants in the constant address space by [@ktf](https://github.com/ktf)
- [#15831](https://github.com/AliceO2Group/AliceO2/pull/15831) 2026-09-23: Use the LHC orbit duration for CTP scaler rates by [@sawenzel](https://github.com/sawenzel)
- [#15844](https://github.com/AliceO2Group/AliceO2/pull/15844) 2026-09-25: Fix compiler warnings and errors related to dictionaries by [@sawenzel](https://github.com/sawenzel)
- [#15853](https://github.com/AliceO2Group/AliceO2/pull/15853) 2026-09-26: Fix couple of invalid debug print arguments by [@davidrohr](https://github.com/davidrohr)
- [#15820](https://github.com/AliceO2Group/AliceO2/pull/15820) 2026-09-28: Add missing resonances PDG codes in the contraints files by [@BongHwi](https://github.com/BongHwi)
- [#15857](https://github.com/AliceO2Group/AliceO2/pull/15857) 2026-09-29: Align Omega(2012) PDG code between database and transport by [@sawenzel](https://github.com/sawenzel)
- [#15861](https://github.com/AliceO2Group/AliceO2/pull/15861) 2026-09-29: Extend RecoContainer to support ITS and MFT cluster access per layer by [@shahor02](https://github.com/shahor02)
- [#15866](https://github.com/AliceO2Group/AliceO2/pull/15866) 2026-10-01: GPU: give Metal an IEEE-754 binary64 in software by [@ktf](https://github.com/ktf)
- [#15878](https://github.com/AliceO2Group/AliceO2/pull/15878) 2026-10-02: Support (staggered) per-layer cluster input in all ITS-dependent workflows by [@shahor02](https://github.com/shahor02)
## Changes in Detectors

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
- [#15843](https://github.com/AliceO2Group/AliceO2/pull/15843) 2026-09-25: o2-sim: Fixes, code refactor, simplification and performance enhancement by [@sawenzel](https://github.com/sawenzel)
- [#15848](https://github.com/AliceO2Group/AliceO2/pull/15848) 2026-09-28: [ALICE3] IOTOF: make sensor thickness configurable in geometry definition by [@maciacco](https://github.com/maciacco)
- [#15820](https://github.com/AliceO2Group/AliceO2/pull/15820) 2026-09-28: Add missing resonances PDG codes in the contraints files by [@BongHwi](https://github.com/BongHwi)
- [#15849](https://github.com/AliceO2Group/AliceO2/pull/15849) 2026-09-28: CAD simulation: Parallelize meshing and solid recognition; feedback fixes by [@sawenzel](https://github.com/sawenzel)
- [#15846](https://github.com/AliceO2Group/AliceO2/pull/15846) 2026-09-28: Fix minor reproducibility issues in TPC hit creation by [@sawenzel](https://github.com/sawenzel)
- [#15852](https://github.com/AliceO2Group/AliceO2/pull/15852) 2026-09-28: Relax the Geant4 field epsilons outside the muon spectrometer and support local field parameters by [@sawenzel](https://github.com/sawenzel)
- [#15862](https://github.com/AliceO2Group/AliceO2/pull/15862) 2026-09-29: Add the electron NIEL damage weights by [@sawenzel](https://github.com/sawenzel)
- [#15861](https://github.com/AliceO2Group/AliceO2/pull/15861) 2026-09-29: Extend RecoContainer to support ITS and MFT cluster access per layer by [@shahor02](https://github.com/shahor02)
- [#15863](https://github.com/AliceO2Group/AliceO2/pull/15863) 2026-09-29: Restrict the TRD PAI model to 14 particle species by [@sawenzel](https://github.com/sawenzel)
- [#15839](https://github.com/AliceO2Group/AliceO2/pull/15839) 2026-09-29: TPC SCD: add clampTgSlp to keep residuals beyond MaxTgSlp by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15868](https://github.com/AliceO2Group/AliceO2/pull/15868) 2026-09-30: Fix the MFT half-disk 3 support pockets to match the technical drawing by [@sawenzel](https://github.com/sawenzel)
- [#15864](https://github.com/AliceO2Group/AliceO2/pull/15864) 2026-09-30: Geometry stability fixes: remove nanometre gaps and overlaps caused by float rounding by [@sawenzel](https://github.com/sawenzel)
- [#15873](https://github.com/AliceO2Group/AliceO2/pull/15873) 2026-09-30: Resolve the TRD chamber by name when chamber assemblies have no volume id by [@sawenzel](https://github.com/sawenzel)
- [#15874](https://github.com/AliceO2Group/AliceO2/pull/15874) 2026-09-30: Use std::abs for floating-point values in TRD and ITS studies by [@sawenzel](https://github.com/sawenzel)
- [#15885](https://github.com/AliceO2Group/AliceO2/pull/15885) 2026-10-01: Cosmetic fix for FT3 GeometryTGeo to pass codechecker by [@shahor02](https://github.com/shahor02)
- [#15876](https://github.com/AliceO2Group/AliceO2/pull/15876) 2026-10-01: Fix the vertical position of IB FPC resistors by [@mario6829](https://github.com/mario6829)
- [#15865](https://github.com/AliceO2Group/AliceO2/pull/15865) 2026-10-01: o2-sim: VecGeom navigation mode for Geant4 by [@sawenzel](https://github.com/sawenzel)
- [#15884](https://github.com/AliceO2Group/AliceO2/pull/15884) 2026-10-02: [ALICE3] IOTOF: Make timing response independent for different pixels by [@maciacco](https://github.com/maciacco)
- [#15889](https://github.com/AliceO2Group/AliceO2/pull/15889) 2026-10-02: Document how to combine a CAD module with built-in detectors by [@sawenzel](https://github.com/sawenzel)
- [#15878](https://github.com/AliceO2Group/AliceO2/pull/15878) 2026-10-02: Support (staggered) per-layer cluster input in all ITS-dependent workflows by [@shahor02](https://github.com/shahor02)
- [#15858](https://github.com/AliceO2Group/AliceO2/pull/15858) 2026-10-03: FT3 New Optimised OT Tiling & Clean up material writing by [@JustusRudolph](https://github.com/JustusRudolph)
## Changes in EventVisualisation

- [#15878](https://github.com/AliceO2Group/AliceO2/pull/15878) 2026-10-02: Support (staggered) per-layer cluster input in all ITS-dependent workflows by [@shahor02](https://github.com/shahor02)
## Changes in Examples

- [#15847](https://github.com/AliceO2Group/AliceO2/pull/15847) 2026-09-27: Fix Hybrid example including new HepMC parameters by [@jackal1-66](https://github.com/jackal1-66)
## Changes in Framework

- [#15829](https://github.com/AliceO2Group/AliceO2/pull/15829) 2026-09-22: Improve ability to sync analysis wagons options with the current release values by [@ktf](https://github.com/ktf)
- [#15856](https://github.com/AliceO2Group/AliceO2/pull/15856) 2026-09-28: Fix ClassDef compilation warning by [@vkucera](https://github.com/vkucera)
- [#15797](https://github.com/AliceO2Group/AliceO2/pull/15797) 2026-09-28: Support skipping invalid timeframes across parent files by [@autumn-mck](https://github.com/autumn-mck)
## Changes in Generators

- [#15834](https://github.com/AliceO2Group/AliceO2/pull/15834) 2026-09-24: BoxGenerator: enable sampling of pT and rapidity instead of p and eta by [@fmazzasc](https://github.com/fmazzasc)
## Changes in Steer

- [#15843](https://github.com/AliceO2Group/AliceO2/pull/15843) 2026-09-25: o2-sim: Fixes, code refactor, simplification and performance enhancement by [@sawenzel](https://github.com/sawenzel)
- [#15857](https://github.com/AliceO2Group/AliceO2/pull/15857) 2026-09-29: Align Omega(2012) PDG code between database and transport by [@sawenzel](https://github.com/sawenzel)

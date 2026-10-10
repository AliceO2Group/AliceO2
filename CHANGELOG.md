# Changes since 2026-09-26

## Changes in Algorithm

- [#15895](https://github.com/AliceO2Group/AliceO2/pull/15895) 2026-10-10: GPU: make the kernel entry-point signature work on Metal by [@ktf](https://github.com/ktf)
## Changes in Analysis

- [#15836](https://github.com/AliceO2Group/AliceO2/pull/15836) 2026-10-06: additions for embedding by [@nzardosh](https://github.com/nzardosh)
## Changes in Common

- [#15855](https://github.com/AliceO2Group/AliceO2/pull/15855) 2026-09-28: Fix memory deletion for bufferptr in SerializedInfo/RootSerializableK… by [@f3sch](https://github.com/f3sch)
- [#15870](https://github.com/AliceO2Group/AliceO2/pull/15870) 2026-09-30: Fix ORT CI test script by [@ChSonnabend](https://github.com/ChSonnabend)
- [#15866](https://github.com/AliceO2Group/AliceO2/pull/15866) 2026-10-01: GPU: give Metal an IEEE-754 binary64 in software by [@ktf](https://github.com/ktf)
- [#15865](https://github.com/AliceO2Group/AliceO2/pull/15865) 2026-10-01: o2-sim: VecGeom navigation mode for Geant4 by [@sawenzel](https://github.com/sawenzel)
- [#15891](https://github.com/AliceO2Group/AliceO2/pull/15891) 2026-10-05: GPU: refuse deterministic mode on Metal by [@ktf](https://github.com/ktf)
- [#15887](https://github.com/AliceO2Group/AliceO2/pull/15887) 2026-10-06: Make the VecGeom navigation mode of o2-sim faster with safety bounds and MultiUnions by [@sawenzel](https://github.com/sawenzel)
- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)
- [#15895](https://github.com/AliceO2Group/AliceO2/pull/15895) 2026-10-10: GPU: make the kernel entry-point signature work on Metal by [@ktf](https://github.com/ktf)
- [#15923](https://github.com/AliceO2Group/AliceO2/pull/15923) 2026-10-10: MathUtils: extend the OpenCL sincos workaround to Metal by [@ktf](https://github.com/ktf)
## Changes in DataFormats

- [#15853](https://github.com/AliceO2Group/AliceO2/pull/15853) 2026-09-26: Fix couple of invalid debug print arguments by [@davidrohr](https://github.com/davidrohr)
- [#15857](https://github.com/AliceO2Group/AliceO2/pull/15857) 2026-09-29: Align Omega(2012) PDG code between database and transport by [@sawenzel](https://github.com/sawenzel)
- [#15861](https://github.com/AliceO2Group/AliceO2/pull/15861) 2026-09-29: Extend RecoContainer to support ITS and MFT cluster access per layer by [@shahor02](https://github.com/shahor02)
- [#15866](https://github.com/AliceO2Group/AliceO2/pull/15866) 2026-10-01: GPU: give Metal an IEEE-754 binary64 in software by [@ktf](https://github.com/ktf)
- [#15878](https://github.com/AliceO2Group/AliceO2/pull/15878) 2026-10-02: Support (staggered) per-layer cluster input in all ITS-dependent workflows by [@shahor02](https://github.com/shahor02)
- [#15897](https://github.com/AliceO2Group/AliceO2/pull/15897) 2026-10-05: SOR adjustment for some MC productions with ITS ramp-up offset by [@altsybee](https://github.com/altsybee)
- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)
- [#15907](https://github.com/AliceO2Group/AliceO2/pull/15907) 2026-10-08: [ALICE3] TF3: group pixel columns into readout columns in digitizer by [@maciacco](https://github.com/maciacco)
- [#15882](https://github.com/AliceO2Group/AliceO2/pull/15882) 2026-10-08: AFIT-81: Implementation of configurable parameters in FDD reco by [@wpierozak](https://github.com/wpierozak)
## Changes in Detectors

- [#15848](https://github.com/AliceO2Group/AliceO2/pull/15848) 2026-09-28: [ALICE3] IOTOF: make sensor thickness configurable in geometry definition by [@maciacco](https://github.com/maciacco)
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
- [#15887](https://github.com/AliceO2Group/AliceO2/pull/15887) 2026-10-06: Make the VecGeom navigation mode of o2-sim faster with safety bounds and MultiUnions by [@sawenzel](https://github.com/sawenzel)
- [#15894](https://github.com/AliceO2Group/AliceO2/pull/15894) 2026-10-06: MatchCosmics: propagate the seed covariance to the DCA by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)
- [#15902](https://github.com/AliceO2Group/AliceO2/pull/15902) 2026-10-06: simple end-of-stave cards for ML barrel by [@altsybee](https://github.com/altsybee)
- [#15906](https://github.com/AliceO2Group/AliceO2/pull/15906) 2026-10-07: [ALICE3] TF3: fix conversion factor in absolute time computation by [@maciacco](https://github.com/maciacco)
- [#15879](https://github.com/AliceO2Group/AliceO2/pull/15879) 2026-10-07: Remove air pockets from the beampipe by [@sawenzel](https://github.com/sawenzel)
- [#15905](https://github.com/AliceO2Group/AliceO2/pull/15905) 2026-10-07: simple EoS cards for disks by [@altsybee](https://github.com/altsybee)
- [#15893](https://github.com/AliceO2Group/AliceO2/pull/15893) 2026-10-07: TPC SCD: add option to keep all cluster and add MC to unbinned residuals by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15907](https://github.com/AliceO2Group/AliceO2/pull/15907) 2026-10-08: [ALICE3] TF3: group pixel columns into readout columns in digitizer by [@maciacco](https://github.com/maciacco)
- [#15882](https://github.com/AliceO2Group/AliceO2/pull/15882) 2026-10-08: AFIT-81: Implementation of configurable parameters in FDD reco by [@wpierozak](https://github.com/wpierozak)
- [#15914](https://github.com/AliceO2Group/AliceO2/pull/15914) 2026-10-08: Read MC kinematics per event in MatchITSTPCQC and fix a reader leak by [@sawenzel](https://github.com/sawenzel)
- [#15910](https://github.com/AliceO2Group/AliceO2/pull/15910) 2026-10-08: TPC track reader: do not accumulate MC labels over entries by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15883](https://github.com/AliceO2Group/AliceO2/pull/15883) 2026-10-09: [TF3] Add efficiency, time resolution and ToA maps from ccdb by [@GiorgioAlbertoLucia](https://github.com/GiorgioAlbertoLucia)
- [#15921](https://github.com/AliceO2Group/AliceO2/pull/15921) 2026-10-09: Option to provide marker/color per histo in residuals plot by [@shahor02](https://github.com/shahor02)
## Changes in EventVisualisation

- [#15878](https://github.com/AliceO2Group/AliceO2/pull/15878) 2026-10-02: Support (staggered) per-layer cluster input in all ITS-dependent workflows by [@shahor02](https://github.com/shahor02)
## Changes in Examples

- [#15847](https://github.com/AliceO2Group/AliceO2/pull/15847) 2026-09-27: Fix Hybrid example including new HepMC parameters by [@jackal1-66](https://github.com/jackal1-66)
## Changes in Framework

- [#15856](https://github.com/AliceO2Group/AliceO2/pull/15856) 2026-09-28: Fix ClassDef compilation warning by [@vkucera](https://github.com/vkucera)
- [#15836](https://github.com/AliceO2Group/AliceO2/pull/15836) 2026-10-06: additions for embedding by [@nzardosh](https://github.com/nzardosh)
- [#15911](https://github.com/AliceO2Group/AliceO2/pull/15911) 2026-10-07: Avoid noise from misc-include-cleaner by [@ktf](https://github.com/ktf)
## Changes in Steer

- [#15857](https://github.com/AliceO2Group/AliceO2/pull/15857) 2026-09-29: Align Omega(2012) PDG code between database and transport by [@sawenzel](https://github.com/sawenzel)
- [#15893](https://github.com/AliceO2Group/AliceO2/pull/15893) 2026-10-07: TPC SCD: add option to keep all cluster and add MC to unbinned residuals by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15914](https://github.com/AliceO2Group/AliceO2/pull/15914) 2026-10-08: Read MC kinematics per event in MatchITSTPCQC and fix a reader leak by [@sawenzel](https://github.com/sawenzel)
- [#15883](https://github.com/AliceO2Group/AliceO2/pull/15883) 2026-10-09: [TF3] Add efficiency, time resolution and ToA maps from ccdb by [@GiorgioAlbertoLucia](https://github.com/GiorgioAlbertoLucia)
## Changes in Utilities

- [#15899](https://github.com/AliceO2Group/AliceO2/pull/15899) 2026-10-06: Prevent broken tests related statements when BUILD_TESTING is OFF by [@ktf](https://github.com/ktf)

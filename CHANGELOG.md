# Changes since 2022-01-29

## Changes in Algorithm

- [#8078](https://github.com/AliceO2Group/AliceO2/pull/8078) 2022-02-04: [QC-741] Mergers: support merging histograms with averages by [@knopers8](https://github.com/knopers8)
## Changes in Analysis

- [#8041](https://github.com/AliceO2Group/AliceO2/pull/8041) 2022-01-31: Introducing McParticle version 001 by [@jgrosseo](https://github.com/jgrosseo)
- [#8048](https://github.com/AliceO2Group/AliceO2/pull/8048) 2022-02-01: McParticle: moving default to 001 by [@jgrosseo](https://github.com/jgrosseo)
- [#8071](https://github.com/AliceO2Group/AliceO2/pull/8071) 2022-02-04: improve comment by [@jgrosseo](https://github.com/jgrosseo)
- [#8121](https://github.com/AliceO2Group/AliceO2/pull/8121) 2022-02-11: DPL Analysis: index builder helper by [@aalkin](https://github.com/aalkin)
- [#8127](https://github.com/AliceO2Group/AliceO2/pull/8127) 2022-02-11: DPL: make sure Condition<> object can be used by [@ktf](https://github.com/ktf)
## Changes in Common

- [#8100](https://github.com/AliceO2Group/AliceO2/pull/8100) 2022-02-09: Check availability of CTF-dictionary before opening the file by [@shahor02](https://github.com/shahor02)
## Changes in DataFormats

- [#8039](https://github.com/AliceO2Group/AliceO2/pull/8039) 2022-01-29: GPU Standalone: Add script to set up build environment for GPU standalone benchmark by [@davidrohr](https://github.com/davidrohr)
- [#8053](https://github.com/AliceO2Group/AliceO2/pull/8053) 2022-02-02: [QC-725] Allow to store TRFCollections as CSV files by [@knopers8](https://github.com/knopers8)
- [#8069](https://github.com/AliceO2Group/AliceO2/pull/8069) 2022-02-03: Truncate digitcontext output by [@sawenzel](https://github.com/sawenzel)
- [#8089](https://github.com/AliceO2Group/AliceO2/pull/8089) 2022-02-08: [EMCAL-548, EMCAL-614,EMCAL-687,EMCAL-757] Fill EMCAL information in AOD in simulations by [@mfasDa](https://github.com/mfasDa)
- [#8100](https://github.com/AliceO2Group/AliceO2/pull/8100) 2022-02-09: Check availability of CTF-dictionary before opening the file by [@shahor02](https://github.com/shahor02)
- [#8086](https://github.com/AliceO2Group/AliceO2/pull/8086) 2022-02-09: add  CreationTime for FT0  and FV0 calibration objects by [@AllaMaevskaya](https://github.com/AllaMaevskaya)
- [#8082](https://github.com/AliceO2Group/AliceO2/pull/8082) 2022-02-10: Add runType to GRPECS + its creator by [@shahor02](https://github.com/shahor02)
- [#8073](https://github.com/AliceO2Group/AliceO2/pull/8073) 2022-02-11: Use Cluster class for MID instead of Cluster2D and Cluster3D by [@dstocco](https://github.com/dstocco)
## Changes in Detectors

- [#8035](https://github.com/AliceO2Group/AliceO2/pull/8035) 2022-01-29: Adapt AOD MCLabels to non-redundant storage of ambiguous tracks by [@shahor02](https://github.com/shahor02)
- [#8036](https://github.com/AliceO2Group/AliceO2/pull/8036) 2022-01-29: DCS proxies use now() in ms to fill DPH.creation time by [@shahor02](https://github.com/shahor02)
- [#8040](https://github.com/AliceO2Group/AliceO2/pull/8040) 2022-01-29: Fix in ambiguous tracks tagging by [@shahor02](https://github.com/shahor02)
- [#8039](https://github.com/AliceO2Group/AliceO2/pull/8039) 2022-01-29: GPU Standalone: Add script to set up build environment for GPU standalone benchmark by [@davidrohr](https://github.com/davidrohr)
- [#8038](https://github.com/AliceO2Group/AliceO2/pull/8038) 2022-01-29: Revert "Improving sensitive hit creation for FT0 in Detector.cxx" by [@sawenzel](https://github.com/sawenzel)
- [#8044](https://github.com/AliceO2Group/AliceO2/pull/8044) 2022-01-31: AOD stores the time with full float precision by [@shahor02](https://github.com/shahor02)
- [#8042](https://github.com/AliceO2Group/AliceO2/pull/8042) 2022-01-31: Load Geant4 libraries (MacOSX Monterey) by [@pzhristov](https://github.com/pzhristov)
- [#8047](https://github.com/AliceO2Group/AliceO2/pull/8047) 2022-01-31: Promote/demote several warnings/errors to alarm, in order to raise infologger-min-severity to important for sync processing by [@davidrohr](https://github.com/davidrohr)
- [#8048](https://github.com/AliceO2Group/AliceO2/pull/8048) 2022-02-01: McParticle: moving default to 001 by [@jgrosseo](https://github.com/jgrosseo)
- [#8045](https://github.com/AliceO2Group/AliceO2/pull/8045) 2022-02-01: switch to disable fall-back to TGeo if MatLUT is missing by [@shahor02](https://github.com/shahor02)
- [#8063](https://github.com/AliceO2Group/AliceO2/pull/8063) 2022-02-02: Fixes in ITS noise calibrator output by [@shahor02](https://github.com/shahor02)
- [#8061](https://github.com/AliceO2Group/AliceO2/pull/8061) 2022-02-02: Possibility to attach multiple CCDB populators by [@shahor02](https://github.com/shahor02)
- [#8058](https://github.com/AliceO2Group/AliceO2/pull/8058) 2022-02-02: Report ROF orbit in ITS/MFT raw decoder error messages by [@shahor02](https://github.com/shahor02)
- [#8051](https://github.com/AliceO2Group/AliceO2/pull/8051) 2022-02-02: [EMCAL-751] Temporal fix for the digitizer by [@hahassan7](https://github.com/hahassan7)
- [#8052](https://github.com/AliceO2Group/AliceO2/pull/8052) 2022-02-02: [MFT] Fix digi2raw output segmentation by [@rpezzi](https://github.com/rpezzi)
- [#8070](https://github.com/AliceO2Group/AliceO2/pull/8070) 2022-02-03: Report wrong double column order in the Alpide data, reorder hits by [@shahor02](https://github.com/shahor02)
- [#8066](https://github.com/AliceO2Group/AliceO2/pull/8066) 2022-02-03: [MCH] keep digit NofSamples within limits by [@pillot](https://github.com/pillot)
- [#8062](https://github.com/AliceO2Group/AliceO2/pull/8062) 2022-02-03: fix printf warnings by [@shahor02](https://github.com/shahor02)
- [#8080](https://github.com/AliceO2Group/AliceO2/pull/8080) 2022-02-04: DCS filepush server emulator + documentation by [@shahor02](https://github.com/shahor02)
- [#8074](https://github.com/AliceO2Group/AliceO2/pull/8074) 2022-02-04: TRD KrClusterer skip shared digits by [@martenole](https://github.com/martenole)
- [#8072](https://github.com/AliceO2Group/AliceO2/pull/8072) 2022-02-04: Use double precision in MID mapping by [@dstocco](https://github.com/dstocco)
- [#8075](https://github.com/AliceO2Group/AliceO2/pull/8075) 2022-02-04: [EMCAL-556] Trace EMCAL run SOR/EOR in EMC DCS DP processor by [@shahor02](https://github.com/shahor02)
- [#8084](https://github.com/AliceO2Group/AliceO2/pull/8084) 2022-02-05: Fix for DCS emulator compilation by [@shahor02](https://github.com/shahor02)
- [#8068](https://github.com/AliceO2Group/AliceO2/pull/8068) 2022-02-08: Add Origin table by [@nburmaso](https://github.com/nburmaso)
- [#8091](https://github.com/AliceO2Group/AliceO2/pull/8091) 2022-02-08: TPC: add configKeyValue option to IDC workflows by [@wiechula](https://github.com/wiechula)
- [#8089](https://github.com/AliceO2Group/AliceO2/pull/8089) 2022-02-08: [EMCAL-548, EMCAL-614,EMCAL-687,EMCAL-757] Fill EMCAL information in AOD in simulations by [@mfasDa](https://github.com/mfasDa)
- [#8077](https://github.com/AliceO2Group/AliceO2/pull/8077) 2022-02-08: remove unused material to avoid FLUKA crash by [@AllaMaevskaya](https://github.com/AllaMaevskaya)
- [#8100](https://github.com/AliceO2Group/AliceO2/pull/8100) 2022-02-09: Check availability of CTF-dictionary before opening the file by [@shahor02](https://github.com/shahor02)
- [#8095](https://github.com/AliceO2Group/AliceO2/pull/8095) 2022-02-09: Do not discard decoded chip data if decoding error was set by [@shahor02](https://github.com/shahor02)
- [#8104](https://github.com/AliceO2Group/AliceO2/pull/8104) 2022-02-09: Suppress header exposing filesystem to clang by [@shahor02](https://github.com/shahor02)
- [#8090](https://github.com/AliceO2Group/AliceO2/pull/8090) 2022-02-09: Timestamp propagation and use in digitizer workflow by [@sawenzel](https://github.com/sawenzel)
- [#8059](https://github.com/AliceO2Group/AliceO2/pull/8059) 2022-02-09: Update of the MFT assessment workflow, added several histograms by [@sarahherrmann](https://github.com/sarahherrmann)
- [#8085](https://github.com/AliceO2Group/AliceO2/pull/8085) 2022-02-09: [O2-2776] produce raw data dumps on ITS/MFT decoding errors by [@shahor02](https://github.com/shahor02)
- [#8086](https://github.com/AliceO2Group/AliceO2/pull/8086) 2022-02-09: add  CreationTime for FT0  and FV0 calibration objects by [@AllaMaevskaya](https://github.com/AllaMaevskaya)
- [#8101](https://github.com/AliceO2Group/AliceO2/pull/8101) 2022-02-09: return to "old style" CCDB access in reconstruction by [@AllaMaevskaya](https://github.com/AllaMaevskaya)
- [#8082](https://github.com/AliceO2Group/AliceO2/pull/8082) 2022-02-10: Add runType to GRPECS + its creator by [@shahor02](https://github.com/shahor02)
- [#8050](https://github.com/AliceO2Group/AliceO2/pull/8050) 2022-02-10: SpaceCharge: adding getters for distortions and corrections by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#8076](https://github.com/AliceO2Group/AliceO2/pull/8076) 2022-02-10: [EMCAL-565]: Added ccdb entry framework. by [@hjbossi](https://github.com/hjbossi)
- [#8105](https://github.com/AliceO2Group/AliceO2/pull/8105) 2022-02-10: [EMCAL-757] Fix return type in cell-reader-workflow by [@mfasDa](https://github.com/mfasDa)
- [#8057](https://github.com/AliceO2Group/AliceO2/pull/8057) 2022-02-10: [MFT] Add timers to tracker workflow by [@rpezzi](https://github.com/rpezzi)
- [#8094](https://github.com/AliceO2Group/AliceO2/pull/8094) 2022-02-10: updated options by [@alindner14](https://github.com/alindner14)
- [#8114](https://github.com/AliceO2Group/AliceO2/pull/8114) 2022-02-11: MFT calib workflow update to accept config key values. by [@tomas-herman](https://github.com/tomas-herman)
- [#8131](https://github.com/AliceO2Group/AliceO2/pull/8131) 2022-02-11: Protection against unset chipID in case of Alpide data corruption by [@shahor02](https://github.com/shahor02)
- [#8073](https://github.com/AliceO2Group/AliceO2/pull/8073) 2022-02-11: Use Cluster class for MID instead of Cluster2D and Cluster3D by [@dstocco](https://github.com/dstocco)
- [#8111](https://github.com/AliceO2Group/AliceO2/pull/8111) 2022-02-11: [EMCAL-670] fixed wrong eta/phi pos of clusters in ClusterFactory by [@fjonasALICE](https://github.com/fjonasALICE)
- [#8108](https://github.com/AliceO2Group/AliceO2/pull/8108) 2022-02-11: fix title offset for dpg (L126-L131) by [@alindner14](https://github.com/alindner14)
## Changes in Framework

- [#8041](https://github.com/AliceO2Group/AliceO2/pull/8041) 2022-01-31: Introducing McParticle version 001 by [@jgrosseo](https://github.com/jgrosseo)
- [#8047](https://github.com/AliceO2Group/AliceO2/pull/8047) 2022-01-31: Promote/demote several warnings/errors to alarm, in order to raise infologger-min-severity to important for sync processing by [@davidrohr](https://github.com/davidrohr)
- [#8048](https://github.com/AliceO2Group/AliceO2/pull/8048) 2022-02-01: McParticle: moving default to 001 by [@jgrosseo](https://github.com/jgrosseo)
- [#8067](https://github.com/AliceO2Group/AliceO2/pull/8067) 2022-02-03: DPL: cleanup remaining messages by [@ktf](https://github.com/ktf)
- [#8064](https://github.com/AliceO2Group/AliceO2/pull/8064) 2022-02-04: DPL: timeout on STOP transition by [@ktf](https://github.com/ktf)
- [#8071](https://github.com/AliceO2Group/AliceO2/pull/8071) 2022-02-04: improve comment by [@jgrosseo](https://github.com/jgrosseo)
- [#8083](https://github.com/AliceO2Group/AliceO2/pull/8083) 2022-02-05: DPL Analysis: index equivalence fix by [@aalkin](https://github.com/aalkin)
- [#8097](https://github.com/AliceO2Group/AliceO2/pull/8097) 2022-02-09: Do not produce an error on default finaliseCCDB by [@shahor02](https://github.com/shahor02)
- [#8116](https://github.com/AliceO2Group/AliceO2/pull/8116) 2022-02-10: DPL: fix warnings by [@ktf](https://github.com/ktf)
- [#8110](https://github.com/AliceO2Group/AliceO2/pull/8110) 2022-02-10: DPL: properly handle pollers on start-stop-start transition (O2-2751) by [@ktf](https://github.com/ktf)
- [#8121](https://github.com/AliceO2Group/AliceO2/pull/8121) 2022-02-11: DPL Analysis: index builder helper by [@aalkin](https://github.com/aalkin)
- [#8127](https://github.com/AliceO2Group/AliceO2/pull/8127) 2022-02-11: DPL: make sure Condition<> object can be used by [@ktf](https://github.com/ktf)
- [#8113](https://github.com/AliceO2Group/AliceO2/pull/8113) 2022-02-11: add first and last shorthand for array by [@jgrosseo](https://github.com/jgrosseo)
## Changes in Steer

- [#8069](https://github.com/AliceO2Group/AliceO2/pull/8069) 2022-02-03: Truncate digitcontext output by [@sawenzel](https://github.com/sawenzel)
- [#8090](https://github.com/AliceO2Group/AliceO2/pull/8090) 2022-02-09: Timestamp propagation and use in digitizer workflow by [@sawenzel](https://github.com/sawenzel)
## Changes in Testing

- [#8073](https://github.com/AliceO2Group/AliceO2/pull/8073) 2022-02-11: Use Cluster class for MID instead of Cluster2D and Cluster3D by [@dstocco](https://github.com/dstocco)
## Changes in Utilities

- [#8037](https://github.com/AliceO2Group/AliceO2/pull/8037) 2022-01-29: Add SIGUSR1 signal handler to ShmManager by [@rbx](https://github.com/rbx)
- [#8060](https://github.com/AliceO2Group/AliceO2/pull/8060) 2022-02-03: update ShmManager::ResetContent to be able to reset after a crash by [@rbx](https://github.com/rbx)
- [#8078](https://github.com/AliceO2Group/AliceO2/pull/8078) 2022-02-04: [QC-741] Mergers: support merging histograms with averages by [@knopers8](https://github.com/knopers8)

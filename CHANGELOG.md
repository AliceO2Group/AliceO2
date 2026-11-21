# Changes since 2026-11-07

## Changes in Algorithm

- [#16110](https://github.com/AliceO2Group/AliceO2/pull/16110) 2026-11-12: GPU: implement the work-group and warp collectives for Metal by [@ktf](https://github.com/ktf)
- [#16126](https://github.com/AliceO2Group/AliceO2/pull/16126) 2026-11-15: GPU Metal: fence only threadgroup memory in the work-group collectives by [@ktf](https://github.com/ktf)
## Changes in Analysis

- [#16080](https://github.com/AliceO2Group/AliceO2/pull/16080) 2026-11-08: DPL Analysis: query the CCDB once per run of equal timestamps by [@ktf](https://github.com/ktf)
- [#16079](https://github.com/AliceO2Group/AliceO2/pull/16079) 2026-11-08: DPL Analysis: send CCDB tables together with the objects they refer to by [@ktf](https://github.com/ktf)
## Changes in Common

- [#16078](https://github.com/AliceO2Group/AliceO2/pull/16078) 2026-11-09: FileFetcher acknowledges suppression of processed files by [@shahor02](https://github.com/shahor02)
- [#16085](https://github.com/AliceO2Group/AliceO2/pull/16085) 2026-11-10: Add optional per-sector seeding to the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16094](https://github.com/AliceO2Group/AliceO2/pull/16094) 2026-11-10: Fix deprecated use of localtime by [@davidrohr](https://github.com/davidrohr)
- [#16088](https://github.com/AliceO2Group/AliceO2/pull/16088) 2026-11-11: DPL: when updating params provenance, check if the iterator is not at the end by [@aalkin](https://github.com/aalkin)
- [#16084](https://github.com/AliceO2Group/AliceO2/pull/16084) 2026-11-12: Add a multithreaded Geant4 mode to o2-sim by [@sawenzel](https://github.com/sawenzel)
- [#16110](https://github.com/AliceO2Group/AliceO2/pull/16110) 2026-11-12: GPU: implement the work-group and warp collectives for Metal by [@ktf](https://github.com/ktf)
- [#16118](https://github.com/AliceO2Group/AliceO2/pull/16118) 2026-11-13: [ALICE3]: Add FD3 hit by [@njacazio](https://github.com/njacazio)
- [#16126](https://github.com/AliceO2Group/AliceO2/pull/16126) 2026-11-15: GPU Metal: fence only threadgroup memory in the work-group collectives by [@ktf](https://github.com/ktf)
- [#16149](https://github.com/AliceO2Group/AliceO2/pull/16149) 2026-11-19: GPU: Unify usage of half precision float into half_t typedef in GPUCommonMath.h by [@davidrohr](https://github.com/davidrohr)
## Changes in DataFormats

- [#16095](https://github.com/AliceO2Group/AliceO2/pull/16095) 2026-11-10: Fix missing include and library for GCC 16 by [@davidrohr](https://github.com/davidrohr)
- [#16118](https://github.com/AliceO2Group/AliceO2/pull/16118) 2026-11-13: [ALICE3]: Add FD3 hit by [@njacazio](https://github.com/njacazio)
- [#16122](https://github.com/AliceO2Group/AliceO2/pull/16122) 2026-11-14: [ALICE3] TF3: Store BC and relative time in cluster by [@maciacco](https://github.com/maciacco)
- [#16132](https://github.com/AliceO2Group/AliceO2/pull/16132) 2026-11-16: MUON: Fix missing include for libfmt12 and GCC 16 by [@davidrohr](https://github.com/davidrohr)
- [#16129](https://github.com/AliceO2Group/AliceO2/pull/16129) 2026-11-16: Stronger protection for NaN (e<m) in e-loss correction + optimization by [@shahor02](https://github.com/shahor02)
- [#16107](https://github.com/AliceO2Group/AliceO2/pull/16107) 2026-11-16: TPC: read rule for the old ClusterNative::qTot by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16138](https://github.com/AliceO2Group/AliceO2/pull/16138) 2026-11-17: adjustments for TF shifts for productions done with MC-prod-2026-v13 by [@altsybee](https://github.com/altsybee)
- [#16141](https://github.com/AliceO2Group/AliceO2/pull/16141) 2026-11-17: fix sor calculation issue for some MC tags, also consider special cases by [@altsybee](https://github.com/altsybee)
- [#16137](https://github.com/AliceO2Group/AliceO2/pull/16137) 2026-11-18: [ALICE3] TRK/FT3 digitizer:  introduced discretization of the timestamp into digits with the ALMIRA ToA timestamp format by [@atriolo](https://github.com/atriolo)
- [#16154](https://github.com/AliceO2Group/AliceO2/pull/16154) 2026-11-19: Align MC timeframes using FirstOrbitMC from MCProdInfo by [@sawenzel](https://github.com/sawenzel)
- [#16158](https://github.com/AliceO2Group/AliceO2/pull/16158) 2026-11-21: fix BC jump offset for NonUniformMuInteractionSampler by [@ehellbar](https://github.com/ehellbar)
## Changes in Detectors

- [#16073](https://github.com/AliceO2Group/AliceO2/pull/16073) 2026-11-07: EMCAL: initialise the MC kinematics reader pointer in the digitizer by [@ktf](https://github.com/ktf)
- [#16081](https://github.com/AliceO2Group/AliceO2/pull/16081) 2026-11-09: Fix MC labels carried over between sectors in the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16089](https://github.com/AliceO2Group/AliceO2/pull/16089) 2026-11-09: Report why a simulation setup library cannot be loaded by [@sawenzel](https://github.com/sawenzel)
- [#16083](https://github.com/AliceO2Group/AliceO2/pull/16083) 2026-11-09: Resolve TPC digitizer calibration objects once per sector by [@sawenzel](https://github.com/sawenzel)
- [#16085](https://github.com/AliceO2Group/AliceO2/pull/16085) 2026-11-10: Add optional per-sector seeding to the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16082](https://github.com/AliceO2Group/AliceO2/pull/16082) 2026-11-10: Do not take ownership of CCDB distortion maps in the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16095](https://github.com/AliceO2Group/AliceO2/pull/16095) 2026-11-10: Fix missing include and library for GCC 16 by [@davidrohr](https://github.com/davidrohr)
- [#16092](https://github.com/AliceO2Group/AliceO2/pull/16092) 2026-11-10: Fix the inverted and the external primary transport selection in the Stack by [@sawenzel](https://github.com/sawenzel)
- [#16106](https://github.com/AliceO2Group/AliceO2/pull/16106) 2026-11-11: Clear mTriggeredBCFT0 container at TF start by [@shahor02](https://github.com/shahor02)
- [#16105](https://github.com/AliceO2Group/AliceO2/pull/16105) 2026-11-11: Set the static TLS reserve for Geant4 inside the sim executables by [@sawenzel](https://github.com/sawenzel)
- [#16104](https://github.com/AliceO2Group/AliceO2/pull/16104) 2026-11-11: TRD: update of T0 value and possibility to disable shift in the tracklet transformer by [@glegras](https://github.com/glegras)
- [#16084](https://github.com/AliceO2Group/AliceO2/pull/16084) 2026-11-12: Add a multithreaded Geant4 mode to o2-sim by [@sawenzel](https://github.com/sawenzel)
- [#16090](https://github.com/AliceO2Group/AliceO2/pull/16090) 2026-11-12: Add a multithreaded TPC digitizer device by [@sawenzel](https://github.com/sawenzel)
- [#16118](https://github.com/AliceO2Group/AliceO2/pull/16118) 2026-11-13: [ALICE3]: Add FD3 hit by [@njacazio](https://github.com/njacazio)
- [#16122](https://github.com/AliceO2Group/AliceO2/pull/16122) 2026-11-14: [ALICE3] TF3: Store BC and relative time in cluster by [@maciacco](https://github.com/maciacco)
- [#16125](https://github.com/AliceO2Group/AliceO2/pull/16125) 2026-11-14: Fix possible OOB access in PV refitter by [@shahor02](https://github.com/shahor02)
- [#16119](https://github.com/AliceO2Group/AliceO2/pull/16119) 2026-11-15: Do not keep physical nodes when aligning the o2-sim geometry by [@sawenzel](https://github.com/sawenzel)
- [#16132](https://github.com/AliceO2Group/AliceO2/pull/16132) 2026-11-16: MUON: Fix missing include for libfmt12 and GCC 16 by [@davidrohr](https://github.com/davidrohr)
- [#16133](https://github.com/AliceO2Group/AliceO2/pull/16133) 2026-11-16: o2-sim: Fix event aborts and the exit block in VecGeom navigation by [@sawenzel](https://github.com/sawenzel)
- [#16124](https://github.com/AliceO2Group/AliceO2/pull/16124) 2026-11-16: TPC: fix pair swap when reading CalArray<PadFlags> written by the cus… by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16107](https://github.com/AliceO2Group/AliceO2/pull/16107) 2026-11-16: TPC: read rule for the old ClusterNative::qTot by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16128](https://github.com/AliceO2Group/AliceO2/pull/16128) 2026-11-16: TRD ionisation model independent of the Geant4 energy loss by [@sawenzel](https://github.com/sawenzel)
- [#16142](https://github.com/AliceO2Group/AliceO2/pull/16142) 2026-11-17: Close the nanometre gaps between EMCAL modules by [@sawenzel](https://github.com/sawenzel)
- [#16145](https://github.com/AliceO2Group/AliceO2/pull/16145) 2026-11-17: declare trigger output in ITS/MFT digit reader only when running with triggers by [@ehellbar](https://github.com/ehellbar)
- [#16143](https://github.com/AliceO2Group/AliceO2/pull/16143) 2026-11-17: Fix invalid iterator dereference in MaterialManager debug log by [@sawenzel](https://github.com/sawenzel)
- [#16146](https://github.com/AliceO2Group/AliceO2/pull/16146) 2026-11-17: Mutex-free TGeo material budget in Propagator for multithreaded users by [@shahor02](https://github.com/shahor02)
- [#16144](https://github.com/AliceO2Group/AliceO2/pull/16144) 2026-11-17: o2-sim: Tell the exited volume by its path in VecGeom navigation by [@sawenzel](https://github.com/sawenzel)
- [#16114](https://github.com/AliceO2Group/AliceO2/pull/16114) 2026-11-17: TPC SCD: explicit spline knots for createFromTrackResiduals by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16148](https://github.com/AliceO2Group/AliceO2/pull/16148) 2026-11-18: [ALICE3] postClusterSizeVsEta.C: Fix missing include by [@vkucera](https://github.com/vkucera)
- [#16150](https://github.com/AliceO2Group/AliceO2/pull/16150) 2026-11-18: [ALICE3] TF3: update default sensor parameters by [@maciacco](https://github.com/maciacco)
- [#16137](https://github.com/AliceO2Group/AliceO2/pull/16137) 2026-11-18: [ALICE3] TRK/FT3 digitizer:  introduced discretization of the timestamp into digits with the ALMIRA ToA timestamp format by [@atriolo](https://github.com/atriolo)
- [#16153](https://github.com/AliceO2Group/AliceO2/pull/16153) 2026-11-18: Avoid hiding of base Print method in HistoManager by [@shahor02](https://github.com/shahor02)
- [#16152](https://github.com/AliceO2Group/AliceO2/pull/16152) 2026-11-18: MCH: Carry the slat geometry in double by [@sawenzel](https://github.com/sawenzel)
- [#16151](https://github.com/AliceO2Group/AliceO2/pull/16151) 2026-11-18: o2-sim: Fix zero-step kills and event aborts in VecGeom navigation by [@sawenzel](https://github.com/sawenzel)
- [#16156](https://github.com/AliceO2Group/AliceO2/pull/16156) 2026-11-19: Extend checkResid w-flow to TRD and TOF residuals extraction by [@shahor02](https://github.com/shahor02)
- [#16161](https://github.com/AliceO2Group/AliceO2/pull/16161) 2026-11-20: IOTOF: Add a TF3 cluster reader by [@mpuccio](https://github.com/mpuccio)
- [#16162](https://github.com/AliceO2Group/AliceO2/pull/16162) 2026-11-21: ITS: apply vertex multiplicity cuts to ROF masks by [@mpuccio](https://github.com/mpuccio)
- [#16134](https://github.com/AliceO2Group/AliceO2/pull/16134) 2026-11-21: o2-sim: VecGeom parallel world for a misaligned ITS by [@sawenzel](https://github.com/sawenzel)
## Changes in EventVisualisation

- [#16132](https://github.com/AliceO2Group/AliceO2/pull/16132) 2026-11-16: MUON: Fix missing include for libfmt12 and GCC 16 by [@davidrohr](https://github.com/davidrohr)
## Changes in Framework

- [#16072](https://github.com/AliceO2Group/AliceO2/pull/16072) 2026-11-08: CCDB: read std::array objects written by libstdc++ on libc++ builds by [@ktf](https://github.com/ktf)
- [#16080](https://github.com/AliceO2Group/AliceO2/pull/16080) 2026-11-08: DPL Analysis: query the CCDB once per run of equal timestamps by [@ktf](https://github.com/ktf)
- [#16079](https://github.com/AliceO2Group/AliceO2/pull/16079) 2026-11-08: DPL Analysis: send CCDB tables together with the objects they refer to by [@ktf](https://github.com/ktf)
- [#16101](https://github.com/AliceO2Group/AliceO2/pull/16101) 2026-11-12: DPL: do not use an attribute value the query parser failed to read by [@ktf](https://github.com/ktf)
- [#16111](https://github.com/AliceO2Group/AliceO2/pull/16111) 2026-11-12: Hyperloop MCP: support inspecting train outputs by [@mpuccio](https://github.com/mpuccio)
- [#16116](https://github.com/AliceO2Group/AliceO2/pull/16116) 2026-11-13: DPL: avoid InheritsFrom call by [@ktf](https://github.com/ktf)
- [#16127](https://github.com/AliceO2Group/AliceO2/pull/16127) 2026-11-15: CCDB: also read std::array members written by libstdc++ on libc++ builds by [@ktf](https://github.com/ktf)
## Changes in Steer

- [#16085](https://github.com/AliceO2Group/AliceO2/pull/16085) 2026-11-10: Add optional per-sector seeding to the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16084](https://github.com/AliceO2Group/AliceO2/pull/16084) 2026-11-12: Add a multithreaded Geant4 mode to o2-sim by [@sawenzel](https://github.com/sawenzel)
- [#16090](https://github.com/AliceO2Group/AliceO2/pull/16090) 2026-11-12: Add a multithreaded TPC digitizer device by [@sawenzel](https://github.com/sawenzel)
- [#16119](https://github.com/AliceO2Group/AliceO2/pull/16119) 2026-11-15: Do not keep physical nodes when aligning the o2-sim geometry by [@sawenzel](https://github.com/sawenzel)
- [#16137](https://github.com/AliceO2Group/AliceO2/pull/16137) 2026-11-18: [ALICE3] TRK/FT3 digitizer:  introduced discretization of the timestamp into digits with the ALMIRA ToA timestamp format by [@atriolo](https://github.com/atriolo)

# Changes since 2026-10-31

## Changes in Algorithm

- [#16110](https://github.com/AliceO2Group/AliceO2/pull/16110) 2026-11-12: GPU: implement the work-group and warp collectives for Metal by [@ktf](https://github.com/ktf)
## Changes in Analysis

- [#16031](https://github.com/AliceO2Group/AliceO2/pull/16031) 2026-11-02: DPL Analysis: harden the native AOD reader against corrupt input by [@ktf](https://github.com/ktf)
- [#16032](https://github.com/AliceO2Group/AliceO2/pull/16032) 2026-11-03: DPL Analysis: cap the read buffers of the native AOD reader by [@ktf](https://github.com/ktf)
- [#16036](https://github.com/AliceO2Group/AliceO2/pull/16036) 2026-11-03: DPL Analysis: do not build the tasks a device does not run by [@ktf](https://github.com/ktf)
- [#16051](https://github.com/AliceO2Group/AliceO2/pull/16051) 2026-11-05: DPL Analysis: retry remote reads and fail over to other replicas in the native AOD reader by [@ktf](https://github.com/ktf)
- [#16061](https://github.com/AliceO2Group/AliceO2/pull/16061) 2026-11-06: DPL Analysis: look for AOD metadata up to the allowed parent level by [@ktf](https://github.com/ktf)
- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
- [#16080](https://github.com/AliceO2Group/AliceO2/pull/16080) 2026-11-08: DPL Analysis: query the CCDB once per run of equal timestamps by [@ktf](https://github.com/ktf)
- [#16079](https://github.com/AliceO2Group/AliceO2/pull/16079) 2026-11-08: DPL Analysis: send CCDB tables together with the objects they refer to by [@ktf](https://github.com/ktf)
## Changes in Common

- [#16057](https://github.com/AliceO2Group/AliceO2/pull/16057) 2026-11-07: Make the magnetic field evaluation reentrant by [@sawenzel](https://github.com/sawenzel)
- [#16078](https://github.com/AliceO2Group/AliceO2/pull/16078) 2026-11-09: FileFetcher acknowledges suppression of processed files by [@shahor02](https://github.com/shahor02)
- [#16085](https://github.com/AliceO2Group/AliceO2/pull/16085) 2026-11-10: Add optional per-sector seeding to the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16094](https://github.com/AliceO2Group/AliceO2/pull/16094) 2026-11-10: Fix deprecated use of localtime by [@davidrohr](https://github.com/davidrohr)
- [#16088](https://github.com/AliceO2Group/AliceO2/pull/16088) 2026-11-11: DPL: when updating params provenance, check if the iterator is not at the end by [@aalkin](https://github.com/aalkin)
- [#16084](https://github.com/AliceO2Group/AliceO2/pull/16084) 2026-11-12: Add a multithreaded Geant4 mode to o2-sim by [@sawenzel](https://github.com/sawenzel)
- [#16110](https://github.com/AliceO2Group/AliceO2/pull/16110) 2026-11-12: GPU: implement the work-group and warp collectives for Metal by [@ktf](https://github.com/ktf)
- [#16118](https://github.com/AliceO2Group/AliceO2/pull/16118) 2026-11-13: [ALICE3]: Add FD3 hit by [@njacazio](https://github.com/njacazio)
## Changes in DataFormats

- [#16035](https://github.com/AliceO2Group/AliceO2/pull/16035) 2026-11-03: TrackCosmicsExt.Reader pushes TF ID info only if explicitly requested by [@shahor02](https://github.com/shahor02)
- [#16095](https://github.com/AliceO2Group/AliceO2/pull/16095) 2026-11-10: Fix missing include and library for GCC 16 by [@davidrohr](https://github.com/davidrohr)
- [#16118](https://github.com/AliceO2Group/AliceO2/pull/16118) 2026-11-13: [ALICE3]: Add FD3 hit by [@njacazio](https://github.com/njacazio)
- [#16122](https://github.com/AliceO2Group/AliceO2/pull/16122) 2026-11-14: [ALICE3] TF3: Store BC and relative time in cluster by [@maciacco](https://github.com/maciacco)
## Changes in Detectors

- [#16029](https://github.com/AliceO2Group/AliceO2/pull/16029) 2026-11-01: CheckResidSpec prints its config from 1st lane by [@shahor02](https://github.com/shahor02)
- [#16030](https://github.com/AliceO2Group/AliceO2/pull/16030) 2026-11-01: TPC SCD: fix getDXI off-by-one for row 127 by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16041](https://github.com/AliceO2Group/AliceO2/pull/16041) 2026-11-03: [ALICE3] TRKFT3 digitizer: compensate for the shift of the reference BC when mCollisionTimeWrtROF < 0 by [@atriolo](https://github.com/atriolo)
- [#16039](https://github.com/AliceO2Group/AliceO2/pull/16039) 2026-11-03: ALICE3: TRK and FT3 topology dictionaries per layer, their builder an… by [@f3sch](https://github.com/f3sch)
- [#16044](https://github.com/AliceO2Group/AliceO2/pull/16044) 2026-11-03: Remove TPC-only tracks from allowed sources of CheckResid by [@shahor02](https://github.com/shahor02)
- [#16043](https://github.com/AliceO2Group/AliceO2/pull/16043) 2026-11-03: suppress completely tfidinfo from TrackCosmicsExtendedReaderSpec by [@shahor02](https://github.com/shahor02)
- [#16040](https://github.com/AliceO2Group/AliceO2/pull/16040) 2026-11-03: TPC SCD: TrackResiduals y/x grid edge from the pad plane by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16035](https://github.com/AliceO2Group/AliceO2/pull/16035) 2026-11-03: TrackCosmicsExt.Reader pushes TF ID info only if explicitly requested by [@shahor02](https://github.com/shahor02)
- [#16033](https://github.com/AliceO2Group/AliceO2/pull/16033) 2026-11-04: ALICE3: TRK/FT3 digitizer response matrix on the heap, not the stack by [@f3sch](https://github.com/f3sch)
- [#16037](https://github.com/AliceO2Group/AliceO2/pull/16037) 2026-11-04: GenericTracking: start and import BoundedAllocator by [@mpuccio](https://github.com/mpuccio)
- [#16046](https://github.com/AliceO2Group/AliceO2/pull/16046) 2026-11-04: TPC SCD: round instead of truncate when packing the unbinned residuals by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16050](https://github.com/AliceO2Group/AliceO2/pull/16050) 2026-11-05: [ALICE3] TRK/FT3 digitizer: noise rate set in hits/px/s instead of hits/px/ROF + uniform ToA for noise digits within ROF  by [@atriolo](https://github.com/atriolo)
- [#16053](https://github.com/AliceO2Group/AliceO2/pull/16053) 2026-11-05: HBFUtilsInitializer throws if tfidinfo vector is empty by [@shahor02](https://github.com/shahor02)
- [#16063](https://github.com/AliceO2Group/AliceO2/pull/16063) 2026-11-07: [ALICE3] VD: fixed digitization for the kIRIS4a and kIRISFullCyl geometries by [@atriolo](https://github.com/atriolo)
- [#16055](https://github.com/AliceO2Group/AliceO2/pull/16055) 2026-11-07: Draw transport random numbers from a per-thread generator by [@sawenzel](https://github.com/sawenzel)
- [#16073](https://github.com/AliceO2Group/AliceO2/pull/16073) 2026-11-07: EMCAL: initialise the MC kinematics reader pointer in the digitizer by [@ktf](https://github.com/ktf)
- [#16056](https://github.com/AliceO2Group/AliceO2/pull/16056) 2026-11-07: Keep stepping state in detector members instead of statics by [@sawenzel](https://github.com/sawenzel)
- [#16060](https://github.com/AliceO2Group/AliceO2/pull/16060) 2026-11-07: MFT: track writer by [@f3sch](https://github.com/f3sch)
- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
- [#16081](https://github.com/AliceO2Group/AliceO2/pull/16081) 2026-11-09: Fix MC labels carried over between sectors in the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16089](https://github.com/AliceO2Group/AliceO2/pull/16089) 2026-11-09: Report why a simulation setup library cannot be loaded by [@sawenzel](https://github.com/sawenzel)
- [#16083](https://github.com/AliceO2Group/AliceO2/pull/16083) 2026-11-09: Resolve TPC digitizer calibration objects once per sector by [@sawenzel](https://github.com/sawenzel)
- [#16085](https://github.com/AliceO2Group/AliceO2/pull/16085) 2026-11-10: Add optional per-sector seeding to the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16082](https://github.com/AliceO2Group/AliceO2/pull/16082) 2026-11-10: Do not take ownership of CCDB distortion maps in the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16095](https://github.com/AliceO2Group/AliceO2/pull/16095) 2026-11-10: Fix missing include and library for GCC 16 by [@davidrohr](https://github.com/davidrohr)
- [#16092](https://github.com/AliceO2Group/AliceO2/pull/16092) 2026-11-10: Fix the inverted and the external primary transport selection in the Stack by [@sawenzel](https://github.com/sawenzel)
- [#16106](https://github.com/AliceO2Group/AliceO2/pull/16106) 2026-11-11: Clear mTriggeredBCFT0 container at TF start by [@shahor02](https://github.com/shahor02)
- [#16105](https://github.com/AliceO2Group/AliceO2/pull/16105) 2026-11-11: Set the static TLS reserve for Geant4 inside the sim executables by [@sawenzel](https://github.com/sawenzel)
- [#16028](https://github.com/AliceO2Group/AliceO2/pull/16028) 2026-11-11: TPC: createFromLocalCorrection with irregular spline knots by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16027](https://github.com/AliceO2Group/AliceO2/pull/16027) 2026-11-11: TPC: rate-limit TPCScaler weight errors by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#16104](https://github.com/AliceO2Group/AliceO2/pull/16104) 2026-11-11: TRD: update of T0 value and possibility to disable shift in the tracklet transformer by [@glegras](https://github.com/glegras)
- [#16084](https://github.com/AliceO2Group/AliceO2/pull/16084) 2026-11-12: Add a multithreaded Geant4 mode to o2-sim by [@sawenzel](https://github.com/sawenzel)
- [#16090](https://github.com/AliceO2Group/AliceO2/pull/16090) 2026-11-12: Add a multithreaded TPC digitizer device by [@sawenzel](https://github.com/sawenzel)
- [#16118](https://github.com/AliceO2Group/AliceO2/pull/16118) 2026-11-13: [ALICE3]: Add FD3 hit by [@njacazio](https://github.com/njacazio)
- [#16122](https://github.com/AliceO2Group/AliceO2/pull/16122) 2026-11-14: [ALICE3] TF3: Store BC and relative time in cluster by [@maciacco](https://github.com/maciacco)
- [#16125](https://github.com/AliceO2Group/AliceO2/pull/16125) 2026-11-14: Fix possible OOB access in PV refitter by [@shahor02](https://github.com/shahor02)
## Changes in Framework

- [#16034](https://github.com/AliceO2Group/AliceO2/pull/16034) 2026-11-02: AODNative: link Threads::Threads publicly by [@mhemmer-cern](https://github.com/mhemmer-cern)
- [#16031](https://github.com/AliceO2Group/AliceO2/pull/16031) 2026-11-02: DPL Analysis: harden the native AOD reader against corrupt input by [@ktf](https://github.com/ktf)
- [#16032](https://github.com/AliceO2Group/AliceO2/pull/16032) 2026-11-03: DPL Analysis: cap the read buffers of the native AOD reader by [@ktf](https://github.com/ktf)
- [#16036](https://github.com/AliceO2Group/AliceO2/pull/16036) 2026-11-03: DPL Analysis: do not build the tasks a device does not run by [@ktf](https://github.com/ktf)
- [#16051](https://github.com/AliceO2Group/AliceO2/pull/16051) 2026-11-05: DPL Analysis: retry remote reads and fail over to other replicas in the native AOD reader by [@ktf](https://github.com/ktf)
- [#16061](https://github.com/AliceO2Group/AliceO2/pull/16061) 2026-11-06: DPL Analysis: look for AOD metadata up to the allowed parent level by [@ktf](https://github.com/ktf)
- [#16072](https://github.com/AliceO2Group/AliceO2/pull/16072) 2026-11-08: CCDB: read std::array objects written by libstdc++ on libc++ builds by [@ktf](https://github.com/ktf)
- [#16080](https://github.com/AliceO2Group/AliceO2/pull/16080) 2026-11-08: DPL Analysis: query the CCDB once per run of equal timestamps by [@ktf](https://github.com/ktf)
- [#16079](https://github.com/AliceO2Group/AliceO2/pull/16079) 2026-11-08: DPL Analysis: send CCDB tables together with the objects they refer to by [@ktf](https://github.com/ktf)
- [#16101](https://github.com/AliceO2Group/AliceO2/pull/16101) 2026-11-12: DPL: do not use an attribute value the query parser failed to read by [@ktf](https://github.com/ktf)
- [#16111](https://github.com/AliceO2Group/AliceO2/pull/16111) 2026-11-12: Hyperloop MCP: support inspecting train outputs by [@mpuccio](https://github.com/mpuccio)
- [#16116](https://github.com/AliceO2Group/AliceO2/pull/16116) 2026-11-13: DPL: avoid InheritsFrom call by [@ktf](https://github.com/ktf)
## Changes in Generators

- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
## Changes in Steer

- [#16050](https://github.com/AliceO2Group/AliceO2/pull/16050) 2026-11-05: [ALICE3] TRK/FT3 digitizer: noise rate set in hits/px/s instead of hits/px/ROF + uniform ToA for noise digits within ROF  by [@atriolo](https://github.com/atriolo)
- [#16055](https://github.com/AliceO2Group/AliceO2/pull/16055) 2026-11-07: Draw transport random numbers from a per-thread generator by [@sawenzel](https://github.com/sawenzel)
- [#16056](https://github.com/AliceO2Group/AliceO2/pull/16056) 2026-11-07: Keep stepping state in detector members instead of statics by [@sawenzel](https://github.com/sawenzel)
- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
- [#16085](https://github.com/AliceO2Group/AliceO2/pull/16085) 2026-11-10: Add optional per-sector seeding to the TPC digitizer by [@sawenzel](https://github.com/sawenzel)
- [#16084](https://github.com/AliceO2Group/AliceO2/pull/16084) 2026-11-12: Add a multithreaded Geant4 mode to o2-sim by [@sawenzel](https://github.com/sawenzel)
- [#16090](https://github.com/AliceO2Group/AliceO2/pull/16090) 2026-11-12: Add a multithreaded TPC digitizer device by [@sawenzel](https://github.com/sawenzel)
## Changes in Utilities

- [#16047](https://github.com/AliceO2Group/AliceO2/pull/16047) 2026-11-04: Avoid having OpenMP as a dependency of the whole O2 by [@ktf](https://github.com/ktf)

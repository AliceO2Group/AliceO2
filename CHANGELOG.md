# Changes since 2026-10-24

## Changes in Analysis

- [#16019](https://github.com/AliceO2Group/AliceO2/pull/16019) 2026-11-01: DPL Analysis: standalone AOD reader which by-passes root by [@ktf](https://github.com/ktf)
- [#16031](https://github.com/AliceO2Group/AliceO2/pull/16031) 2026-11-02: DPL Analysis: harden the native AOD reader against corrupt input by [@ktf](https://github.com/ktf)
- [#16032](https://github.com/AliceO2Group/AliceO2/pull/16032) 2026-11-03: DPL Analysis: cap the read buffers of the native AOD reader by [@ktf](https://github.com/ktf)
- [#16036](https://github.com/AliceO2Group/AliceO2/pull/16036) 2026-11-03: DPL Analysis: do not build the tasks a device does not run by [@ktf](https://github.com/ktf)
- [#16051](https://github.com/AliceO2Group/AliceO2/pull/16051) 2026-11-05: DPL Analysis: retry remote reads and fail over to other replicas in the native AOD reader by [@ktf](https://github.com/ktf)
- [#15978](https://github.com/AliceO2Group/AliceO2/pull/15978) 2026-11-06: Detect failed reads when a tree is missing from a DF by [@autumn-mck](https://github.com/autumn-mck)
- [#16061](https://github.com/AliceO2Group/AliceO2/pull/16061) 2026-11-06: DPL Analysis: look for AOD metadata up to the allowed parent level by [@ktf](https://github.com/ktf)
- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
## Changes in Common

- [#15999](https://github.com/AliceO2Group/AliceO2/pull/15999) 2026-10-24: TrackTools: shared TrackGeometry utilities for track acceptance and PID studies" by [@miranov25](https://github.com/miranov25)
- [#16057](https://github.com/AliceO2Group/AliceO2/pull/16057) 2026-11-07: Make the magnetic field evaluation reentrant by [@sawenzel](https://github.com/sawenzel)
## Changes in DataFormats

- [#16012](https://github.com/AliceO2Group/AliceO2/pull/16012) 2026-10-27: [ALICE3] TRKFT3: first implementation to store Time of Arrival into digits by [@atriolo](https://github.com/atriolo)
- [#16017](https://github.com/AliceO2Group/AliceO2/pull/16017) 2026-10-28: TrackCosmicExtended reader, attach to InputHelper and RecoContainer by [@shahor02](https://github.com/shahor02)
- [#16035](https://github.com/AliceO2Group/AliceO2/pull/16035) 2026-11-03: TrackCosmicsExt.Reader pushes TF ID info only if explicitly requested by [@shahor02](https://github.com/shahor02)
## Changes in Detectors

- [#16002](https://github.com/AliceO2Group/AliceO2/pull/16002) 2026-10-24: [ALICE3] TF3: fetch pixel maps inside digitizer workflow by [@maciacco](https://github.com/maciacco)
- [#16003](https://github.com/AliceO2Group/AliceO2/pull/16003) 2026-10-24: Add optional residuals from IB/OB tracks extrapolations to OB/IB clusters by [@shahor02](https://github.com/shahor02)
- [#16006](https://github.com/AliceO2Group/AliceO2/pull/16006) 2026-10-25: CheckResid: add per layer chip residuals and pulls by [@shahor02](https://github.com/shahor02)
- [#16004](https://github.com/AliceO2Group/AliceO2/pull/16004) 2026-10-25: Protect HistoManager from OOB access by [@shahor02](https://github.com/shahor02)
- [#15981](https://github.com/AliceO2Group/AliceO2/pull/15981) 2026-10-26: [ALICE3] TF3: double number of TDC pairs + add RO column masking by [@maciacco](https://github.com/maciacco)
- [#15998](https://github.com/AliceO2Group/AliceO2/pull/15998) 2026-10-26: [ALICE3] TF3: geometry update steered by the final definition of the chip size and module architecture by [@maciacco](https://github.com/maciacco)
- [#16009](https://github.com/AliceO2Group/AliceO2/pull/16009) 2026-10-26: [ALICE3] TRK update: New radii for the simplified-realistic OT paving by [@bulukutlu](https://github.com/bulukutlu)
- [#16010](https://github.com/AliceO2Group/AliceO2/pull/16010) 2026-10-26: Fix display fans across wire-trimmed quadric patches by [@sawenzel](https://github.com/sawenzel)
- [#16012](https://github.com/AliceO2Group/AliceO2/pull/16012) 2026-10-27: [ALICE3] TRKFT3: first implementation to store Time of Arrival into digits by [@atriolo](https://github.com/atriolo)
- [#16007](https://github.com/AliceO2Group/AliceO2/pull/16007) 2026-10-27: Allow use of CA on limited amount of layers by [@mpuccio](https://github.com/mpuccio)
- [#16011](https://github.com/AliceO2Group/AliceO2/pull/16011) 2026-10-27: ITS/MFT digi: fix for late hits seen by the next ROF by [@shahor02](https://github.com/shahor02)
- [#16013](https://github.com/AliceO2Group/AliceO2/pull/16013) 2026-10-27: Leave holes open in the CADSupport display mesh by [@sawenzel](https://github.com/sawenzel)
- [#16020](https://github.com/AliceO2Group/AliceO2/pull/16020) 2026-10-28: Temporarily disable ITS3 alignment-related stuff by [@shahor02](https://github.com/shahor02)
- [#16018](https://github.com/AliceO2Group/AliceO2/pull/16018) 2026-10-28: TPC TimeSeries: TRD − TPC residuals at a common plane, optional track parameters, TRD readout flag by [@miranov25](https://github.com/miranov25)
- [#16017](https://github.com/AliceO2Group/AliceO2/pull/16017) 2026-10-28: TrackCosmicExtended reader, attach to InputHelper and RecoContainer by [@shahor02](https://github.com/shahor02)
- [#16016](https://github.com/AliceO2Group/AliceO2/pull/16016) 2026-10-29: ITS/MFT digi: account BC reassignment in time wrt BC by [@shahor02](https://github.com/shahor02)
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
- [#16073](https://github.com/AliceO2Group/AliceO2/pull/16073) 2026-11-07: EMCAL: initialise the MC kinematics reader pointer in the digitizer by [@ktf](https://github.com/ktf)
- [#16056](https://github.com/AliceO2Group/AliceO2/pull/16056) 2026-11-07: Keep stepping state in detector members instead of statics by [@sawenzel](https://github.com/sawenzel)
- [#16060](https://github.com/AliceO2Group/AliceO2/pull/16060) 2026-11-07: MFT: track writer by [@f3sch](https://github.com/f3sch)
- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
## Changes in Framework

- [#16001](https://github.com/AliceO2Group/AliceO2/pull/16001) 2026-10-24: DPL CCDB fetcher was ignorint condition-timestamp option by [@shahor02](https://github.com/shahor02)
- [#16023](https://github.com/AliceO2Group/AliceO2/pull/16023) 2026-10-30: DPL: add support to retrieve smaps of running processes by [@ktf](https://github.com/ktf)
- [#16019](https://github.com/AliceO2Group/AliceO2/pull/16019) 2026-11-01: DPL Analysis: standalone AOD reader which by-passes root by [@ktf](https://github.com/ktf)
- [#16034](https://github.com/AliceO2Group/AliceO2/pull/16034) 2026-11-02: AODNative: link Threads::Threads publicly by [@mhemmer-cern](https://github.com/mhemmer-cern)
- [#16031](https://github.com/AliceO2Group/AliceO2/pull/16031) 2026-11-02: DPL Analysis: harden the native AOD reader against corrupt input by [@ktf](https://github.com/ktf)
- [#16032](https://github.com/AliceO2Group/AliceO2/pull/16032) 2026-11-03: DPL Analysis: cap the read buffers of the native AOD reader by [@ktf](https://github.com/ktf)
- [#16036](https://github.com/AliceO2Group/AliceO2/pull/16036) 2026-11-03: DPL Analysis: do not build the tasks a device does not run by [@ktf](https://github.com/ktf)
- [#16051](https://github.com/AliceO2Group/AliceO2/pull/16051) 2026-11-05: DPL Analysis: retry remote reads and fail over to other replicas in the native AOD reader by [@ktf](https://github.com/ktf)
- [#15978](https://github.com/AliceO2Group/AliceO2/pull/15978) 2026-11-06: Detect failed reads when a tree is missing from a DF by [@autumn-mck](https://github.com/autumn-mck)
- [#16061](https://github.com/AliceO2Group/AliceO2/pull/16061) 2026-11-06: DPL Analysis: look for AOD metadata up to the allowed parent level by [@ktf](https://github.com/ktf)
## Changes in Generators

- [#16008](https://github.com/AliceO2Group/AliceO2/pull/16008) 2026-10-27: Apply correct TBB shutdown mechanism by [@jackal1-66](https://github.com/jackal1-66)
- [#16015](https://github.com/AliceO2Group/AliceO2/pull/16015) 2026-10-30: Avoid multi-loopers generation when using embedding by [@jackal1-66](https://github.com/jackal1-66)
- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
## Changes in Steer

- [#16002](https://github.com/AliceO2Group/AliceO2/pull/16002) 2026-10-24: [ALICE3] TF3: fetch pixel maps inside digitizer workflow by [@maciacco](https://github.com/maciacco)
- [#16050](https://github.com/AliceO2Group/AliceO2/pull/16050) 2026-11-05: [ALICE3] TRK/FT3 digitizer: noise rate set in hits/px/s instead of hits/px/ROF + uniform ToA for noise digits within ROF  by [@atriolo](https://github.com/atriolo)
- [#16056](https://github.com/AliceO2Group/AliceO2/pull/16056) 2026-11-07: Keep stepping state in detector members instead of statics by [@sawenzel](https://github.com/sawenzel)
- [#16054](https://github.com/AliceO2Group/AliceO2/pull/16054) 2026-11-07: Move the FairRoot simulation base into O2 by [@sawenzel](https://github.com/sawenzel)
## Changes in Utilities

- [#16047](https://github.com/AliceO2Group/AliceO2/pull/16047) 2026-11-04: Avoid having OpenMP as a dependency of the whole O2 by [@ktf](https://github.com/ktf)

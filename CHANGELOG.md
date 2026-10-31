# Changes since 2026-10-17

## Changes in Algorithm

- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
- [#15985](https://github.com/AliceO2Group/AliceO2/pull/15985) 2026-10-20: GPUCommonAlgorithm: partitionPoint by [@f3sch](https://github.com/f3sch)
## Changes in Analysis

- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15976](https://github.com/AliceO2Group/AliceO2/pull/15976) 2026-10-17: DPL: hoist std::views::filter to speedup compilation by [@ktf](https://github.com/ktf)
- [#15971](https://github.com/AliceO2Group/AliceO2/pull/15971) 2026-10-27: DPL Analysis: distinct ccdb path configurables by [@aalkin](https://github.com/aalkin)
## Changes in Common

- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
- [#15985](https://github.com/AliceO2Group/AliceO2/pull/15985) 2026-10-20: GPUCommonAlgorithm: partitionPoint by [@f3sch](https://github.com/f3sch)
- [#15986](https://github.com/AliceO2Group/AliceO2/pull/15986) 2026-10-20: GPUCommonMath: double functions, 64-bit bit counts, atomic load/store… by [@f3sch](https://github.com/f3sch)
- [#15990](https://github.com/AliceO2Group/AliceO2/pull/15990) 2026-10-21: TrackParametrization: use double math functions for TrackParD by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15991](https://github.com/AliceO2Group/AliceO2/pull/15991) 2026-10-22: GPUCommonDefAPI: add GPUhdii() by [@f3sch](https://github.com/f3sch)
- [#15999](https://github.com/AliceO2Group/AliceO2/pull/15999) 2026-10-24: TrackTools: shared TrackGeometry utilities for track acceptance and PID studies" by [@miranov25](https://github.com/miranov25)
## Changes in DataFormats

- [#15988](https://github.com/AliceO2Group/AliceO2/pull/15988) 2026-10-20: BaseCluster: initialise errors to 0 by default by [@hyungjun-lee](https://github.com/hyungjun-lee)
- [#15987](https://github.com/AliceO2Group/AliceO2/pull/15987) 2026-10-20: Cosmics cluster collector: store TPC occupancy per cosmic and per TF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15990](https://github.com/AliceO2Group/AliceO2/pull/15990) 2026-10-21: TrackParametrization: use double math functions for TrackParD by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15996](https://github.com/AliceO2Group/AliceO2/pull/15996) 2026-10-22: Add contributesToAHit flag to MCTrack by [@sawenzel](https://github.com/sawenzel)
- [#16012](https://github.com/AliceO2Group/AliceO2/pull/16012) 2026-10-27: [ALICE3] TRKFT3: first implementation to store Time of Arrival into digits by [@atriolo](https://github.com/atriolo)
- [#16017](https://github.com/AliceO2Group/AliceO2/pull/16017) 2026-10-28: TrackCosmicExtended reader, attach to InputHelper and RecoContainer by [@shahor02](https://github.com/shahor02)
## Changes in Detectors

- [#15959](https://github.com/AliceO2Group/AliceO2/pull/15959) 2026-10-17: [ITS] Track Follower: use getPhiPos() for the phi search window by [@AizatDaribayeva](https://github.com/AizatDaribayeva)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
- [#15957](https://github.com/AliceO2Group/AliceO2/pull/15957) 2026-10-17: ITS2 - anomalous cluster task improvement  by [@nicolovalle](https://github.com/nicolovalle)
- [#15979](https://github.com/AliceO2Group/AliceO2/pull/15979) 2026-10-20: [ALICE3] TRK: modified macros to check digits and clusters.  by [@atriolo](https://github.com/atriolo)
- [#15987](https://github.com/AliceO2Group/AliceO2/pull/15987) 2026-10-20: Cosmics cluster collector: store TPC occupancy per cosmic and per TF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15972](https://github.com/AliceO2Group/AliceO2/pull/15972) 2026-10-21: ALICE3: add FT3 clusters, reader and writer by [@f3sch](https://github.com/f3sch)
- [#15994](https://github.com/AliceO2Group/AliceO2/pull/15994) 2026-10-21: TPC: fix the weight and grid scaling of the spline derivatives in addCorrections by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15989](https://github.com/AliceO2Group/AliceO2/pull/15989) 2026-10-21: TPC: store run and time range in the voxRes UserInfo by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15992](https://github.com/AliceO2Group/AliceO2/pull/15992) 2026-10-22: [ALICE3] TRK: rInnRow column for staggered ML layers in the config file by [@lorber98](https://github.com/lorber98)
- [#15996](https://github.com/AliceO2Group/AliceO2/pull/15996) 2026-10-22: Add contributesToAHit flag to MCTrack by [@sawenzel](https://github.com/sawenzel)
- [#16000](https://github.com/AliceO2Group/AliceO2/pull/16000) 2026-10-23: CheckResid stores cl.size and track snp for points by [@shahor02](https://github.com/shahor02)
- [#15997](https://github.com/AliceO2Group/AliceO2/pull/15997) 2026-10-23: Extending track references to other detectors by [@tubagundem](https://github.com/tubagundem)
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
## Changes in Framework

- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15976](https://github.com/AliceO2Group/AliceO2/pull/15976) 2026-10-17: DPL: hoist std::views::filter to speedup compilation by [@ktf](https://github.com/ktf)
- [#15980](https://github.com/AliceO2Group/AliceO2/pull/15980) 2026-10-17: DPL: out of line HistogramSpec related destructors by [@ktf](https://github.com/ktf)
- [#16001](https://github.com/AliceO2Group/AliceO2/pull/16001) 2026-10-24: DPL CCDB fetcher was ignorint condition-timestamp option by [@shahor02](https://github.com/shahor02)
- [#15971](https://github.com/AliceO2Group/AliceO2/pull/15971) 2026-10-27: DPL Analysis: distinct ccdb path configurables by [@aalkin](https://github.com/aalkin)
- [#15974](https://github.com/AliceO2Group/AliceO2/pull/15974) 2026-10-28: DPL Analysis: update the ccdb paths options from configuration by [@aalkin](https://github.com/aalkin)
- [#15927](https://github.com/AliceO2Group/AliceO2/pull/15927) 2026-10-28: DPL: Allow overriding array values in ConfigParamRegistry by [@aalkin](https://github.com/aalkin)
- [#16023](https://github.com/AliceO2Group/AliceO2/pull/16023) 2026-10-30: DPL: add support to retrieve smaps of running processes by [@ktf](https://github.com/ktf)
## Changes in Generators

- [#15977](https://github.com/AliceO2Group/AliceO2/pull/15977) 2026-10-19: Silence importing task line on hyperloop by [@jackal1-66](https://github.com/jackal1-66)
- [#15995](https://github.com/AliceO2Group/AliceO2/pull/15995) 2026-10-22: Decay generator particles before transport by [@sawenzel](https://github.com/sawenzel)
- [#16008](https://github.com/AliceO2Group/AliceO2/pull/16008) 2026-10-27: Apply correct TBB shutdown mechanism by [@jackal1-66](https://github.com/jackal1-66)
- [#16015](https://github.com/AliceO2Group/AliceO2/pull/16015) 2026-10-30: Avoid multi-loopers generation when using embedding by [@jackal1-66](https://github.com/jackal1-66)
## Changes in Steer

- [#15935](https://github.com/AliceO2Group/AliceO2/pull/15935) 2026-10-19: Skip combined device creation for empty spec list in DigitizerWorkflow by [@jackal1-66](https://github.com/jackal1-66)
- [#15972](https://github.com/AliceO2Group/AliceO2/pull/15972) 2026-10-21: ALICE3: add FT3 clusters, reader and writer by [@f3sch](https://github.com/f3sch)
- [#16002](https://github.com/AliceO2Group/AliceO2/pull/16002) 2026-10-24: [ALICE3] TF3: fetch pixel maps inside digitizer workflow by [@maciacco](https://github.com/maciacco)

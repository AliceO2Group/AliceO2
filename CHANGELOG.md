# Changes since 2026-10-10

## Changes in Algorithm

- [#15920](https://github.com/AliceO2Group/AliceO2/pull/15920) 2026-10-10: GPU: pass the work-item indices to the helpers that lack them by [@ktf](https://github.com/ktf)
- [#15933](https://github.com/AliceO2Group/AliceO2/pull/15933) 2026-10-11: GPU: Faster QuickSort implementation for large data, and some cleanup of option names and help / debug messages by [@davidrohr](https://github.com/davidrohr)
- [#15929](https://github.com/AliceO2Group/AliceO2/pull/15929) 2026-10-12: GPU: Add CPUDevice backend to emulate device-like processing on host by [@davidrohr](https://github.com/davidrohr)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
- [#15985](https://github.com/AliceO2Group/AliceO2/pull/15985) 2026-10-20: GPUCommonAlgorithm: partitionPoint by [@f3sch](https://github.com/f3sch)
## Changes in Analysis

- [#15952](https://github.com/AliceO2Group/AliceO2/pull/15952) 2026-10-13: DPL Analysis: keep reading after a skipped timeframe by [@autumn-mck](https://github.com/autumn-mck)
- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15976](https://github.com/AliceO2Group/AliceO2/pull/15976) 2026-10-17: DPL: hoist std::views::filter to speedup compilation by [@ktf](https://github.com/ktf)
## Changes in Common

- [#15920](https://github.com/AliceO2Group/AliceO2/pull/15920) 2026-10-10: GPU: pass the work-item indices to the helpers that lack them by [@ktf](https://github.com/ktf)
- [#15923](https://github.com/AliceO2Group/AliceO2/pull/15923) 2026-10-10: MathUtils: extend the OpenCL sincos workaround to Metal by [@ktf](https://github.com/ktf)
- [#15933](https://github.com/AliceO2Group/AliceO2/pull/15933) 2026-10-11: GPU: Faster QuickSort implementation for large data, and some cleanup of option names and help / debug messages by [@davidrohr](https://github.com/davidrohr)
- [#15932](https://github.com/AliceO2Group/AliceO2/pull/15932) 2026-10-11: GPU: give Metal its own spellings for two math helpers by [@ktf](https://github.com/ktf)
- [#15934](https://github.com/AliceO2Group/AliceO2/pull/15934) 2026-10-12: Fix missing include by [@davidrohr](https://github.com/davidrohr)
- [#15929](https://github.com/AliceO2Group/AliceO2/pull/15929) 2026-10-12: GPU: Add CPUDevice backend to emulate device-like processing on host by [@davidrohr](https://github.com/davidrohr)
- [#15936](https://github.com/AliceO2Group/AliceO2/pull/15936) 2026-10-12: GPU: reach global memory through the generic address space on Metal by [@ktf](https://github.com/ktf)
- [#15946](https://github.com/AliceO2Group/AliceO2/pull/15946) 2026-10-12: MathUtils: let bringTo* deduce the type of the call they forward to by [@ktf](https://github.com/ktf)
- [#15947](https://github.com/AliceO2Group/AliceO2/pull/15947) 2026-10-12: MathUtils: make getMean()'s ternary unambiguous for an emulated double by [@ktf](https://github.com/ktf)
- [#15944](https://github.com/AliceO2Group/AliceO2/pull/15944) 2026-10-14: GPU: make the Metal atomics operate on plain counters by [@ktf](https://github.com/ktf)
- [#15949](https://github.com/AliceO2Group/AliceO2/pull/15949) 2026-10-14: MathUtils: declare fastATan2's Pi where its lambdas can use it by [@ktf](https://github.com/ktf)
- [#15963](https://github.com/AliceO2Group/AliceO2/pull/15963) 2026-10-15: GPU: let the constant-address-space constants be built on Metal by [@ktf](https://github.com/ktf)
- [#15962](https://github.com/AliceO2Group/AliceO2/pull/15962) 2026-10-15: GPU: make std::is_pointer address-space aware on Metal by [@ktf](https://github.com/ktf)
- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
- [#15985](https://github.com/AliceO2Group/AliceO2/pull/15985) 2026-10-20: GPUCommonAlgorithm: partitionPoint by [@f3sch](https://github.com/f3sch)
- [#15986](https://github.com/AliceO2Group/AliceO2/pull/15986) 2026-10-20: GPUCommonMath: double functions, 64-bit bit counts, atomic load/store… by [@f3sch](https://github.com/f3sch)
- [#15990](https://github.com/AliceO2Group/AliceO2/pull/15990) 2026-10-21: TrackParametrization: use double math functions for TrackParD by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15991](https://github.com/AliceO2Group/AliceO2/pull/15991) 2026-10-22: GPUCommonDefAPI: add GPUhdii() by [@f3sch](https://github.com/f3sch)
## Changes in DataFormats

- [#15915](https://github.com/AliceO2Group/AliceO2/pull/15915) 2026-10-11: MatchCosmics: cosmics in collision data + raw-cluster writer with TOF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15919](https://github.com/AliceO2Group/AliceO2/pull/15919) 2026-10-12: adjustments for orbit shifts for other MCs done with MC-prod-v12 by [@altsybee](https://github.com/altsybee)
- [#15963](https://github.com/AliceO2Group/AliceO2/pull/15963) 2026-10-15: GPU: let the constant-address-space constants be built on Metal by [@ktf](https://github.com/ktf)
- [#15916](https://github.com/AliceO2Group/AliceO2/pull/15916) 2026-10-19: Make O2DatabasePDG::Instance() thread-safe and remove few PDG warnings by [@jackal1-66](https://github.com/jackal1-66)
- [#15988](https://github.com/AliceO2Group/AliceO2/pull/15988) 2026-10-20: BaseCluster: initialise errors to 0 by default by [@hyungjun-lee](https://github.com/hyungjun-lee)
- [#15987](https://github.com/AliceO2Group/AliceO2/pull/15987) 2026-10-20: Cosmics cluster collector: store TPC occupancy per cosmic and per TF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15990](https://github.com/AliceO2Group/AliceO2/pull/15990) 2026-10-21: TrackParametrization: use double math functions for TrackParD by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15996](https://github.com/AliceO2Group/AliceO2/pull/15996) 2026-10-22: Add contributesToAHit flag to MCTrack by [@sawenzel](https://github.com/sawenzel)
## Changes in Detectors

- [#15915](https://github.com/AliceO2Group/AliceO2/pull/15915) 2026-10-11: MatchCosmics: cosmics in collision data + raw-cluster writer with TOF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15954](https://github.com/AliceO2Group/AliceO2/pull/15954) 2026-10-13: Add mean res. and sigmas histo per h-barrel by [@shahor02](https://github.com/shahor02)
- [#15948](https://github.com/AliceO2Group/AliceO2/pull/15948) 2026-10-13: Add per HB histos with 1 channel/chip by [@shahor02](https://github.com/shahor02)
- [#15953](https://github.com/AliceO2Group/AliceO2/pull/15953) 2026-10-13: GRPGeomHelper optionally provides aligned geom + alignments by [@shahor02](https://github.com/shahor02)
- [#15918](https://github.com/AliceO2Group/AliceO2/pull/15918) 2026-10-13: ITS GPU seeding vertexer: add clamps to adaptive low mult threshold, and propate cells towards beamline by [@cima22](https://github.com/cima22)
- [#15943](https://github.com/AliceO2Group/AliceO2/pull/15943) 2026-10-13: TPC SCD: write and read the MC truth of the residual tracks by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15956](https://github.com/AliceO2Group/AliceO2/pull/15956) 2026-10-14: Fix misplaced default in the function by [@shahor02](https://github.com/shahor02)
- [#15912](https://github.com/AliceO2Group/AliceO2/pull/15912) 2026-10-16: TRK: IRIS service aluminium from the MaterialManager by [@f3sch](https://github.com/f3sch)
- [#15959](https://github.com/AliceO2Group/AliceO2/pull/15959) 2026-10-17: [ITS] Track Follower: use getPhiPos() for the phi search window by [@AizatDaribayeva](https://github.com/AizatDaribayeva)
- [#15960](https://github.com/AliceO2Group/AliceO2/pull/15960) 2026-10-17: ITS GPU Seeding vertexer: use warp reductions for fit peak kernel by [@cima22](https://github.com/cima22)
- [#15957](https://github.com/AliceO2Group/AliceO2/pull/15957) 2026-10-17: ITS2 - anomalous cluster task improvement  by [@nicolovalle](https://github.com/nicolovalle)
- [#15916](https://github.com/AliceO2Group/AliceO2/pull/15916) 2026-10-19: Make O2DatabasePDG::Instance() thread-safe and remove few PDG warnings by [@jackal1-66](https://github.com/jackal1-66)
- [#15979](https://github.com/AliceO2Group/AliceO2/pull/15979) 2026-10-20: [ALICE3] TRK: modified macros to check digits and clusters.  by [@atriolo](https://github.com/atriolo)
- [#15987](https://github.com/AliceO2Group/AliceO2/pull/15987) 2026-10-20: Cosmics cluster collector: store TPC occupancy per cosmic and per TF by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15972](https://github.com/AliceO2Group/AliceO2/pull/15972) 2026-10-21: ALICE3: add FT3 clusters, reader and writer by [@f3sch](https://github.com/f3sch)
- [#15994](https://github.com/AliceO2Group/AliceO2/pull/15994) 2026-10-21: TPC: fix the weight and grid scaling of the spline derivatives in addCorrections by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15989](https://github.com/AliceO2Group/AliceO2/pull/15989) 2026-10-21: TPC: store run and time range in the voxRes UserInfo by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#15992](https://github.com/AliceO2Group/AliceO2/pull/15992) 2026-10-22: [ALICE3] TRK: rInnRow column for staggered ML layers in the config file by [@lorber98](https://github.com/lorber98)
- [#15996](https://github.com/AliceO2Group/AliceO2/pull/15996) 2026-10-22: Add contributesToAHit flag to MCTrack by [@sawenzel](https://github.com/sawenzel)
- [#16000](https://github.com/AliceO2Group/AliceO2/pull/16000) 2026-10-23: CheckResid stores cl.size and track snp for points by [@shahor02](https://github.com/shahor02)
- [#15997](https://github.com/AliceO2Group/AliceO2/pull/15997) 2026-10-23: Extending track references to other detectors by [@tubagundem](https://github.com/tubagundem)
## Changes in Framework

- [#15922](https://github.com/AliceO2Group/AliceO2/pull/15922) 2026-10-12: DPL: fix DataDescriptorMatcher and the associated query builder by [@ktf](https://github.com/ktf)
- [#15940](https://github.com/AliceO2Group/AliceO2/pull/15940) 2026-10-13: [Framework] Add `is_not_filtered_iterator` concept by [@mhemmer-cern](https://github.com/mhemmer-cern)
- [#15942](https://github.com/AliceO2Group/AliceO2/pull/15942) 2026-10-13: DPL Analysis: fix incompatible Join check by [@aalkin](https://github.com/aalkin)
- [#15952](https://github.com/AliceO2Group/AliceO2/pull/15952) 2026-10-13: DPL Analysis: keep reading after a skipped timeframe by [@autumn-mck](https://github.com/autumn-mck)
- [#15938](https://github.com/AliceO2Group/AliceO2/pull/15938) 2026-10-13: DPL: Fix leaks and self-assignment in Array2D by [@aalkin](https://github.com/aalkin)
- [#15941](https://github.com/AliceO2Group/AliceO2/pull/15941) 2026-10-13: DPL: reduce verbosity of the missing resources warning by [@ktf](https://github.com/ktf)
- [#15939](https://github.com/AliceO2Group/AliceO2/pull/15939) 2026-10-14: DPL: Fix ownership of the content of Variant by [@aalkin](https://github.com/aalkin)
- [#15975](https://github.com/AliceO2Group/AliceO2/pull/15975) 2026-10-16: DPL Analysis: out-of-line HistogramSpec related construction by [@ktf](https://github.com/ktf)
- [#15965](https://github.com/AliceO2Group/AliceO2/pull/15965) 2026-10-17: DPL Analysis: skip input files which cannot be opened by [@autumn-mck](https://github.com/autumn-mck)
- [#15976](https://github.com/AliceO2Group/AliceO2/pull/15976) 2026-10-17: DPL: hoist std::views::filter to speedup compilation by [@ktf](https://github.com/ktf)
- [#15980](https://github.com/AliceO2Group/AliceO2/pull/15980) 2026-10-17: DPL: out of line HistogramSpec related destructors by [@ktf](https://github.com/ktf)
- [#16001](https://github.com/AliceO2Group/AliceO2/pull/16001) 2026-10-24: DPL CCDB fetcher was ignorint condition-timestamp option by [@shahor02](https://github.com/shahor02)
## Changes in Generators

- [#15916](https://github.com/AliceO2Group/AliceO2/pull/15916) 2026-10-19: Make O2DatabasePDG::Instance() thread-safe and remove few PDG warnings by [@jackal1-66](https://github.com/jackal1-66)
- [#15977](https://github.com/AliceO2Group/AliceO2/pull/15977) 2026-10-19: Silence importing task line on hyperloop by [@jackal1-66](https://github.com/jackal1-66)
- [#15995](https://github.com/AliceO2Group/AliceO2/pull/15995) 2026-10-22: Decay generator particles before transport by [@sawenzel](https://github.com/sawenzel)
## Changes in Steer

- [#15935](https://github.com/AliceO2Group/AliceO2/pull/15935) 2026-10-19: Skip combined device creation for empty spec list in DigitizerWorkflow by [@jackal1-66](https://github.com/jackal1-66)
- [#15972](https://github.com/AliceO2Group/AliceO2/pull/15972) 2026-10-21: ALICE3: add FT3 clusters, reader and writer by [@f3sch](https://github.com/f3sch)

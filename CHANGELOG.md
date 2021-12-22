# Changes since 2021-12-08

## Changes in Algorithm

- [#7884](https://github.com/AliceO2Group/AliceO2/pull/7884) 2021-12-18: [QC-715] Mergers: forbid trees larger than 100MB by [@knopers8](https://github.com/knopers8)
## Changes in Analysis

- [#7827](https://github.com/AliceO2Group/AliceO2/pull/7827) 2021-12-14: DPL Analysis: Event mixing interface by [@saganatt](https://github.com/saganatt)
- [#7877](https://github.com/AliceO2Group/AliceO2/pull/7877) 2021-12-15: use natan2, index for ambigoustracks by [@jgrosseo](https://github.com/jgrosseo)
- [#7878](https://github.com/AliceO2Group/AliceO2/pull/7878) 2021-12-16: DPL Analysis: make sure the partitions are set once per DF by [@aalkin](https://github.com/aalkin)
- [#7849](https://github.com/AliceO2Group/AliceO2/pull/7849) 2021-12-16: Moving T0 and V0 to VLAs by [@jgrosseo](https://github.com/jgrosseo)
## Changes in Common

- [#7819](https://github.com/AliceO2Group/AliceO2/pull/7819) 2021-12-08: TRD merge CommonParam into SimParam and more by [@martenole](https://github.com/martenole)
- [#7835](https://github.com/AliceO2Group/AliceO2/pull/7835) 2021-12-08: o2-sim: Introduce persistent aligned geometry by [@sawenzel](https://github.com/sawenzel)
- [#7852](https://github.com/AliceO2Group/AliceO2/pull/7852) 2021-12-09: DPL: do not initialise InfoLogger when not requested by [@ktf](https://github.com/ktf)
- [#7850](https://github.com/AliceO2Group/AliceO2/pull/7850) 2021-12-10: Revert "To be reverted, add temporary dummy file for QC compatibility" by [@davidrohr](https://github.com/davidrohr)
- [#7861](https://github.com/AliceO2Group/AliceO2/pull/7861) 2021-12-11: Fixes/macros for new GRP objects, retrofit Pilot Beam GRPs by [@shahor02](https://github.com/shahor02)
- [#7885](https://github.com/AliceO2Group/AliceO2/pull/7885) 2021-12-18: CCDBPopulator adds runNumber (if any) tag to metadata by [@shahor02](https://github.com/shahor02)
- [#7897](https://github.com/AliceO2Group/AliceO2/pull/7897) 2021-12-21: o2-sim: Fix inconsistency in timestamp assignment by [@sawenzel](https://github.com/sawenzel)
## Changes in DataFormats

- [#7810](https://github.com/AliceO2Group/AliceO2/pull/7810) 2021-12-09: Allow PV contributors, refit with mat.corr. in S.Vertexing by [@shahor02](https://github.com/shahor02)
- [#7850](https://github.com/AliceO2Group/AliceO2/pull/7850) 2021-12-10: Revert "To be reverted, add temporary dummy file for QC compatibility" by [@davidrohr](https://github.com/davidrohr)
- [#7861](https://github.com/AliceO2Group/AliceO2/pull/7861) 2021-12-11: Fixes/macros for new GRP objects, retrofit Pilot Beam GRPs by [@shahor02](https://github.com/shahor02)
- [#7822](https://github.com/AliceO2Group/AliceO2/pull/7822) 2021-12-11: TOF full access to ccdb calibration and anchoring for MC by [@noferini](https://github.com/noferini)
- [#7888](https://github.com/AliceO2Group/AliceO2/pull/7888) 2021-12-18: Add helper method to TRD Tracklet64 by [@martenole](https://github.com/martenole)
- [#7889](https://github.com/AliceO2Group/AliceO2/pull/7889) 2021-12-18: [FT0] Fix uninitialized Members in Digits by [@MichaelLettrich](https://github.com/MichaelLettrich)
## Changes in Detectors

- [#7800](https://github.com/AliceO2Group/AliceO2/pull/7800) 2021-12-08: Optional mat.corr., full propagation and refit in DCAFitter by [@shahor02](https://github.com/shahor02)
- [#7819](https://github.com/AliceO2Group/AliceO2/pull/7819) 2021-12-08: TRD merge CommonParam into SimParam and more by [@martenole](https://github.com/martenole)
- [#7835](https://github.com/AliceO2Group/AliceO2/pull/7835) 2021-12-08: o2-sim: Introduce persistent aligned geometry by [@sawenzel](https://github.com/sawenzel)
- [#7810](https://github.com/AliceO2Group/AliceO2/pull/7810) 2021-12-09: Allow PV contributors, refit with mat.corr. in S.Vertexing by [@shahor02](https://github.com/shahor02)
- [#7834](https://github.com/AliceO2Group/AliceO2/pull/7834) 2021-12-09: MRRTF-158: more MCH elecmap functions to get nof pads and ds per FeeId by [@aphecetche](https://github.com/aphecetche)
- [#7842](https://github.com/AliceO2Group/AliceO2/pull/7842) 2021-12-09: MRRTF-161: move MCH EntropyDecoder out of MCHWorkflow lib by [@aphecetche](https://github.com/aphecetche)
- [#7846](https://github.com/AliceO2Group/AliceO2/pull/7846) 2021-12-10: Fix T0 filling by [@jgrosseo](https://github.com/jgrosseo)
- [#7860](https://github.com/AliceO2Group/AliceO2/pull/7860) 2021-12-10: Include TPC unconstrained track to S.Vertexing by [@shahor02](https://github.com/shahor02)
- [#7850](https://github.com/AliceO2Group/AliceO2/pull/7850) 2021-12-10: Revert "To be reverted, add temporary dummy file for QC compatibility" by [@davidrohr](https://github.com/davidrohr)
- [#7862](https://github.com/AliceO2Group/AliceO2/pull/7862) 2021-12-10: Set default Alpide noise to 1e-8 (was 1e-7) by [@shahor02](https://github.com/shahor02)
- [#7863](https://github.com/AliceO2Group/AliceO2/pull/7863) 2021-12-11: Fix in refit of TPC-ITSAB tracks in TRD matching by [@shahor02](https://github.com/shahor02)
- [#7831](https://github.com/AliceO2Group/AliceO2/pull/7831) 2021-12-11: GlobalFwdMatcher: add support for external matching cut function by [@rpezzi](https://github.com/rpezzi)
- [#7854](https://github.com/AliceO2Group/AliceO2/pull/7854) 2021-12-11: MRRTF-157: add the concept of trackable ROF for MCH digits by [@aphecetche](https://github.com/aphecetche)
- [#7857](https://github.com/AliceO2Group/AliceO2/pull/7857) 2021-12-11: Remove V0C from AOD producer by [@jgrosseo](https://github.com/jgrosseo)
- [#7822](https://github.com/AliceO2Group/AliceO2/pull/7822) 2021-12-11: TOF full access to ccdb calibration and anchoring for MC by [@noferini](https://github.com/noferini)
- [#7858](https://github.com/AliceO2Group/AliceO2/pull/7858) 2021-12-11: [EMCAL-582] Add static getters for EMCAL CCDB paths by [@mfasDa](https://github.com/mfasDa)
- [#7865](https://github.com/AliceO2Group/AliceO2/pull/7865) 2021-12-13: Fix in bi-square weights calculation by [@shahor02](https://github.com/shahor02)
- [#7873](https://github.com/AliceO2Group/AliceO2/pull/7873) 2021-12-14: MCH: remove unnecessary links to mapping by [@pillot](https://github.com/pillot)
- [#7872](https://github.com/AliceO2Group/AliceO2/pull/7872) 2021-12-14: Send dummy output if data was not found in TF by [@shahor02](https://github.com/shahor02)
- [#7871](https://github.com/AliceO2Group/AliceO2/pull/7871) 2021-12-14: TPC: Restructure pad-wise calibration, implement missing workflows by [@wiechula](https://github.com/wiechula)
- [#7869](https://github.com/AliceO2Group/AliceO2/pull/7869) 2021-12-14: [MCH] add protections by [@pillot](https://github.com/pillot)
- [#7880](https://github.com/AliceO2Group/AliceO2/pull/7880) 2021-12-15: Default for CCDB populator set from NameConf.mCCDBServer by [@shahor02](https://github.com/shahor02)
- [#7876](https://github.com/AliceO2Group/AliceO2/pull/7876) 2021-12-15: TPC: Make FileWriterSpec buffer data until all sectors received by [@wiechula](https://github.com/wiechula)
- [#7841](https://github.com/AliceO2Group/AliceO2/pull/7841) 2021-12-15: Use VMC standalone by [@benedikt-voelkel](https://github.com/benedikt-voelkel)
- [#7849](https://github.com/AliceO2Group/AliceO2/pull/7849) 2021-12-16: Moving T0 and V0 to VLAs by [@jgrosseo](https://github.com/jgrosseo)
- [#7888](https://github.com/AliceO2Group/AliceO2/pull/7888) 2021-12-18: Add helper method to TRD Tracklet64 by [@martenole](https://github.com/martenole)
- [#7885](https://github.com/AliceO2Group/AliceO2/pull/7885) 2021-12-18: CCDBPopulator adds runNumber (if any) tag to metadata by [@shahor02](https://github.com/shahor02)
- [#7889](https://github.com/AliceO2Group/AliceO2/pull/7889) 2021-12-18: [FT0] Fix uninitialized Members in Digits by [@MichaelLettrich](https://github.com/MichaelLettrich)
## Changes in EventVisualisation

- [#7850](https://github.com/AliceO2Group/AliceO2/pull/7850) 2021-12-10: Revert "To be reverted, add temporary dummy file for QC compatibility" by [@davidrohr](https://github.com/davidrohr)
## Changes in Framework

- [#7829](https://github.com/AliceO2Group/AliceO2/pull/7829) 2021-12-08: DPL Analysis: fix VLA reading by [@aalkin](https://github.com/aalkin)
- [#7839](https://github.com/AliceO2Group/AliceO2/pull/7839) 2021-12-08: DPL: actually populate Dict with associated ptree by [@ktf](https://github.com/ktf)
- [#7832](https://github.com/AliceO2Group/AliceO2/pull/7832) 2021-12-08: DPL: do not exit with 0 on error by [@ktf](https://github.com/ktf)
- [#7826](https://github.com/AliceO2Group/AliceO2/pull/7826) 2021-12-08: DPL: support VariantType::Dict in merged workflows by [@ktf](https://github.com/ktf)
- [#7824](https://github.com/AliceO2Group/AliceO2/pull/7824) 2021-12-08: print branch type in exception by [@jgrosseo](https://github.com/jgrosseo)
- [#7847](https://github.com/AliceO2Group/AliceO2/pull/7847) 2021-12-09: DPL Analysis: Add interleaved_pack_t and unique_pack_t by [@saganatt](https://github.com/saganatt)
- [#7852](https://github.com/AliceO2Group/AliceO2/pull/7852) 2021-12-09: DPL: do not initialise InfoLogger when not requested by [@ktf](https://github.com/ktf)
- [#7853](https://github.com/AliceO2Group/AliceO2/pull/7853) 2021-12-10: DPL: close (almost) all file descriptors before forking by [@ktf](https://github.com/ktf)
- [#7856](https://github.com/AliceO2Group/AliceO2/pull/7856) 2021-12-10: DPL: use fatal to report uncaught exceptions by [@ktf](https://github.com/ktf)
- [#7864](https://github.com/AliceO2Group/AliceO2/pull/7864) 2021-12-11: DPL: double the number of processes which can be created by the driver by [@ktf](https://github.com/ktf)
- [#7859](https://github.com/AliceO2Group/AliceO2/pull/7859) 2021-12-11: DPL: simplify output on double Ctrl-C by [@ktf](https://github.com/ktf)
- [#7827](https://github.com/AliceO2Group/AliceO2/pull/7827) 2021-12-14: DPL Analysis: Event mixing interface by [@saganatt](https://github.com/saganatt)
- [#7874](https://github.com/AliceO2Group/AliceO2/pull/7874) 2021-12-14: DPL Analysis: fix algorithm to clean inputs based on process switches by [@aalkin](https://github.com/aalkin)
- [#7868](https://github.com/AliceO2Group/AliceO2/pull/7868) 2021-12-14: DPL: fix quoting environment by [@ktf](https://github.com/ktf)
- [#7866](https://github.com/AliceO2Group/AliceO2/pull/7866) 2021-12-14: [O2-2712] DPL Analysis: add an exception on invalid index access by [@aalkin](https://github.com/aalkin)
- [#7867](https://github.com/AliceO2Group/AliceO2/pull/7867) 2021-12-15: DPL Analysis: update advanced indices by [@aalkin](https://github.com/aalkin)
- [#7877](https://github.com/AliceO2Group/AliceO2/pull/7877) 2021-12-15: use natan2, index for ambigoustracks by [@jgrosseo](https://github.com/jgrosseo)
- [#7879](https://github.com/AliceO2Group/AliceO2/pull/7879) 2021-12-16: AliECS dump: _plain_cmdline as a multiline block by [@knopers8](https://github.com/knopers8)
- [#7878](https://github.com/AliceO2Group/AliceO2/pull/7878) 2021-12-16: DPL Analysis: make sure the partitions are set once per DF by [@aalkin](https://github.com/aalkin)
- [#7882](https://github.com/AliceO2Group/AliceO2/pull/7882) 2021-12-16: DPL: force exit on error after 15 seconds by [@ktf](https://github.com/ktf)
- [#7849](https://github.com/AliceO2Group/AliceO2/pull/7849) 2021-12-16: Moving T0 and V0 to VLAs by [@jgrosseo](https://github.com/jgrosseo)
- [#7890](https://github.com/AliceO2Group/AliceO2/pull/7890) 2021-12-18: DPL Analysis: groundwork for caching the slicing information of a table by [@aalkin](https://github.com/aalkin)
- [#7891](https://github.com/AliceO2Group/AliceO2/pull/7891) 2021-12-18: Propaedeutic for CCDB - DPL integration by [@ktf](https://github.com/ktf)
- [#7893](https://github.com/AliceO2Group/AliceO2/pull/7893) 2021-12-19: Avoid new libuv API by [@ktf](https://github.com/ktf)
## Changes in Steer

- [#7835](https://github.com/AliceO2Group/AliceO2/pull/7835) 2021-12-08: o2-sim: Introduce persistent aligned geometry by [@sawenzel](https://github.com/sawenzel)
- [#7822](https://github.com/AliceO2Group/AliceO2/pull/7822) 2021-12-11: TOF full access to ccdb calibration and anchoring for MC by [@noferini](https://github.com/noferini)
## Changes in Utilities

- [#7833](https://github.com/AliceO2Group/AliceO2/pull/7833) 2021-12-09: [QC-624] Mergers: use labels to match in customizeInfrastructure by [@knopers8](https://github.com/knopers8)
- [#7884](https://github.com/AliceO2Group/AliceO2/pull/7884) 2021-12-18: [QC-715] Mergers: forbid trees larger than 100MB by [@knopers8](https://github.com/knopers8)

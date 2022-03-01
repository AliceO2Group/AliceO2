# Changes since 2022-02-14

## Changes in Analysis

- [#8166](https://github.com/AliceO2Group/AliceO2/pull/8166) 2022-02-16: Using DataRefUtils to get payload size by [@matthiasrichter](https://github.com/matthiasrichter)
- [#8182](https://github.com/AliceO2Group/AliceO2/pull/8182) 2022-02-18: DPL Analysis: allow index builder to be used with filtered input by [@aalkin](https://github.com/aalkin)
## Changes in Common

- [#8149](https://github.com/AliceO2Group/AliceO2/pull/8149) 2022-02-14: [o2-sim] En-/disable hit creation per detector by [@benedikt-voelkel](https://github.com/benedikt-voelkel)
- [#8156](https://github.com/AliceO2Group/AliceO2/pull/8156) 2022-02-16: [o2-sim] Fatal in certain cases by [@benedikt-voelkel](https://github.com/benedikt-voelkel)
- [#8158](https://github.com/AliceO2Group/AliceO2/pull/8158) 2022-02-17: Fixes for TRD GPU tracking + related and unrelated cleanup by [@davidrohr](https://github.com/davidrohr)
- [#8175](https://github.com/AliceO2Group/AliceO2/pull/8175) 2022-02-18: DPL: Get free SHM memory from correct segment, if --shmid is in command line by [@davidrohr](https://github.com/davidrohr)
## Changes in DataFormats

- [#8146](https://github.com/AliceO2Group/AliceO2/pull/8146) 2022-02-14: GPU: Fix standalone compilation after changes to TPC dEdx class by [@davidrohr](https://github.com/davidrohr)
- [#8160](https://github.com/AliceO2Group/AliceO2/pull/8160) 2022-02-15: Prototype of the reco TF skimming workflow by [@shahor02](https://github.com/shahor02)
- [#8188](https://github.com/AliceO2Group/AliceO2/pull/8188) 2022-02-20: Extend cell time range by [@peressounko](https://github.com/peressounko)
- [#8215](https://github.com/AliceO2Group/AliceO2/pull/8215) 2022-02-26: [MRRTF-146] MCH: Introduce CSV version of the Bad Channel List by [@aphecetche](https://github.com/aphecetche)
## Changes in Detectors

- [#8150](https://github.com/AliceO2Group/AliceO2/pull/8150) 2022-02-14: Cosmetic fixes for detector raw file names: cru->crorc for CRORC detector, capital detector names, correct FIT FLPs by [@davidrohr](https://github.com/davidrohr)
- [#8146](https://github.com/AliceO2Group/AliceO2/pull/8146) 2022-02-14: GPU: Fix standalone compilation after changes to TPC dEdx class by [@davidrohr](https://github.com/davidrohr)
- [#8143](https://github.com/AliceO2Group/AliceO2/pull/8143) 2022-02-14: Split of the ITS threshold wf: Calibrator + aggregator - ADDED support for no EoS case by [@iravasen](https://github.com/iravasen)
- [#8149](https://github.com/AliceO2Group/AliceO2/pull/8149) 2022-02-14: [o2-sim] En-/disable hit creation per detector by [@benedikt-voelkel](https://github.com/benedikt-voelkel)
- [#8147](https://github.com/AliceO2Group/AliceO2/pull/8147) 2022-02-14: executable to create an aligned geom from CCDB entries + loadGeometry defaults change by [@shahor02](https://github.com/shahor02)
- [#8163](https://github.com/AliceO2Group/AliceO2/pull/8163) 2022-02-15: Fixed runType in ITS calib workflow by [@iravasen](https://github.com/iravasen)
- [#8160](https://github.com/AliceO2Group/AliceO2/pull/8160) 2022-02-15: Prototype of the reco TF skimming workflow by [@shahor02](https://github.com/shahor02)
- [#8159](https://github.com/AliceO2Group/AliceO2/pull/8159) 2022-02-15: Spec and workflow to inject DISTSUBTIMEFRAME message by [@shahor02](https://github.com/shahor02)
- [#8166](https://github.com/AliceO2Group/AliceO2/pull/8166) 2022-02-16: Using DataRefUtils to get payload size by [@matthiasrichter](https://github.com/matthiasrichter)
- [#8168](https://github.com/AliceO2Group/AliceO2/pull/8168) 2022-02-17: Change the CruId into the Geo.h by [@fapfap69](https://github.com/fapfap69)
- [#8161](https://github.com/AliceO2Group/AliceO2/pull/8161) 2022-02-17: Fix HMP mc->raw output for for-cru option by [@shahor02](https://github.com/shahor02)
- [#8158](https://github.com/AliceO2Group/AliceO2/pull/8158) 2022-02-17: Fixes for TRD GPU tracking + related and unrelated cleanup by [@davidrohr](https://github.com/davidrohr)
- [#8172](https://github.com/AliceO2Group/AliceO2/pull/8172) 2022-02-17: Minor fixes in HMPID equipment IDs definition by [@shahor02](https://github.com/shahor02)
- [#8179](https://github.com/AliceO2Group/AliceO2/pull/8179) 2022-02-18: Event display: Add option to not throw when no input + unrelated cleanup by [@davidrohr](https://github.com/davidrohr)
- [#8167](https://github.com/AliceO2Group/AliceO2/pull/8167) 2022-02-18: Hmpid dcsccdb improvments by [@fapfap69](https://github.com/fapfap69)
- [#8178](https://github.com/AliceO2Group/AliceO2/pull/8178) 2022-02-19: Fix OB Half Staves relative position and U-leg length by [@mario6829](https://github.com/mario6829)
- [#8185](https://github.com/AliceO2Group/AliceO2/pull/8185) 2022-02-19: Fixed mask for runType once more + bug fixes for ITHR scan by [@iravasen](https://github.com/iravasen)
- [#8186](https://github.com/AliceO2Group/AliceO2/pull/8186) 2022-02-19: Parallelise ITS noise calibration, use digits by default by [@shahor02](https://github.com/shahor02)
- [#8188](https://github.com/AliceO2Group/AliceO2/pull/8188) 2022-02-20: Extend cell time range by [@peressounko](https://github.com/peressounko)
- [#8189](https://github.com/AliceO2Group/AliceO2/pull/8189) 2022-02-20: reset all reco->aod mapping indices after each TF by [@shahor02](https://github.com/shahor02)
- [#8190](https://github.com/AliceO2Group/AliceO2/pull/8190) 2022-02-21: Add parameter of time pre-samples by [@peressounko](https://github.com/peressounko)
- [#8202](https://github.com/AliceO2Group/AliceO2/pull/8202) 2022-02-22: Updated Doxygen  keywords by [@ihrivnac](https://github.com/ihrivnac)
- [#8197](https://github.com/AliceO2Group/AliceO2/pull/8197) 2022-02-22: Uset vector instead of TClonesArray to pass AlignParams by [@shahor02](https://github.com/shahor02)
- [#8232](https://github.com/AliceO2Group/AliceO2/pull/8232) 2022-02-25: Fix in elimination of pixels multiply fired in the same ROF by [@shahor02](https://github.com/shahor02)
- [#8204](https://github.com/AliceO2Group/AliceO2/pull/8204) 2022-02-25: Produce Alpide raw data dumps only on request or in EPNSYNCMODE=1 mode by [@shahor02](https://github.com/shahor02)
- [#8217](https://github.com/AliceO2Group/AliceO2/pull/8217) 2022-02-25: Sort AOD track daughters indices by [@shahor02](https://github.com/shahor02)
- [#8215](https://github.com/AliceO2Group/AliceO2/pull/8215) 2022-02-26: [MRRTF-146] MCH: Introduce CSV version of the Bad Channel List by [@aphecetche](https://github.com/aphecetche)
- [#8218](https://github.com/AliceO2Group/AliceO2/pull/8218) 2022-02-28: MCH: add ability to write digits in Root format in the digits-writer by [@aphecetche](https://github.com/aphecetche)
## Changes in EventVisualisation

- [#8179](https://github.com/AliceO2Group/AliceO2/pull/8179) 2022-02-18: Event display: Add option to not throw when no input + unrelated cleanup by [@davidrohr](https://github.com/davidrohr)
## Changes in Framework

- [#8155](https://github.com/AliceO2Group/AliceO2/pull/8155) 2022-02-17: Bugfix DPL raw proxy: stable walking through DataHeaders by [@matthiasrichter](https://github.com/matthiasrichter)
- [#8164](https://github.com/AliceO2Group/AliceO2/pull/8164) 2022-02-17: Bugfix: using message size as payload size by [@matthiasrichter](https://github.com/matthiasrichter)
- [#8182](https://github.com/AliceO2Group/AliceO2/pull/8182) 2022-02-18: DPL Analysis: allow index builder to be used with filtered input by [@aalkin](https://github.com/aalkin)
- [#8175](https://github.com/AliceO2Group/AliceO2/pull/8175) 2022-02-18: DPL: Get free SHM memory from correct segment, if --shmid is in command line by [@davidrohr](https://github.com/davidrohr)
- [#8180](https://github.com/AliceO2Group/AliceO2/pull/8180) 2022-02-18: DPL: prefetch the channel by name by [@ktf](https://github.com/ktf)
- [#8171](https://github.com/AliceO2Group/AliceO2/pull/8171) 2022-02-18: DPL: reset channel state on PreRun by [@ktf](https://github.com/ktf)
- [#8187](https://github.com/AliceO2Group/AliceO2/pull/8187) 2022-02-19: DPL Analysis: allow index builder to use extended tables as input by [@aalkin](https://github.com/aalkin)
- [#8169](https://github.com/AliceO2Group/AliceO2/pull/8169) 2022-02-20: DPL: fix start-stop-start transition by [@ktf](https://github.com/ktf)
- [#8191](https://github.com/AliceO2Group/AliceO2/pull/8191) 2022-02-21: CCDB fetcher remapping to different hosts or local files by [@shahor02](https://github.com/shahor02)
- [#8177](https://github.com/AliceO2Group/AliceO2/pull/8177) 2022-02-21: DPL Analysis: do not call operator* twice for slice index by [@aalkin](https://github.com/aalkin)
- [#8192](https://github.com/AliceO2Group/AliceO2/pull/8192) 2022-02-21: DPL GUI: allow tracing different code paths independently by [@ktf](https://github.com/ktf)
- [#8193](https://github.com/AliceO2Group/AliceO2/pull/8193) 2022-02-21: DPL: add support for run number dependent ccdb objects by [@ktf](https://github.com/ktf)
- [#8207](https://github.com/AliceO2Group/AliceO2/pull/8207) 2022-02-22: DPL: provide missing timeout by [@ktf](https://github.com/ktf)
- [#8198](https://github.com/AliceO2Group/AliceO2/pull/8198) 2022-02-22: DPL: stop timer after EoS has been reached by [@ktf](https://github.com/ktf)
- [#8206](https://github.com/AliceO2Group/AliceO2/pull/8206) 2022-02-22: Revert "DPL: stop timer after EoS has been reached" by [@ktf](https://github.com/ktf)
- [#8202](https://github.com/AliceO2Group/AliceO2/pull/8202) 2022-02-22: Updated Doxygen  keywords by [@ihrivnac](https://github.com/ihrivnac)
- [#8214](https://github.com/AliceO2Group/AliceO2/pull/8214) 2022-02-24: DPL: add method to rescan DataRelayer / TimesliceIndex by [@ktf](https://github.com/ktf)
- [#8199](https://github.com/AliceO2Group/AliceO2/pull/8199) 2022-02-25: DPL: additional tracing / debug statements by [@ktf](https://github.com/ktf)
- [#8225](https://github.com/AliceO2Group/AliceO2/pull/8225) 2022-02-25: DPL: fix warnings by [@ktf](https://github.com/ktf)
- [#8226](https://github.com/AliceO2Group/AliceO2/pull/8226) 2022-02-25: DPL: keep ServiceSpec around for debug purposes by [@ktf](https://github.com/ktf)
- [#8234](https://github.com/AliceO2Group/AliceO2/pull/8234) 2022-02-27: DPL: provide a way to specify lifetime from text query by [@ktf](https://github.com/ktf)
- [#8208](https://github.com/AliceO2Group/AliceO2/pull/8208) 2022-02-28: Improved testing of the DPL proxies by [@matthiasrichter](https://github.com/matthiasrichter)
## Changes in Utilities

- [#8215](https://github.com/AliceO2Group/AliceO2/pull/8215) 2022-02-26: [MRRTF-146] MCH: Introduce CSV version of the Bad Channel List by [@aphecetche](https://github.com/aphecetche)

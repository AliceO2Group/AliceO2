# Changes since 2021-01-01

## Changes in Analysis

- [#5125](https://github.com/AliceO2Group/AliceO2/pull/5125) 2021-01-06: Fix nCand=0 case by [@aimeric-landou](https://github.com/aimeric-landou)
- [#5127](https://github.com/AliceO2Group/AliceO2/pull/5127) 2021-01-07: Subscribe PID tasks to collisions by [@njacazio](https://github.com/njacazio)
- [#5132](https://github.com/AliceO2Group/AliceO2/pull/5132) 2021-01-11: Introduced 2d arrays in Variant by [@aalkin](https://github.com/aalkin)
- [#5138](https://github.com/AliceO2Group/AliceO2/pull/5138) 2021-01-12: PWGDQ utility classes moved from AnalysisCore to Analysis/Tasks/PWGDQ by [@iarsene](https://github.com/iarsene)
- [#5160](https://github.com/AliceO2Group/AliceO2/pull/5160) 2021-01-12: add alien SE by [@jgrosseo](https://github.com/jgrosseo)
- [#5155](https://github.com/AliceO2Group/AliceO2/pull/5155) 2021-01-13: Add multiplicity distribution task by [@jgrosseo](https://github.com/jgrosseo)
- [#5177](https://github.com/AliceO2Group/AliceO2/pull/5177) 2021-01-13: AliEn metrics by [@jgrosseo](https://github.com/jgrosseo)
- [#5158](https://github.com/AliceO2Group/AliceO2/pull/5158) 2021-01-13: use filters by [@jgrosseo](https://github.com/jgrosseo)
- [#5174](https://github.com/AliceO2Group/AliceO2/pull/5174) 2021-01-14: Add histrogram register and track selection by [@lbariogl](https://github.com/lbariogl)
- [#5183](https://github.com/AliceO2Group/AliceO2/pull/5183) 2021-01-14: Adding opening and run time metrics by [@jgrosseo](https://github.com/jgrosseo)
## Changes in Common

- [#5116](https://github.com/AliceO2Group/AliceO2/pull/5116) 2021-01-05: Global (currently TPC only) refit on GPU using either GPU or TrackParCov track model by [@davidrohr](https://github.com/davidrohr)
- [#5134](https://github.com/AliceO2Group/AliceO2/pull/5134) 2021-01-08: RootSerializableKeyValueStore: Add print function by [@sawenzel](https://github.com/sawenzel)
- [#5133](https://github.com/AliceO2Group/AliceO2/pull/5133) 2021-01-11: Populate MC event header with information from current Pythia8 event by [@preghenella](https://github.com/preghenella)
- [#5178](https://github.com/AliceO2Group/AliceO2/pull/5178) 2021-01-13: Fixes in entropy compression memory management by [@shahor02](https://github.com/shahor02)
- [#5184](https://github.com/AliceO2Group/AliceO2/pull/5184) 2021-01-14: Update CommonUtilsLinkDef.h by [@sawenzel](https://github.com/sawenzel)
## Changes in DataFormats

- [#5104](https://github.com/AliceO2Group/AliceO2/pull/5104) 2021-01-04: Use DBSCAN for Time-Z clustering in PVertexing, debris reduction by [@shahor02](https://github.com/shahor02)
- [#5122](https://github.com/AliceO2Group/AliceO2/pull/5122) 2021-01-05: Fix codechecker violations by [@davidrohr](https://github.com/davidrohr)
- [#5135](https://github.com/AliceO2Group/AliceO2/pull/5135) 2021-01-08: Add trigger inputs branch  by [@AllaMaevskaya](https://github.com/AllaMaevskaya)
- [#5139](https://github.com/AliceO2Group/AliceO2/pull/5139) 2021-01-08: [EMCAL-677] Propagate trigger bits from RDH to TriggerRecord by [@mfasDa](https://github.com/mfasDa)
- [#5142](https://github.com/AliceO2Group/AliceO2/pull/5142) 2021-01-09: Several unrelated fixes in GPU code by [@davidrohr](https://github.com/davidrohr)
- [#5141](https://github.com/AliceO2Group/AliceO2/pull/5141) 2021-01-09: Use GPUTPCO2InterfaceRefit for TPC-ITS matches refit + misc fixes. by [@shahor02](https://github.com/shahor02)
- [#5144](https://github.com/AliceO2Group/AliceO2/pull/5144) 2021-01-10: Fix: increment EMCAL TriggerRecord class version by [@shahor02](https://github.com/shahor02)
- [#5129](https://github.com/AliceO2Group/AliceO2/pull/5129) 2021-01-11: CPV raw writing and reconstruction by [@peressounko](https://github.com/peressounko)
- [#5133](https://github.com/AliceO2Group/AliceO2/pull/5133) 2021-01-11: Populate MC event header with information from current Pythia8 event by [@preghenella](https://github.com/preghenella)
- [#5163](https://github.com/AliceO2Group/AliceO2/pull/5163) 2021-01-12: Bring back info treatment in MCEventHeader by [@sawenzel](https://github.com/sawenzel)
- [#5147](https://github.com/AliceO2Group/AliceO2/pull/5147) 2021-01-12: [FV0][O2-1849] Trigger inputs for CTP simulation by [@mslupeck](https://github.com/mslupeck)
- [#5178](https://github.com/AliceO2Group/AliceO2/pull/5178) 2021-01-13: Fixes in entropy compression memory management by [@shahor02](https://github.com/shahor02)
## Changes in Detectors

- [#5118](https://github.com/AliceO2Group/AliceO2/pull/5118) 2021-01-04: Default mat.corr. is with LUT, fall-back to TGeo if LUT is not set by [@shahor02](https://github.com/shahor02)
- [#5105](https://github.com/AliceO2Group/AliceO2/pull/5105) 2021-01-04: Disable CA_DEBUG in ITS Tracking by [@davidrohr](https://github.com/davidrohr)
- [#5104](https://github.com/AliceO2Group/AliceO2/pull/5104) 2021-01-04: Use DBSCAN for Time-Z clustering in PVertexing, debris reduction by [@shahor02](https://github.com/shahor02)
- [#5122](https://github.com/AliceO2Group/AliceO2/pull/5122) 2021-01-05: Fix codechecker violations by [@davidrohr](https://github.com/davidrohr)
- [#5116](https://github.com/AliceO2Group/AliceO2/pull/5116) 2021-01-05: Global (currently TPC only) refit on GPU using either GPU or TrackParCov track model by [@davidrohr](https://github.com/davidrohr)
- [#5120](https://github.com/AliceO2Group/AliceO2/pull/5120) 2021-01-06: Correctly use object passed to MID decoder constructor by [@dstocco](https://github.com/dstocco)
- [#5121](https://github.com/AliceO2Group/AliceO2/pull/5121) 2021-01-06: Match all subspecs unless one subspec is passed explicitely by [@dstocco](https://github.com/dstocco)
- [#5111](https://github.com/AliceO2Group/AliceO2/pull/5111) 2021-01-06: Rdev tof updates by [@preghenella](https://github.com/preghenella)
- [#5130](https://github.com/AliceO2Group/AliceO2/pull/5130) 2021-01-07: [ITS] Various fix for codechecker and Clang by [@mconcas](https://github.com/mconcas)
- [#5135](https://github.com/AliceO2Group/AliceO2/pull/5135) 2021-01-08: Add trigger inputs branch  by [@AllaMaevskaya](https://github.com/AllaMaevskaya)
- [#5140](https://github.com/AliceO2Group/AliceO2/pull/5140) 2021-01-08: Fix: do not invoke FIT recpoints reader with --use-fit in raw data input mode by [@shahor02](https://github.com/shahor02)
- [#5139](https://github.com/AliceO2Group/AliceO2/pull/5139) 2021-01-08: [EMCAL-677] Propagate trigger bits from RDH to TriggerRecord by [@mfasDa](https://github.com/mfasDa)
- [#5141](https://github.com/AliceO2Group/AliceO2/pull/5141) 2021-01-09: Use GPUTPCO2InterfaceRefit for TPC-ITS matches refit + misc fixes. by [@shahor02](https://github.com/shahor02)
- [#5144](https://github.com/AliceO2Group/AliceO2/pull/5144) 2021-01-10: Fix: increment EMCAL TriggerRecord class version by [@shahor02](https://github.com/shahor02)
- [#5128](https://github.com/AliceO2Group/AliceO2/pull/5128) 2021-01-10: bugfix in SpaceCharge distortion class by [@matthias-kleiner](https://github.com/matthias-kleiner)
- [#5129](https://github.com/AliceO2Group/AliceO2/pull/5129) 2021-01-11: CPV raw writing and reconstruction by [@peressounko](https://github.com/peressounko)
- [#5143](https://github.com/AliceO2Group/AliceO2/pull/5143) 2021-01-11: Fixes for OMP and for dumping events for the standalone benchmark by [@davidrohr](https://github.com/davidrohr)
- [#5153](https://github.com/AliceO2Group/AliceO2/pull/5153) 2021-01-12: Avoid 2D params in SVertexer configurable params by [@shahor02](https://github.com/shahor02)
- [#5147](https://github.com/AliceO2Group/AliceO2/pull/5147) 2021-01-12: [FV0][O2-1849] Trigger inputs for CTP simulation by [@mslupeck](https://github.com/mslupeck)
- [#5164](https://github.com/AliceO2Group/AliceO2/pull/5164) 2021-01-13: Allow multiple test workflows with non-overlapping TF-ids by [@shahor02](https://github.com/shahor02)
- [#5178](https://github.com/AliceO2Group/AliceO2/pull/5178) 2021-01-13: Fixes in entropy compression memory management by [@shahor02](https://github.com/shahor02)
- [#5175](https://github.com/AliceO2Group/AliceO2/pull/5175) 2021-01-13: GPU: remove leftover debug messages by [@davidrohr](https://github.com/davidrohr)
- [#5179](https://github.com/AliceO2Group/AliceO2/pull/5179) 2021-01-13: Work towards getting the TPC Tracking QA run standalone from a tracks ROOT file by [@davidrohr](https://github.com/davidrohr)
- [#5185](https://github.com/AliceO2Group/AliceO2/pull/5185) 2021-01-14: Add TPC QC histograms for (limited) monitoring of cluster rejection on the fly while processing without MC information by [@davidrohr](https://github.com/davidrohr)
- [#5180](https://github.com/AliceO2Group/AliceO2/pull/5180) 2021-01-14: Do not encode TDC errors to compressed output stream by [@preghenella](https://github.com/preghenella)
- [#5181](https://github.com/AliceO2Group/AliceO2/pull/5181) 2021-01-14: Standalone TPC Tracking QA (independent from o2-tpc-reco-workflow) by [@davidrohr](https://github.com/davidrohr)
## Changes in Examples

- [#5133](https://github.com/AliceO2Group/AliceO2/pull/5133) 2021-01-11: Populate MC event header with information from current Pythia8 event by [@preghenella](https://github.com/preghenella)
## Changes in Framework

- [#5113](https://github.com/AliceO2Group/AliceO2/pull/5113) 2021-01-04: DPL: drop unneeded include statements by [@ktf](https://github.com/ktf)
- [#5099](https://github.com/AliceO2Group/AliceO2/pull/5099) 2021-01-04: DPL: move GUI to a plugin by [@ktf](https://github.com/ktf)
- [#5124](https://github.com/AliceO2Group/AliceO2/pull/5124) 2021-01-05: DPL: do not compile GUISupport if AliceO2::DebugGUI is not found by [@ktf](https://github.com/ktf)
- [#5126](https://github.com/AliceO2Group/AliceO2/pull/5126) 2021-01-06: DPL: add helper to printout current state by [@ktf](https://github.com/ktf)
- [#5132](https://github.com/AliceO2Group/AliceO2/pull/5132) 2021-01-11: Introduced 2d arrays in Variant by [@aalkin](https://github.com/aalkin)
- [#5160](https://github.com/AliceO2Group/AliceO2/pull/5160) 2021-01-12: add alien SE by [@jgrosseo](https://github.com/jgrosseo)
- [#5177](https://github.com/AliceO2Group/AliceO2/pull/5177) 2021-01-13: AliEn metrics by [@jgrosseo](https://github.com/jgrosseo)
- [#5165](https://github.com/AliceO2Group/AliceO2/pull/5165) 2021-01-13: DPL utils: allow customising output-proxy by [@ktf](https://github.com/ktf)
- [#5168](https://github.com/AliceO2Group/AliceO2/pull/5168) 2021-01-13: DPL: increase max size of string metrics to 256 bytes by [@ktf](https://github.com/ktf)
- [#5183](https://github.com/AliceO2Group/AliceO2/pull/5183) 2021-01-14: Adding opening and run time metrics by [@jgrosseo](https://github.com/jgrosseo)
- [#5159](https://github.com/AliceO2Group/AliceO2/pull/5159) 2021-01-14: fix to allow iteratorAt on filtered tables by [@jgrosseo](https://github.com/jgrosseo)
## Changes in Generators

- [#5114](https://github.com/AliceO2Group/AliceO2/pull/5114) 2021-01-06: Rdev evgen updates by [@preghenella](https://github.com/preghenella)
- [#5133](https://github.com/AliceO2Group/AliceO2/pull/5133) 2021-01-11: Populate MC event header with information from current Pythia8 event by [@preghenella](https://github.com/preghenella)
- [#5163](https://github.com/AliceO2Group/AliceO2/pull/5163) 2021-01-12: Bring back info treatment in MCEventHeader by [@sawenzel](https://github.com/sawenzel)
- [#5152](https://github.com/AliceO2Group/AliceO2/pull/5152) 2021-01-12: Revert "Populate event header with information of the current Pythia8… by [@sawenzel](https://github.com/sawenzel)
## Changes in Steer

- [#5135](https://github.com/AliceO2Group/AliceO2/pull/5135) 2021-01-08: Add trigger inputs branch  by [@AllaMaevskaya](https://github.com/AllaMaevskaya)
- [#5139](https://github.com/AliceO2Group/AliceO2/pull/5139) 2021-01-08: [EMCAL-677] Propagate trigger bits from RDH to TriggerRecord by [@mfasDa](https://github.com/mfasDa)
- [#5129](https://github.com/AliceO2Group/AliceO2/pull/5129) 2021-01-11: CPV raw writing and reconstruction by [@peressounko](https://github.com/peressounko)
- [#5147](https://github.com/AliceO2Group/AliceO2/pull/5147) 2021-01-12: [FV0][O2-1849] Trigger inputs for CTP simulation by [@mslupeck](https://github.com/mslupeck)
## Changes in Utilities

- [#5122](https://github.com/AliceO2Group/AliceO2/pull/5122) 2021-01-05: Fix codechecker violations by [@davidrohr](https://github.com/davidrohr)
- [#5151](https://github.com/AliceO2Group/AliceO2/pull/5151) 2021-01-12: Small fixes by [@sawenzel](https://github.com/sawenzel)
- [#5156](https://github.com/AliceO2Group/AliceO2/pull/5156) 2021-01-12: o2-sim: Better signal propagation; small fix in jobutils by [@sawenzel](https://github.com/sawenzel)
- [#5169](https://github.com/AliceO2Group/AliceO2/pull/5169) 2021-01-13: jobutils: exit workflows on first task error by [@sawenzel](https://github.com/sawenzel)
- [#5172](https://github.com/AliceO2Group/AliceO2/pull/5172) 2021-01-14: Fix o2_add_dpl_workflow on Ubuntu and some other systems by [@davidrohr](https://github.com/davidrohr)

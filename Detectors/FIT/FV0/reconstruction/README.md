# FV0 reconstruction

Please note that this readme shows how the reconstruction is done _at the moment_. The current reconstruction is not necessarily done the way it _should be_. The algorithm is under review and subject to change.

The FV0 reconstruction workflow is

- [O2/Detectors/FIT/FV0/workflow/src/fv0-reco-workflow.cxx](../workflow/src/fv0-reco-workflow.cxx)

and the actual reconstruction happens in

- [src/BaseRecoTask.cxx](src/BaseRecoTask.cxx) ([include/FV0Reconstruction/BaseRecoTask.h](include/BaseRecoTask.h))

## Channel data reconstruction

Currently, there's no channel data filtering applied in the FV0 reconstruction. All digit channel data are propagated to RecPoints. The following conversions are made:

- Channel ID (`uint8_t`) -> Channel ID (`int`)
- CFD time (TDC units, `int16_t`) -> Time in ns (`double`)
    - Furthermore, if available and enabled, a time offset correction is applied. For FV0 it is not available nor enabled at the moment.
- Amplitude (ADC channels, `int16_t`) -> Ampltidue (ADC channels, `double`)
- Channel bits (`uint8_t`) -> Channel bits (`int`)

## RecPoint reconstruction

The reconstructed RecPoints contain:

- BC information (`o2::InteractionRecord`)
- Channel data
- Three different times:
    - "TimeFirst": time of first signal (w/ charge > `FV0DigParam::chargeThrForMeanTime`)
    - "TimeGlobalMean": average of all signals passing some filtering (including CFD in gate)
    - "TimeSelectedMean": average of all signals passing a bit stricter filtering (includling CFD in gate)

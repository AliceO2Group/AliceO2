/// \file Detector.h
/// \brief Defines unique identifier for all AliceO2 detector systems, needed for stack filtring

#ifndef ALICEO2_DATA_DETECTORLIST_H_
#define ALICEO2_DATA_DETECTORLIST_H_

// kSTOPHERE is needed for iteration over the enum. All detectors have to be put before.
enum DetectorId
{
    kAliIts, kAliTpc, kAliMft, kSTOPHERE
};

#endif

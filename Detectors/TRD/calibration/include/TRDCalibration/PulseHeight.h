// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PulseHeight.h
/// \brief Creates PH spectra from TRD digits found on tracks

#ifndef O2_TRD_PULSEHEIGHT_H
#define O2_TRD_PULSEHEIGHT_H

#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"

namespace o2
{

namespace globaltracking
{
class RecoContainer;
}

namespace trd
{

class PulseHeight
{

 public:
  PulseHeight() = default;

  /// Initialize what is needed
  void init();

  /// Initialize the input arrays
  void setInput(const o2::globaltracking::RecoContainer& input, gsl::span<const Digit>* digits);

  /// Reset the output
  void reset();

  /// Main processing function
  void process();

 private:
  // input arrays which should not be modified since they are provided externally
  gsl::span<const TrackTRD> mTracksInITSTPCTRD;                      ///< TRD tracks reconstructed from TPC or ITS-TPC seeds
  gsl::span<const TrackTRD> mTracksInTPCTRD;                         ///< TRD tracks reconstructed from TPC or TPC seeds
  gsl::span<const Tracklet64> mTrackletsRaw;                         ///< array of raw tracklets
  const gsl::span<const Digit>* mDigits = nullptr;                   ///< array of digits
  gsl::span<const TriggerRecord> mTriggerRecords;                    ///< array of trigger records
  gsl::span<const TrackTriggerRecord> mTrackTriggerRecordsTPCTRD;    ///< array of track trigger records
  gsl::span<const TrackTriggerRecord> mTrackTriggerRecordsITSTPCTRD; ///< array of track trigger records

  ClassDefNV(PulseHeight, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_PULSEHEIGHT_H

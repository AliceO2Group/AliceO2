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

/// \file PulseHeight.cxx
/// \brief Creates PH spectra from TRD digits found on tracks

#include "TRDCalibration/PulseHeight.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include <fairlogger/Logger.h>

using namespace o2::trd;
using namespace o2::trd::constants;

void PulseHeight::reset()
{
}

void PulseHeight::init()
{
}

void PulseHeight::setInput(const o2::globaltracking::RecoContainer& input, gsl::span<const Digit>* digits)
{
  mTracksInITSTPCTRD = input.getITSTPCTRDTracks<TrackTRD>();
  mTracksInTPCTRD = input.getTPCTRDTracks<TrackTRD>();
  mTrackletsRaw = input.getTRDTracklets();
  mTriggerRecords = input.getTRDTriggerRecords();
  mTrackTriggerRecordsTPCTRD = input.getTPCTRDTriggers();
  mTrackTriggerRecordsITSTPCTRD = input.getITSTPCTRDTriggers();
  mDigits = digits;
}

void PulseHeight::process()
{
  LOGP(info, "Processing {} tracklets and {} digits from {} triggers and {} ITS-TPC-TRD tracks and {} TPC-TRD tracks",
       mTrackletsRaw.size(), mDigits->size(), mTriggerRecords.size(), mTracksInITSTPCTRD.size(), mTracksInTPCTRD.size());
}

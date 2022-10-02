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

/// @file   CalibratorPadGainTracks.cxx
/// @author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#include "TPCCalibration/CalibratorPadGainTracks.h"

using namespace o2::tpc;

void CalibratorPadGainTracks::initOutput()
{
  // Here we initialize the vector of our output objects
  mIntervals.clear();
  mCalibs.clear();
  mTimeIntervals.clear();
}

void CalibratorPadGainTracks::setTruncationRange(const float low, const float high)
{
  mLowTruncation = low;
  mUpTruncation = high;
}

void CalibratorPadGainTracks::setRelGainRange(const float minRelgain, const float maxRelgain)
{
  mMinRelgain = minRelgain;
  mMaxRelgain = maxRelgain;
}

void CalibratorPadGainTracks::finalizeSlot(Slot& slot)
{
  const TFType startTF = slot.getTFStart();
  const TFType endTF = slot.getTFEnd();
  LOGP(info, "Finalizing slot {} <= TF <= {}", startTF, endTF);

  auto& calibPadGainTracks = *slot.getContainer();
  calibPadGainTracks.setNormalizationType(mNormType);
  calibPadGainTracks.finalize(mMinEntriesMean, mMinRelgain, mMaxRelgain, mLowTruncation, mUpTruncation);
  mIntervals.emplace_back(startTF, endTF);
  mTimeIntervals.emplace_back(slot.getStartTimeMS(), slot.getEndTimeMS());
  auto& extractedGainMap = calibPadGainTracks.getPadGainMap();

  if (mUseLastExtractedMapAsReference) {
    // multiply extracted gain map with last extracted gain map
    if (mGainMapLastIteration) {
      LOGP(info, "Multiplying gain map with the extracted map from the last iteration");
      extractedGainMap *= *mGainMapLastIteration;
    }
  }

  std::unordered_map<std::string, CalPad> cal({{"GainMap", extractedGainMap}, {"SigmaMap", calibPadGainTracks.getSigmaMap()}});
  mCalibs.emplace_back(cal);

  if (mUseLastExtractedMapAsReference) {
    // buffer current extracted gain map
    LOGP(info, "Buffering extracted gain map from the current iteration");
    mGainMapLastIteration = std::make_unique<CalPad>(extractedGainMap);
  }

  if (mWriteDebug) {
    calibPadGainTracks.dumpToFile(fmt::format("calPadGainTracksBase_TF_{}_to_{}.root", startTF, endTF).data());
  }
}

CalibratorPadGainTracks::Slot& CalibratorPadGainTracks::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<CalibPadGainTracksBase>());
  return slot;
}

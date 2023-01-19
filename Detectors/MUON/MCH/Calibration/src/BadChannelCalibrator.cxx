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

#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Logger.h"
#include "MCHCalibration/BadChannelCalibrator.h"
#include "MCHCalibration/BadChannelCalibratorParam.h"
#include "MathUtils/fit.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

namespace o2::mch::calibration
{

void BadChannelCalibrator::initOutput()
{
  mPedestalsVector.clear();
  mBadChannelsVector.clear();
}

bool BadChannelCalibrator::readyToSend(std::string& reason) const
{
  reason = "";

  // let's check our hypothesis about this object (nslots=1) is actually true
  auto nslots = getNSlots();
  if (nslots != 1) {
    LOGP(error, "nslots={} while it is expected to be 1", nslots);
    return false;
  }

  auto& slot = getFirstSlot();
  bool statIsEnough = hasEnoughData(slot);
  if (statIsEnough) {
    reason = "enough statistics";
    const o2::mch::calibration::PedestalData* pedData = slot.getContainer();
  }
  return statIsEnough;
}

void BadChannelCalibrator::finalize()
{
  // let's check our hypothesis about this object (nslots=1) is actually true
  auto nslots = getNSlots();
  if (nslots != 1) {
    LOGP(fatal, "nslots={} while it is expected to be 1", nslots);
  }

  auto& slot = getSlot(0);
  finalizeSlot(slot);
}

bool BadChannelCalibrator::hasEnoughData(const Slot& slot) const
{
  const int minNofEntries = BadChannelCalibratorParam::Instance().minRequiredNofEntriesPerChannel;
  const o2::mch::calibration::PedestalData* pedData = slot.getContainer();
  auto nofChannels = pedData->size();
  const int requiredChannels = static_cast<int>(BadChannelCalibratorParam::Instance().minRequiredCalibratedFraction * nofChannels);

  auto nofCalibrated = std::count_if(pedData->cbegin(), pedData->cend(),
                                     [&](const PedestalChannel& c) { return c.mEntries > minNofEntries; });

  bool hasEnough = nofCalibrated > requiredChannels;

  LOGP(info,
       "nofChannelWithEnoughStat(>{})={} nofChannels={} requiredChannels={} hasEnough={}",
       minNofEntries, nofCalibrated, nofChannels, requiredChannels, hasEnough);

  return hasEnough;
}

void BadChannelCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  auto pedestalThreshold = BadChannelCalibratorParam::Instance().maxPed;
  auto noiseThreshold = BadChannelCalibratorParam::Instance().maxNoise;

  mPedestalsVector.clear();
  mBadChannelsVector.clear();

  o2::mch::calibration::PedestalData* pedestalData = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // keep track of first TimeFrame
  if (slot.getTFStart() < mTFStart) {
    mTFStart = slot.getTFStart();
  }

  for (const auto& ped : *pedestalData) {
    if (ped.mEntries == 0) {
      continue;
    }
    mPedestalsVector.emplace_back(ped);
    bool bad = true;
    if (ped.mPedestal < pedestalThreshold) {
      if (ped.getRms() < noiseThreshold) {
        bad = false;
      }
    }
    if (bad) {
      LOG(info) << ped;
      mBadChannelsVector.emplace_back(ped.dsChannelId);
    }
  }
}

BadChannelCalibrator::Slot&
  BadChannelCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<PedestalData>());
  return slot;
}

} // namespace o2::mch::calibration

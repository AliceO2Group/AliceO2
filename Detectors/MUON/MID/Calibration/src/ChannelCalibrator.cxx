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

/// \file   MID/Calibration/src/ChannelCalibrator.cxx
/// \brief  MID noise and dead channels calibrator
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 February 2022

#include "MIDCalibration/ChannelCalibrator.h"

#include <iostream>
#include <sstream>
#include "DetectorsCalibration/Utils.h"
#include "MIDFiltering/MaskMaker.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/ROBoardConfigHandler.h"

namespace o2
{
namespace mid
{

using Slot = o2::calibration::TimeSlot<NoiseData>;

void NoiseData::fill(const gsl::span<const ColumnData> data)
{
  for (auto& col : data) {
    mChannelScalers.count(col);
  }
}

void NoiseData::merge(const NoiseData* prev)
{
  mChannelScalers.merge(prev->mChannelScalers);
}

void NoiseData::print()
{
  std::cout << mChannelScalers;
}

void ChannelCalibrator::initOutput()
{
  mBadChannels.clear();
  mEventsCounter = 0;
  mMasksString = "";
}

bool ChannelCalibrator::hasEnoughData(const Slot& slot) const
{
  // MID calibration occurs on a dedicated run.
  // We therefore collect the full statistics and compute the masks only at the end of run.
  // So we return true here
  return true;
}

Slot& ChannelCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<NoiseData>());
  return slot;
}

void ChannelCalibrator::finalizeSlot(Slot& slot)
{
  o2::mid::NoiseData* noiseData = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // Keep track of last TimeFrame, since the masks will be valid from now on
  mBadChannels = makeBadChannels(noiseData->getScalers(), mEventsCounter, mThreshold);

  // Get the masks for the electronics
  // First convert the dead channels into masks
  ChannelMasksHandler masksHandler;
  masksHandler.switchOffChannels(mBadChannels);

  // Complete with the expected masks from mapping
  masksHandler.merge(mRefMasks);

  // Convert column data masks to local board masks
  ColumnDataToLocalBoard colToBoard;
  colToBoard.process(masksHandler.getMasks(), true);

  // Update local board configuration with the masks
  ROBoardConfigHandler roBoardCfgHandler;
  roBoardCfgHandler.updateMasks(colToBoard.getData());
  std::stringstream ss;
  roBoardCfgHandler.write(ss);

  mMasksString = ss.str();
}

} // namespace mid
} // namespace o2

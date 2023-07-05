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

/// \file   MID/Calibration/src/ChannelCalibratorFinalizer.cxx
/// \brief  MID noise and dead channels calibrator finalizer
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   31 October 2022

#include "MIDCalibration/ChannelCalibratorFinalizer.h"

#include <sstream>
#include "MIDBase/ColumnDataHandler.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/ROBoardConfigHandler.h"

namespace o2
{
namespace mid
{

void ChannelCalibratorFinalizer::process(const gsl::span<const ColumnData> noise, const gsl::span<const ColumnData> dead)
{
  ColumnDataHandler colHandler;
  colHandler.merge(noise);
  colHandler.merge(dead);

  // Keep track of last TimeFrame, since the masks will be valid from now on
  mBadChannels = colHandler.getMerged();

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

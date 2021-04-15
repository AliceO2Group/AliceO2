// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0Calibration/FT0CalibrationObject.h"
#include "FT0Calibration/FT0ChannelDataTimeSlotContainer.h"

using namespace o2::ft0;

void FT0CalibrationObjectAlgorithm::calibrate(FT0CalibrationObject& calibrationObject,
                                              const FT0ChannelDataTimeSlotContainer& container)
{
  for (unsigned int iCh = 0; iCh < o2::ft0::Nchannels_FT0; ++iCh) {
    int16_t dOffset = container.getAverageTimeForChannel(iCh);
    calibrationObject.mChannelOffsets[iCh] -= dOffset;
  }
}
std::unique_ptr<TGraph> FT0CalibrationObjectConverter::toTGraph(const FT0CalibrationObject& object)
{
  auto graph = std::make_unique<TGraph>(o2::ft0::Nchannels_FT0);
  uint8_t channelID = 0;

  for (const auto& channelOffset : object.mChannelOffsets) {
    graph->SetPoint(channelID, channelID, channelOffset);
    ++channelID;
  }
  return graph;
}

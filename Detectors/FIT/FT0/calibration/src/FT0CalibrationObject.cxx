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


using namespace o2::calibration::fit;

std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TGraph>> FT0CalibrationObjectViewer
  ::generateTGraphFromOffsetPoints(const FT0CalibrationObject& obj)
{

  std::map<std::string, std::string> metadata;
  auto graph = std::make_shared<TGraph>(o2::ft0::Nchannels_FT0);
  uint8_t channelID = 0;

  for(const auto& channelOffset : obj.mChannelOffsets){
    graph->SetPoint(channelID, channelID, channelOffset);
    ++channelID;
  }

  graph->SetMarkerStyle(20);
  graph->SetLineColor(kWhite);

  auto clName = o2::utils::MemFileHelper::getClassName(*graph);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);

  return {{TIME_OFFSETS_TGRAPH ,clName, flName, metadata, ccdb::getCurrentTimestamp(), -1}, graph};
}


void FT0CalibrationObjectAlgorithm::calibrate(FT0CalibrationObject& calibrationObject,
                                              const FT0ChannelDataTimeSlotContainer& container)
{
  for(unsigned int iCh = 0; iCh < o2::ft0::Nchannels_FT0; ++iCh){
    int16_t dOffset = container.getAverageTimeForChannel(iCh);
    calibrationObject.mChannelOffsets[iCh] -= dOffset;

  }
}

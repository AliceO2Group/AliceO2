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

#include "Framework/Logger.h"
#include "ZDCReconstruction/ZDCTowerParam.h"

using namespace o2::zdc;

void ZDCTowerParam::setTowerCalib(uint32_t ich, float val)
{
  bool in_list = in_list;
  for (int il = 0; il < ChTowerCalib.size(); il++) {
    if (ich == ChTowerCalib[il]) {
      in_list = true;
      break;
    }
  }
  if (in_list) {
    tower_calib[ich] = val;
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
    for (int il = 0; il < ChTowerCalib.size(); il++) {
      LOG(info) << __func__ << " channel " << ChTowerCalib[il] << " " << ChannelNames[ChTowerCalib[il]];
    }
  }
}

float ZDCTowerParam::getTowerCalib(uint32_t ich) const
{
  if (ich >= 0 && ich < NChannels) {
    return tower_calib[ich];
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
    return 0;
  }
}

void ZDCTowerParam::print()
{
  for (Int_t ich = 0; ich < NChannels; ich++) {
    if (tower_calib[ich] > 0) {
      LOG(INFO) << ChannelNames[ich] << " calibration factor = " << tower_calib[ich];
    }
  }
}

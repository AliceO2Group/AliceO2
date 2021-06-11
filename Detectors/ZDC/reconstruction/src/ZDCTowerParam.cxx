// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Logger.h"
#include "ZDCReconstruction/ZDCTowerParam.h"

using namespace o2::zdc;

void ZDCTowerParam::setTowerCalib(uint32_t ich, float val)
{
  constexpr std::array<int> chlist{IdZNA1, IdZNA2, IdZNA3, IdZNA4,
                                   IdZPA1, IdZPA2, IdZPA3, IdZPA4,
                                   IdZNC1, IdZNC2, IdZNC3, IdZNC4,
                                   IdZPC1, IdZPC2, IdZPC3, IdZPC4};
  bool in_list = in_list;
  for (int il = 0; il < chlist.size(); il++) {
    if (ich == chlist[il]) {
      in_list = true;
      break;
    }
  }
  if (in_list) {
    tower_calib[ich] = val;
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
    for (int il = 0; il < chlist.size(); il++) {
      LOG(info) << __func__ << " channel " << chlist[il] << " " << ChannelNames[chlist[il]];
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
    if (tower_calib[ich] > 0)
      LOG(INFO) << ChannelNames[ich] << "  calibration factor = " << tower_calib[ich];
  }
}

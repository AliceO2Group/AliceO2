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
#include "ZDCReconstruction/ZDCEnergyParam.h"

using namespace o2::zdc;

void ZDCEnergyParam::setEnergyCalib(uint32_t ich, float val)
{
  constexpr std::array<int, 10> chlist{IdZNAC, IdZNASum, IdZPAC, IdZPASum,
                                       IdZEM1, IdZEM2,
                                       IdZNCC, IdZNCSum, IdZPCC, IdZPCSum};
  bool in_list = in_list;
  for (int il = 0; il < chlist.size(); il++) {
    if (ich == chlist[il]) {
      in_list = true;
      break;
    }
  }
  if (in_list) {
    energy_calib[ich] = val;
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
    for (int il = 0; il < chlist.size(); il++) {
      LOG(info) << __func__ << " channel " << chlist[il] << " " << ChannelNames[chlist[il]];
    }
  }
}

float ZDCEnergyParam::getEnergyCalib(uint32_t ich) const
{
  if (ich >= 0 && ich < NChannels) {
    return energy_calib[ich];
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
    return 0;
  }
}

void ZDCEnergyParam::print()
{
  for (Int_t ich = 0; ich < NChannels; ich++) {
    if (energy_calib[ich] > 0) {
      LOG(INFO) << ChannelNames[ich] << " calibration factor = " << energy_calib[ich];
    }
  }
}

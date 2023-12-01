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
#include "ZDCReconstruction/ZDCEnergyParam.h"

using namespace o2::zdc;

void ZDCEnergyParam::setEnergyCalib(uint32_t ich, float val)
{
  bool in_list = false;
  for (int il = 0; il < ChEnergyCalib.size(); il++) {
    if (ich == ChEnergyCalib[il]) {
      in_list = true;
      break;
    }
  }
  if (in_list) {
    energy_calib[ich] = val;
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    for (int il = 0; il < ChEnergyCalib.size(); il++) {
      LOG(info) << __func__ << " channel " << ChEnergyCalib[il] << " " << ChannelNames[ChEnergyCalib[il]];
    }
  }
}

float ZDCEnergyParam::getEnergyCalib(uint32_t ich) const
{
  if (ich >= 0 && ich < NChannels) {
    return energy_calib[ich];
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    return 0;
  }
}

void ZDCEnergyParam::setOffset(uint32_t ich, float val)
{
  bool in_list = false;
  for (int il = 0; il < ChEnergyCalib.size(); il++) {
    if (ich == ChEnergyCalib[il]) {
      in_list = true;
      break;
    }
  }
  if (in_list) {
    adc_offset[ich] = val;
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    for (int il = 0; il < ChEnergyCalib.size(); il++) {
      LOG(info) << __func__ << " channel " << ChEnergyCalib[il] << " " << ChannelNames[ChEnergyCalib[il]];
    }
  }
}

float ZDCEnergyParam::getOffset(uint32_t ich) const
{
  if (ich >= 0 && ich < NChannels) {
    return adc_offset[ich];
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    return 0;
  }
}

void ZDCEnergyParam::print() const
{
  for (Int_t ich = 0; ich < NChannels; ich++) {
    if (energy_calib[ich] > 0) {
      LOG(info) << ChannelNames[ich] << " calibration factor = " << energy_calib[ich];
    }
    if (adc_offset[ich] > 0) {
      LOG(info) << ChannelNames[ich] << " adc offset = " << adc_offset[ich];
    }
  }
}

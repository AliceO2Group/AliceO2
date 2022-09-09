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

#include <TH1.h>
#include <TFile.h>
#include <TDirectory.h>
#include "Framework/Logger.h"
#include "ZDCReconstruction/NoiseParam.h"

using namespace o2::zdc;

void NoiseParam::setCalib(uint32_t ich, float val)
{
  if (ich >= NChannels) {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    return;
  }
  noise[ich] = val;
}

float NoiseParam::getCalib(uint32_t ich) const
{
  if (ich < NChannels) {
    return noise[ich];
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    return 0;
  }
}

void NoiseParam::print() const
{
  for (Int_t ich = 0; ich < NChannels; ich++) {
    LOG(info) << ChannelNames[ich] << " Noise = " << noise[ich];
  }
}

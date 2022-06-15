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
#include "ZDCReconstruction/BaselineParam.h"

using namespace o2::zdc;

BaselineParam::BaselineParam()
{
  modified.fill(false);
  for (int i = 0; i < NChannels; i++) {
    baseline[i] = -std::numeric_limits<float>::infinity();
  }
}

void BaselineParam::setCalib(uint32_t ich, float val, bool ismodified)
{
  if (ich >= NChannels) {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    return;
  }
  baseline[ich] = val;
  modified[ich] = ismodified;
}

float BaselineParam::getCalib(uint32_t ich) const
{
  if (ich < NChannels) {
    return baseline[ich];
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    return 0;
  }
}

void BaselineParam::print(bool printall) const
{
  for (Int_t ich = 0; ich < NChannels; ich++) {
    if (baseline[ich] < ADCMin || baseline[ich] > ADCMax) {
      LOG(warn) << ChannelNames[ich] << (modified[ich] ? " NEW" : " OLD") << " Baseline = " << baseline[ich] << " OUT OF RANGE";
    } else {
      if (printall) {
        LOG(info) << ChannelNames[ich] << (modified[ich] ? " NEW" : " OLD") << " Baseline = " << baseline[ich];
      }
    }
  }
}

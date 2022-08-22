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
#include "ZDCCalib/BaselineCalibConfig.h"

using namespace o2::zdc;
BaselineCalibConfig::BaselineCalibConfig()
{
  for (int isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = BaselineMin;
    cutHigh[isig] = BaselineMax;
  }
}

void BaselineCalibConfig::print() const
{
  LOG(info) << "BaselineCalibConfig::print()";
  for (int isig = 0; isig < NChannels; isig++) {
    LOG(info) << ChannelNames[isig] << " limits (" << cutLow[isig] << " : " << cutHigh[isig] << ") min_entries = " << min_e[isig];
  }
}

void BaselineCalibConfig::setMinEntries(uint32_t val)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    min_e[isig] = val;
  }
}

void BaselineCalibConfig::setMinEntries(int isig, uint32_t val)
{
  min_e[isig] = val;
}

void BaselineCalibConfig::resetCuts()
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = BaselineMin;
    cutHigh[isig] = BaselineMax;
  }
}

void BaselineCalibConfig::setCutLow(int val)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = val;
  }
}

void BaselineCalibConfig::setCutHigh(int val)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutHigh[isig] = val;
  }
}

void BaselineCalibConfig::setCutLow(int isig, int val)
{
  cutLow[isig] = val;
}

void BaselineCalibConfig::setCutHigh(int isig, int val)
{
  cutHigh[isig] = val;
}

void BaselineCalibConfig::setCuts(int low, int high)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = low;
    cutHigh[isig] = high;
  }
}

void BaselineCalibConfig::setCuts(int isig, int low, int high)
{
  cutHigh[isig] = low;
  cutLow[isig] = high;
}

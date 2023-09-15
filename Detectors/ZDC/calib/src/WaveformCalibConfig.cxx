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
#include "ZDCCalib/WaveformCalibConfig.h"
#include <limits>

using namespace o2::zdc;
WaveformCalibConfig::WaveformCalibConfig()
{
  for (int isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = -std::numeric_limits<float>::infinity();
    cutHigh[isig] = std::numeric_limits<float>::infinity();
  }
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    cutTimeLow[itdc] = -1.25;
    cutTimeHigh[itdc] = 1.25;
  }
}

void WaveformCalibConfig::restrictRange(int ib, int ie)
{
  ibeg = -WaveformCalib_NBB;
  iend = WaveformCalib_NBA;
  if (ib >= ibeg && ib <= 0) {
    ibeg = ib;
  } else {
    LOG(fatal) << "WaveformCalibConfig::restrictRange wrong setting for ibeg = " << ib;
  }
  if (ie <= iend && ie >= 0) {
    iend = ie;
  } else {
    LOG(fatal) << "WaveformCalibConfig::restrictRange wrong setting for iend = " << ie;
  }
  nbun = iend - ibeg + 1;
}

void WaveformCalibConfig::print() const
{
  LOG(info) << "WaveformCalibConfig range [" << ibeg << ":" << iend << "]";
  for (int isig = 0; isig < NChannels; isig++) {
    LOG(info) << ChannelNames[isig] << " limits A = (" << cutLow[isig] << " : " << cutHigh[isig] << ") min_entries = " << min_e[isig];
  }
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    LOG(info) << ChannelNames[TDCSignal[itdc]] << " T = (" << cutTimeLow[itdc] << " : " << cutTimeHigh[itdc] << ")";
  }
}

void WaveformCalibConfig::setMinEntries(double val)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    min_e[isig] = val;
  }
}

void WaveformCalibConfig::setMinEntries(int isig, double val)
{
  min_e[isig] = val;
}

void WaveformCalibConfig::resetCuts()
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = -std::numeric_limits<float>::infinity();
    cutHigh[isig] = std::numeric_limits<float>::infinity();
  }
}

void WaveformCalibConfig::resetCutLow()
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = -std::numeric_limits<float>::infinity();
  }
}

void WaveformCalibConfig::resetCutHigh()
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutHigh[isig] = std::numeric_limits<float>::infinity();
  }
}

void WaveformCalibConfig::resetCutLow(int isig)
{
  cutLow[isig] = -std::numeric_limits<float>::infinity();
}

void WaveformCalibConfig::resetCutHigh(int isig)
{
  cutHigh[isig] = std::numeric_limits<float>::infinity();
}

void WaveformCalibConfig::setCutLow(double val)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = val;
  }
}

void WaveformCalibConfig::setCutHigh(double val)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutHigh[isig] = val;
  }
}

void WaveformCalibConfig::setCutLow(int isig, double val)
{
  cutLow[isig] = val;
}

void WaveformCalibConfig::setCutHigh(int isig, double val)
{
  cutHigh[isig] = val;
}

void WaveformCalibConfig::setCuts(double low, double high)
{
  for (int32_t isig = 0; isig < NChannels; isig++) {
    cutLow[isig] = low;
    cutHigh[isig] = high;
  }
}

void WaveformCalibConfig::setCuts(int isig, double low, double high)
{
  cutLow[isig] = low;
  cutHigh[isig] = high;
}

void WaveformCalibConfig::setTimeCuts(double low, double high)
{
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    cutTimeLow[itdc] = low;
    cutTimeHigh[itdc] = high;
  }
}

void WaveformCalibConfig::setTimeCuts(int itdc, double low, double high)
{
  cutTimeHigh[itdc] = low;
  cutTimeLow[itdc] = high;
}

int WaveformCalibConfig::getFirst() const
{
  return ibeg;
}

int WaveformCalibConfig::getLast() const
{
  return iend;
}

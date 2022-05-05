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

using namespace o2::zdc;
WaveformCalibConfig::WaveformCalibConfig()
{
  for (int i = 0; i < NH; i++) {
    cutLow[i] = -std::numeric_limits<float>::infinity();
    cutHigh[i] = std::numeric_limits<float>::infinity();
  }
}

void WaveformCalibConfig::restrictRange(int ib, int ie)
{
  ibeg = -NBB;
  iend = NBA;
  if (ib >= ibeg && ib < 0) {
    ibeg = ib;
  } else {
    LOG(fatal) << __func__ << " wrong setting for ibeg = " << ib;
  }
  if (ie <= iend && ie > 0) {
    iend = ie;
  } else {
    LOG(fatal) << __func__ << " wrong setting for iend = " << ie;
  }
  nbun = iend - ibeg + 1;
}

void WaveformCalibConfig::print() const
{
  LOG(info) << "WaveformCalibConfig range [" << ibeg << ":" << iend << "]";
  for (Int_t ih = 0; ih < NH; ih++) {
    LOG(info) << ChannelNames[TDCSignal[ih]] << " limits = (" << cutLow[ih] << " : " << cutHigh[ih] << ") min_entries = " << min_e[ih];
  }
}

void WaveformCalibConfig::setMinEntries(double val)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    min_e[ih] = val;
  }
}

void WaveformCalibConfig::setMinEntries(int ih, double val)
{
  min_e[ih] = val;
}

void WaveformCalibConfig::resetCuts()
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = -std::numeric_limits<float>::infinity();
    cutHigh[ih] = std::numeric_limits<float>::infinity();
  }
}

void WaveformCalibConfig::resetCutLow()
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = -std::numeric_limits<float>::infinity();
  }
}

void WaveformCalibConfig::resetCutHigh()
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutHigh[ih] = std::numeric_limits<float>::infinity();
  }
}

void WaveformCalibConfig::resetCutLow(int ih)
{
  cutLow[ih] = -std::numeric_limits<float>::infinity();
}

void WaveformCalibConfig::resetCutHigh(int ih)
{
  cutHigh[ih] = std::numeric_limits<float>::infinity();
}

void WaveformCalibConfig::setCutLow(double val)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = val;
  }
}

void WaveformCalibConfig::setCutHigh(double val)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutHigh[ih] = val;
  }
}

void WaveformCalibConfig::setCutLow(int ih, double val)
{
  cutLow[ih] = val;
}

void WaveformCalibConfig::setCutHigh(int ih, double val)
{
  cutHigh[ih] = val;
}

void WaveformCalibConfig::setCuts(double low, double high)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = low;
    cutHigh[ih] = high;
  }
}

void WaveformCalibConfig::setCuts(int ih, double low, double high)
{
  cutHigh[ih] = low;
  cutLow[ih] = high;
}

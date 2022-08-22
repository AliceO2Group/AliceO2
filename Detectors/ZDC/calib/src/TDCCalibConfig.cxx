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
#include "ZDCCalib/TDCCalibConfig.h"

using namespace o2::zdc;

void TDCCalibConfig::print() const
{
  const char* hn[NTDCChannels] = {"ZNAC", "ZNAS", "ZPAC", "ZPAS", "ZEM1", "ZEM2", "ZNCC", "ZNCS", "ZPCC", "ZPCS"};
  for (Int_t ih = 0; ih < NTDCChannels; ih++) {
    LOG(info) << hn[ih] << " limits = (" << cutLow[ih] << " : " << cutHigh[ih] << ")";
  }
  for (Int_t ih = 0; ih < NTDCChannels; ih++) {
    LOG(info) << hn[ih] << " booking 1D = (" << nb1[ih] << ", " << amin1[ih] << ", " << amax1[ih] << ")";
  }
  for (Int_t ih = 0; ih < NTDCChannels; ih++) {
    LOG(info) << hn[ih] << " booking 2D = (" << nb2[ih] << ", " << amin2[ih] << ", " << amax2[ih] << ")";
  }
}

void TDCCalibConfig::setMinEntries(double val)
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    min_e[ih] = val;
  }
}

void TDCCalibConfig::setMinEntries(int ih, double val)
{
  min_e[ih] = val;
}

void TDCCalibConfig::resetCuts()
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    cutLow[ih] = -std::numeric_limits<float>::infinity();
    cutHigh[ih] = std::numeric_limits<float>::infinity();
  }
}

void TDCCalibConfig::resetCutLow()
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    cutLow[ih] = -std::numeric_limits<float>::infinity();
  }
}

void TDCCalibConfig::resetCutHigh()
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    cutHigh[ih] = std::numeric_limits<float>::infinity();
  }
}

void TDCCalibConfig::resetCutLow(int ih)
{
  cutLow[ih] = -std::numeric_limits<float>::infinity();
}

void TDCCalibConfig::resetCutHigh(int ih)
{
  cutHigh[ih] = std::numeric_limits<float>::infinity();
}

void TDCCalibConfig::setCutLow(double val)
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    cutLow[ih] = val;
  }
}

void TDCCalibConfig::setCutHigh(double val)
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    cutHigh[ih] = val;
  }
}

void TDCCalibConfig::setCutLow(int ih, double val)
{
  cutLow[ih] = val;
}

void TDCCalibConfig::setCutHigh(int ih, double val)
{
  cutHigh[ih] = val;
}

void TDCCalibConfig::setCuts(double low, double high)
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    cutLow[ih] = low;
    cutHigh[ih] = high;
  }
}

void TDCCalibConfig::setCuts(int ih, double low, double high)
{
  cutHigh[ih] = low;
  cutLow[ih] = high;
}

void TDCCalibConfig::setBinning1D(int nb, double amin, double amax)
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    nb1[ih] = nb;
    amin1[ih] = amin;
    amax1[ih] = amax;
  }
}

void TDCCalibConfig::setBinning2D(int nb, double amin, double amax)
{
  for (int32_t ih = 0; ih < NTDCChannels; ih++) {
    nb2[ih] = nb;
    amin2[ih] = amin;
    amax2[ih] = amax;
  }
}

void TDCCalibConfig::setBinning1D(int ih, int nb, double amin, double amax)
{
  nb1[ih] = nb;
  amin1[ih] = amin;
  amax1[ih] = amax;
}

void TDCCalibConfig::setBinning2D(int ih, int nb, double amin, double amax)
{
  nb2[ih] = nb;
  amin2[ih] = amin;
  amax2[ih] = amax;
}

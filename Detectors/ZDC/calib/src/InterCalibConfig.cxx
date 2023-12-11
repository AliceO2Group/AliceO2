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
#include "ZDCCalib/InterCalibConfig.h"

using namespace o2::zdc;

void InterCalibConfig::print() const
{
  const char* hn[NH] = {"ZNA", "ZPA", "ZNC", "ZPC", "ZEM", "ZNI", "ZPI", "ZPAX", "ZPCX"};
  for (Int_t ih = 0; ih < NH; ih++) {
    LOG(info) << hn[ih] << " limits = (" << cutLow[ih] << " : " << cutHigh[ih] << ")";
  }
  for (Int_t ih = 0; ih < NH; ih++) {
    LOG(info) << hn[ih] << " booking 1D = (" << nb1[ih] << ", " << amin1[ih] << ", " << amax1[ih] << ")";
  }
  for (Int_t ih = 0; ih < NH; ih++) {
    LOG(info) << hn[ih] << " booking 2D = (" << nb2[ih] << ", " << amin2[ih] << ", " << amax2[ih] << ")";
  }
  LOG(info) << "xcut_ZPA = " << xcut_ZPA;
  LOG(info) << "xcut_ZPC = " << xcut_ZPC;
  LOG(info) << "tower_cut_ZP = " << tower_cut_ZP;
  if (cross_check) {
    LOG(warn) << "THIS IS A CROSS CHECK CONFIGURATION (vs SUM)";
  }
}

void InterCalibConfig::setMinEntries(double val)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    min_e[ih] = val;
  }
}

void InterCalibConfig::setMinEntries(int ih, double val)
{
  min_e[ih] = val;
}

void InterCalibConfig::resetCuts()
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = -std::numeric_limits<float>::infinity();
    cutHigh[ih] = std::numeric_limits<float>::infinity();
  }
}

void InterCalibConfig::resetCutLow()
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = -std::numeric_limits<float>::infinity();
  }
}

void InterCalibConfig::resetCutHigh()
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutHigh[ih] = std::numeric_limits<float>::infinity();
  }
}

void InterCalibConfig::resetCutLow(int ih)
{
  cutLow[ih] = -std::numeric_limits<float>::infinity();
}

void InterCalibConfig::resetCutHigh(int ih)
{
  cutHigh[ih] = std::numeric_limits<float>::infinity();
}

void InterCalibConfig::setCutLow(double val)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = val;
  }
}

void InterCalibConfig::setCutHigh(double val)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutHigh[ih] = val;
  }
}

void InterCalibConfig::setCutLow(int ih, double val)
{
  cutLow[ih] = val;
}

void InterCalibConfig::setCutHigh(int ih, double val)
{
  cutHigh[ih] = val;
}

void InterCalibConfig::setCuts(double low, double high)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    cutLow[ih] = low;
    cutHigh[ih] = high;
  }
}

void InterCalibConfig::setCuts(int ih, double low, double high)
{
  cutLow[ih] = low;
  cutHigh[ih] = high;
}

void InterCalibConfig::setBinning1D(int nb, double amin, double amax)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    nb1[ih] = nb;
    amin1[ih] = amin;
    amax1[ih] = amax;
  }
}

void InterCalibConfig::setBinning2D(int nb, double amin, double amax)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    nb2[ih] = nb;
    amin2[ih] = amin;
    amax2[ih] = amax;
  }
}

void InterCalibConfig::setBinning1D(int ih, int nb, double amin, double amax)
{
  nb1[ih] = nb;
  amin1[ih] = amin;
  amax1[ih] = amax;
}

void InterCalibConfig::setBinning2D(int ih, int nb, double amin, double amax)
{
  nb2[ih] = nb;
  amin2[ih] = amin;
  amax2[ih] = amax;
}

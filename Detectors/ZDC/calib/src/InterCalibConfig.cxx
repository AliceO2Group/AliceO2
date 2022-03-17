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

void InterCalibConfig::print()
{
  const char* hn[NH] = {"ZNA", "ZPA", "ZNC", "ZPC", "ZEM"};
  for (Int_t ih = 0; ih < NH; ih++) {
    LOG(info) << hn[ih] << " limits = (" << cutLow[ih] << " : " << cutHigh[ih] << ")";
  }
  for (Int_t ih = 0; ih < NH; ih++) {
    LOG(info) << hn[ih] << " booking 1D = (" << nb1[ih] << ", " << amin1[ih] << ", " << amax1[ih] << ")";
  }
  for (Int_t ih = 0; ih < NH; ih++) {
    LOG(info) << hn[ih] << " booking 2D = (" << nb2[ih] << ", " << amin2[ih] << ", " << amax2[ih] << ")";
  }
}

InterCalibConfig::void setBinning1D(int nb, double amin, double amax)
{
  for (int32_t ih = 0; ih < nh; ih++) {
    nb1[ih] = nb;
    amin1[ih] = amin;
    amax1[ih] = amax;
  }
}
InterCalibConfig::void setBinning2D(int nb, double amin, double amax)
{
  for (int32_t ih = 0; ih < nh; ih++) {
    nb2[ih] = nb;
    amin2[ih] = amin;
    amax2[ih] = amax;
  }
}

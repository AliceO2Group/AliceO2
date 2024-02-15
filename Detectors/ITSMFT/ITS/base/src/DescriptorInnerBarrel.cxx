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

#include <fairlogger/Logger.h> // for LOG, LOG_IF
#include "TGeoTube.h"          // for TGeoTube
#include "ITSBase/DescriptorInnerBarrel.h"

using namespace o2::its;

ClassImp(DescriptorInnerBarrel);

//________________________________________________________________
void DescriptorInnerBarrel::getConfigurationWrapperVolume(double& minradius, double& maxradius, double& zspan) const
{
  minradius = mWrapperMinRadius;
  maxradius = mWrapperMaxRadius;
  zspan = mWrapperZSpan;
}

//________________________________________________________________
void DescriptorInnerBarrel::setConfigurationWrapperVolume(double minradius, double maxradius, double zspan)
{
  mWrapperMinRadius = minradius;
  mWrapperMaxRadius = maxradius;
  mWrapperZSpan = zspan;
}

//________________________________________________________________
TGeoTube* DescriptorInnerBarrel::defineWrapperVolume() const
{
  TGeoTube* wrap = new TGeoTube(mWrapperMinRadius, mWrapperMaxRadius, mWrapperZSpan / 2.);
  LOGP(info, "Creating IB Wrappervolume with Rmin={}, Rmax={}, ZSpan={}", mWrapperMinRadius, mWrapperMaxRadius, mWrapperZSpan);
  return wrap;
}
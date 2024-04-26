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

#include "ITS3Simulation/DescriptorInnerBarrelITS3.h"
#include "fairlogger/Logger.h"

using namespace o2::its3;

ClassImp(DescriptorInnerBarrelITS3);

void DescriptorInnerBarrelITS3::createLayer(int iLayer, TGeoVolume* dest)
{
  LOGP(info, "ITS3-IB: Creating Layer {}", iLayer);
  mIBLayers[iLayer] = std::make_unique<ITS3Layer>(iLayer);
  mIBLayers[iLayer]->createLayer(dest);
}

void DescriptorInnerBarrelITS3::createServices(TGeoVolume* dest)
{
  LOGP(info, "ITS3-IB: Creating Services");
  mServices = std::make_unique<ITS3Services>();
  mServices->createCYSSAssembly(dest);
}

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

/// \file DescriptorInnerBarrelITS3.h
/// \brief Definition of the DescriptorInnerBarrelITS3 class

#ifndef ALICEO2_ITS3_DESCRIPTORINNERBARRELITS3_H
#define ALICEO2_ITS3_DESCRIPTORINNERBARRELITS3_H

#include "ITS3Base/SpecsV2.h"
#include "ITSBase/DescriptorInnerBarrel.h"
#include "ITS3Simulation/ITS3Layer.h"
#include "ITS3Simulation/ITS3Services.h"

#include "TGeoVolume.h"

#include <array>
#include <memory>

namespace o2::its3
{

class DescriptorInnerBarrelITS3 : public o2::its::DescriptorInnerBarrel
{
 public:
  DescriptorInnerBarrelITS3()
  {
    // redefine the wrapper volume
    setConfigurationWrapperVolume(mWrapperMinRadiusITS3, mWrapperMaxRadiusITS3, mWrapperZSpanITS3);
  }

  void createLayer(int idLayer, TGeoVolume* dest);
  void createServices(TGeoVolume* dest);
  void configure() {}

 protected:
  int mNumLayers{constants::nLayers};

  // wrapper volume properties
  double mWrapperMinRadiusITS3{1.8};
  double mWrapperMaxRadiusITS3{4.};
  double mWrapperZSpanITS3{50};

 private:
  std::array<std::unique_ptr<ITS3Layer>, constants::nLayers> mIBLayers;
  std::unique_ptr<ITS3Services> mServices;

  ClassDefNV(DescriptorInnerBarrelITS3, 0); /// ITS3 inner barrel geometry descriptor
};

} // namespace o2::its3

#endif
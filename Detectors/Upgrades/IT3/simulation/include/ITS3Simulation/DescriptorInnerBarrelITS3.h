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

#include <string>
#include <vector>
#include <TObject.h>

#include "ITSBase/DescriptorInnerBarrel.h"
#include "ITS3Simulation/ITS3Layer.h"

namespace o2
{
namespace its3
{

class DescriptorInnerBarrelITS3 : public o2::its::DescriptorInnerBarrel
{
 public:
  enum Version {
    ThreeLayersNoDeadZones,
    ThreeLayers,
    FourLayers,
    FiveLayers
  };

  // standard constructor
  DescriptorInnerBarrelITS3(Version version);
  // default constructor
  DescriptorInnerBarrelITS3() = default;

  DescriptorInnerBarrelITS3(const DescriptorInnerBarrelITS3& src) = delete;
  DescriptorInnerBarrelITS3& operator=(const DescriptorInnerBarrelITS3& geom) = delete;

  void configure();
  ITS3Layer* createLayer(int idLayer, TGeoVolume* dest);

 private:
  Version mVersion{ThreeLayersNoDeadZones}; //! version of ITS3
  std::vector<double> mLayerZLen{};         //! Vector of layer length in Z coordinate
  std::vector<double> mGap{};               //! Vector of gap between empispheres
  std::vector<ITS3Layer*> mLayer{};         //! Vector of layers

  /// \cond CLASSIMP
  ClassDef(DescriptorInnerBarrelITS3, 1); /// ITS inner barrel geometry descriptor
  /// \endcond
};
} // namespace its3
} // namespace o2

#endif
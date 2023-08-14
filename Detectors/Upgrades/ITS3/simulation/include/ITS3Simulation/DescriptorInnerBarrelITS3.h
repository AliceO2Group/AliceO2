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
  // default constructor
  DescriptorInnerBarrelITS3() = default;

  DescriptorInnerBarrelITS3(const DescriptorInnerBarrelITS3& src) = delete;
  DescriptorInnerBarrelITS3& operator=(const DescriptorInnerBarrelITS3& geom) = delete;
  void setVersion(std::string version) { mVersion = version; }

  void configure();
  ITS3Layer* createLayer(int idLayer, TGeoVolume* dest);
  void createServices(TGeoVolume* dest);

 private:
  std::string mVersion{"ThreeLayersNoDeadZones"}; //! version of ITS3
  std::vector<double> mLayerZLen{};               //! Vector of layer length in Z coordinate (in cm)
  std::vector<double> mGapY{};                    //! Vector of gap between empispheres in Y direction (in cm)
  std::vector<double> mGapPhi{};                  //! Vector of gap between empispheres in phi (distance in Y direction in cm)
  std::vector<int> mNumSubSensorsHalfLayer{};     //! Vector of num of subsensors in half layer
  std::vector<double> mFringeChipWidth{};         //! Vector of fringe chip width (in cm)
  std::vector<double> mMiddleChipWidth{};         //! Vector of middle chip width (in cm)
  std::vector<double> mHeightStripFoam{};         //! Vector of strip foam height (in cm)
  std::vector<double> mLengthSemiCircleFoam{};    //! Vector of semi-circle foam length (in cm)
  std::vector<double> mThickGluedFoam{};          //! Vector of glued foam thickness (in cm)
  double mGapXDirection4thLayer{0.};              //! x-direction gap for layer 4  (in cm)

  std::vector<ITS3Layer*> mLayer{}; //! Vector of layers

  double mCyssCylInnerD{0.};       //! CYSS cylinder inner diameter
  double mCyssCylOuterD{0.};       //! CYSS cylinder outer diameter
  double mCyssCylFabricThick{0.};  //! CYSS cylinder fabric thickness
  double mCyssConeIntSectDmin{0.}; //! CYSS cone internal section min diameter
  double mCyssConeIntSectDmax{0.}; //! CYSS cone internal section max diameter
  double mCyssConeFabricThick{0.}; //! CYSS cone fabric thickness
  double mCyssFlangeCDExt{0.};     //! CYSS flange on side C external diameter

  double mAddMaterial3rdLayer{0.}; //! additional material for layer 3 to mimic services (in cm)

  /// \cond CLASSIMP
  ClassDef(DescriptorInnerBarrelITS3, 1); /// ITS3 inner barrel geometry descriptor
  /// \endcond
};
} // namespace its3
} // namespace o2

#endif
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

/// \file DescriptorInnerBarrel.h
/// \brief Definition of the DescriptorInnerBarrel class

#ifndef ALICEO2_ITS_DESCRIPTORINNERBARREL_H
#define ALICEO2_ITS_DESCRIPTORINNERBARREL_H

#include <string>
#include <vector>
#include <TObject.h>
#include <TGeoTube.h>

namespace o2
{
namespace its
{
class DescriptorInnerBarrel : public TObject
{
 public:
  /// Default constructor
  DescriptorInnerBarrel();
  /// Standard constructor
  DescriptorInnerBarrel(int nlayers);

  DescriptorInnerBarrel(const DescriptorInnerBarrel& src) = delete;
  DescriptorInnerBarrel& operator=(const DescriptorInnerBarrel& geom) = delete;

  double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
  {
    // compute turbo angle from radii and sensor width
    return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
  }

  int GetNumberOfLayers() const { return fNumLayers; }
  double GetSensorThickness() const { return fSensorLayerThickness; }
  void GetConfigurationWrapperVolume(double& minradius, double& maxradius, double& zspan);
  TGeoTube* DefineWrapperVolume();

 protected:
  int fNumLayers{3};

  // wrapper volume properties
  double fWrapperMinRadius{2.1};
  double fWrapperMaxRadius{16.4};
  double fWrapperZSpan{70.};

  // sensor properties
  double fSensorLayerThickness{};

  // layer properties
  std::vector<double> fLayerRadii{};
  std::vector<double> fLayerZLen{};
  std::vector<double> fDetectorThickness{};
  std::vector<int> fChipTypeID{};
  std::vector<int> fBuildLevel{};

  /// \cond CLASSIMP
  ClassDef(DescriptorInnerBarrel, 1); /// ITS inner barrel geometry descriptor
  /// \endcond
};
} // namespace its
} // namespace o2

#endif
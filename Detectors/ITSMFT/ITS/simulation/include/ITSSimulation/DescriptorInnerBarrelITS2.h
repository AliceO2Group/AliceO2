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

/// \file DescriptorInnerBarrelITS2.h
/// \brief Definition of the DescriptorInnerBarrelITS2 class

#ifndef ALICEO2_ITS3_DESCRIPTORINNERBARRELITS2_H
#define ALICEO2_ITS3_DESCRIPTORINNERBARRELITS2_H

#include <string>
#include <vector>
#include <TObject.h>
#include <TGeoVolume.h>

#include "ITSBase/DescriptorInnerBarrel.h"

namespace o2
{
namespace its
{
class V3Layer;
class V3Services;

class DescriptorInnerBarrelITS2 : public o2::its::DescriptorInnerBarrel
{
 public:
  // standard constructor
  DescriptorInnerBarrelITS2(int nlayers);
  // default constructor
  DescriptorInnerBarrelITS2();

  DescriptorInnerBarrelITS2(const DescriptorInnerBarrelITS2& src) = delete;
  DescriptorInnerBarrelITS2& operator=(const DescriptorInnerBarrelITS2& geom) = delete;

  void Configure();

  V3Layer* CreateLayer(int idLayer, TGeoVolume* dest);
  void CreateServices(TGeoVolume* dest);

  void AddAlignableVolumesLayer(int idLayer, int wrapperLayerId, TString& parentPath, int& lastUID);

 private:
  void AddAlignableVolumesHalfBarrel(int idLayer, int iHalfBarrel, TString& parentPath, int& lastUID) const;
  void AddAlignableVolumesStave(int idLayer, int iHalfBarrel, int iStave, TString& parentPath, int& lastUID) const;
  void AddAlignableVolumesHalfStave(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, TString& parentPath, int& lastUID) const;
  void AddAlignableVolumesModule(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, int iModule, TString& parentPath, int& lastUID) const;
  void AddAlignableVolumesChip(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, int iModule, int iChip, TString& parentPath, int& lastUID) const;

  // layer properties
  std::vector<bool> fTurboLayer{};      //! True for "turbo" layers
  std::vector<double> fLayerPhi0{};     //! Vector of layer's 1st stave phi in lab
  std::vector<int> fStavePerLayer{};    //! Vector of number of staves per layer
  std::vector<int> fUnitPerStave{};     //! Vector of number of "units" per stave
  std::vector<double> fChipThickness{}; //! Vector of chip thicknesses
  std::vector<double> fStaveWidth{};    //! Vector of stave width (only used for turbo)
  std::vector<double> fStaveTilt{};     //! Vector of stave tilt (only used for turbo)
  std::vector<V3Layer*> fLayer{};       //! Vector of layers

  /// \cond CLASSIMP
  ClassDef(DescriptorInnerBarrelITS2, 1); /// ITS inner barrel geometry descriptor
  /// \endcond
};
} // namespace its
} // namespace o2

#endif
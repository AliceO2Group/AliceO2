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

namespace o2
{
namespace its3 // to be decided
{
class DescriptorInnerBarrelITS3 : public o2::its::DescriptorInnerBarrel // could inherit from DescriptorInnerBarrel in the future
{
 public:
  // default constructor
  DescriptorInnerBarrelITS3() {}
  // standard constructor
  DescriptorInnerBarrelITS3(int nlayers);

  /// Default destructor
  ~DescriptorInnerBarrelITS3() {}

  DescriptorInnerBarrelITS3(const DescriptorInnerBarrelITS3& src) = delete;
  DescriptorInnerBarrelITS3& operator=(const DescriptorInnerBarrelITS3& geom) = delete;

  void ConfigureITS3();
  void GetConfigurationLayers(std::vector<double>& radii,
                              std::vector<double>& zlen,
                              std::vector<double>& thickness,
                              std::vector<int>& chipID,
                              std::vector<int>& buildlev);

  //  private:
  //   // sensor properties

  /// \cond CLASSIMP
  ClassDef(DescriptorInnerBarrelITS3, 1); /// ITS inner barrel geometry descriptor
  /// \endcond
};
} // namespace its3
} // namespace o2

#endif
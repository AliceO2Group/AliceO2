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

#ifndef O2_TRK_SERVICES_H
#define O2_TRK_SERVICES_H

#include <TGeoManager.h>
#include <FairModule.h>

namespace o2
{
namespace trk
{

class TRKServices : public FairModule
{
 public:
  TRKServices() = default;
  TRKServices(float rMin, float zLength, float thickness);
  void createMaterials();
  void createServices(TGeoVolume* motherVolume);

 protected:
  float mRMin;
  float mZLength;
  float mThickness;
  float mX0;
  ClassDefOverride(TRKServices, 1);
};
} // namespace trk
} // namespace o2
#endif // O2_TRK_SERVICES_H
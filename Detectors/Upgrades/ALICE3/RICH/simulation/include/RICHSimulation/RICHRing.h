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

#ifndef ALICEO2_RICH_RING_H
#define ALICEO2_RICH_RING_H

#include <TGeoManager.h>
#include <Rtypes.h>

namespace o2
{
namespace rich
{
class RICHRing
{
 public:
  RICHRing() = default;
  RICHRing(int ringNumber, std::string ringName, float rInn, float rOut, float zLength, float ringX2X0);
  RICHRing(int ringNumber, std::string ringName, float rInn, float zLength, float thick);
  ~RICHRing() = default;

  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getZ() const { return mZ; }
  auto getx2X0() const { return mX2X0; }
  auto getChipThickness() const { return mChipThickness; }
  auto getNumber() const { return mRingNumber; }
  auto getName() const { return mRingName; }

  void createRing(TGeoVolume* motherVolume);

 private:
  int mRingNumber;
  std::string mRingName;
  float mInnerRadius;
  float mOuterRadius;
  float mZ;
  float mX2X0;
  float mChipThickness;

  ClassDef(RICHRing, 0);
};

} // namespace trk
} // namespace o2
#endif // ALICEO2_RICH_RING_H
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

#ifndef ALICE3_PASSIVE_MAGNET_H
#define ALICE3_PASSIVE_MAGNET_H

#include "Alice3DetectorsPassive/PassiveBase.h"

namespace o2
{
namespace passive
{

class Alice3Magnet : public Alice3PassiveBase
{
 public:
  Alice3Magnet(const char* name, const char* Title = "ALICE3 Magnet");
  Alice3Magnet();
  ~Alice3Magnet() override;
  void ConstructGeometry() override;
  void createMaterials();

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

 private:
  Alice3Magnet(const Alice3Magnet& orig);
  Alice3Magnet& operator=(const Alice3Magnet&);

  float mInnerWrapInnerRadius{160.f}; // cm
  float mTotalThickness{10.8f};       // cm
  float mCoilThickness{0.3f};         // cm
  float mZLength{800.f};              // cm

  ClassDefOverride(o2::passive::Alice3Magnet, 1);
};
} // namespace passive
} // namespace o2
#endif
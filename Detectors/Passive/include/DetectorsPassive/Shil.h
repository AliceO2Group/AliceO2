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

#ifndef ALICEO2_PASSIVE_SHIL_H
#define ALICEO2_PASSIVE_SHIL_H

#include "DetectorsPassive/PassiveBase.h"

namespace o2
{
namespace passive
{
class Shil : public PassiveBase
{
 public:
  Shil(const char* name, const char* Title = "ALICE Shil");
  Shil();
  ~Shil() override;
  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

 private:
  Shil(const Shil& orig);
  Shil& operator=(const Shil&);
  void createMaterials();

  ClassDefOverride(o2::passive::Shil, 1);
};
} // namespace passive
} // namespace o2

#endif

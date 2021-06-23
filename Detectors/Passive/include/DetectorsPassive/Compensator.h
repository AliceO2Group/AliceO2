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

#ifndef ALICEO2_PASSIVE_COMPENSATOR_H
#define ALICEO2_PASSIVE_COMPENSATOR_H

#include "DetectorsPassive/PassiveBase.h"
#include "Rtypes.h"

namespace o2
{
namespace passive
{
// The dipole compensator on the A side
class Compensator : public PassiveBase
{
 public:
  Compensator(const char* name, const char* Title = "ALICE Compensator");
  Compensator();
  ~Compensator() override;
  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

 private:
  Compensator(const Compensator& orig);
  Compensator& operator=(const Compensator&);

  void createMaterials();
  void createCompensator();
  TGeoVolume* createMagnetYoke();

  ClassDefOverride(o2::passive::Compensator, 1);
};
} // namespace passive
} // namespace o2

#endif /* ALICEO2_PASSIVE_COMPENSATOR_H */

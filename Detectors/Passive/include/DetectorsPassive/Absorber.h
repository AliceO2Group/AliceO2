// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PASSIVE_ABSORBER_H
#define ALICEO2_PASSIVE_ABSORBER_H

#include "FairModule.h" // for FairModule

namespace o2
{
namespace passive
{
class Absorber : public FairModule
{
 public:
  Absorber(const char* name, const char* Title = "ALICE Absorber");
  Absorber();
  ~Absorber() override;
  void ConstructGeometry() override;
  void createMaterials();

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

 private:
  Absorber(const Absorber& orig);
  Absorber& operator=(const Absorber&);

  ClassDefOverride(o2::passive::Absorber, 1);
};
} // namespace passive
} // namespace o2

#endif

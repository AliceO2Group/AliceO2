// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PASSIVE_DIPOLE_H
#define ALICEO2_PASSIVE_DIPOLE_H

#include "FairModule.h"
#include "Rtypes.h"

namespace o2
{
namespace passive
{
class Dipole : public FairModule
{
 public:
  Dipole(const char* name, const char* Title = "ALICE Dipole");
  Dipole();
  ~Dipole() override;
  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

 private:
  Dipole(const Dipole& orig);
  Dipole& operator=(const Dipole&);

  void createMaterials();
  void createSpectrometerDipole();

  ClassDefOverride(o2::passive::Dipole, 1);
};
} // namespace passive
} // namespace o2

#endif // DIPOLE_H

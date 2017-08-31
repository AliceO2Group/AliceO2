// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PASSIVE_HALL_H
#define ALICEO2_PASSIVE_HALL_H

#include "FairModule.h" // for FairModule

namespace o2
{
namespace passive
{
class Hall : public FairModule
{
 public:
  Hall(const char* name, const char* Title = "ALICE Experimental Hall");
  Hall();
  ~Hall() override;
  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

 private:
  void createMaterials();
  Hall(const Hall& orig);
  Hall& operator=(const Hall&);

  void setNewShield24() { mNewShield24 = true; }
  void setRackShield() { mRackShield = true; }
  bool mNewShield24 = false; // Option for new shielding in PX24 and RB24
  bool mRackShield = false;  // Additional rack shielding

  ClassDefOverride(o2::passive::Hall, 1)
};
}
}

#endif

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

#ifndef ALICE3_PASSIVE_PIPE_H
#define ALICE3_PASSIVE_PIPE_H

#include "Rtypes.h"
#include "Alice3DetectorsPassive/PassiveBase.h"

namespace o2
{
namespace passive
{
class Alice3Pipe : public Alice3PassiveBase
{
 public:
  Alice3Pipe();
  ~Alice3Pipe() override;
  Alice3Pipe(const char* name,
             const char* title = "Alice 3 Pipe",
             const bool isTRKActivated = false,
             const bool isFT3Activated = false,
             const float pipeRIn = 0.f,
             const float pipeThickness = 0.f,
             const float a3ipLength = 0.f,
             const float vacuumVesselRIn = 0.f,
             const float vacuumVesselThickness = 0.f,
             const float vacuumVesselASideLength = 0.f);

  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

  float getPipeRIn() const { return mPipeRIn; }
  float getPipeRMax() const { return mPipeRIn + mPipeThick; }
  float getPipeWidth() const { return mPipeThick; }
  float getA3IPLength() const { return mA3IPLength; }

  float getVacuumVesselRIn() const { return mVacuumVesselRIn; }
  float getVacuumVesselRMax() const { return mVacuumVesselRIn + mVacuumVesselThick; }
  float getVacuumVesselWidth() const { return mVacuumVesselThick; }
  float getVacuumVesselLength() const { return mVacuumVesselASideLength; }

  bool IsTRKActivated() const { return mIsTRKActivated; }
  bool IsFT3Activated() const { return mIsFT3Activated; }

 private:
  void createMaterials();
  Alice3Pipe(const Alice3Pipe& orig) = default;
  Alice3Pipe& operator=(const Alice3Pipe&);

  float mPipeRIn = 0.;    // inner diameter of the pipe
  float mPipeThick = 0.;  // inner beam pipe section thickness
  float mA3IPLength = 0.; // Length of A3IP

  float mVacuumVesselRIn = 0.;    // inner diameter of the vacuum vessel
  float mVacuumVesselThick = 0.;  // outer beam pipe section thickness
  float mVacuumVesselASideLength = 0.; // Length of the A Side of the vacuum vessel around the IP

  bool mIsTRKActivated = true; // If TRK is not active don't create TRK layers allocations in the vacuum volume
  bool mIsFT3Activated = true;

  ClassDefOverride(Alice3Pipe, 1);
};
} // namespace passive
} // namespace o2
#endif

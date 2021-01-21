// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
             const float innerRho = 0.f,
             const float innerThickness = 0.f,
             const float innerLength = 0.f,
             const float outerRho = 0.f,
             const float outerThickness = 0.f,
             const float outerLength = 0.f);

  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

  float getInnerRmin() const { return mBeInnerPipeRmax - mBeInnerPipeThick; }
  float getInnerRmax() const { return mBeInnerPipeRmax; }
  float getInnerWidth() const { return mBeInnerPipeThick; }
  float getInnerDz() const { return mInnerIpHLength; }

  float getOuterRmin() const { return mBeOuterPipeRmax - mBeOuterPipeThick; }
  float getOuterRmax() const { return mBeOuterPipeRmax; }
  float getOuterWidth() const { return mBeOuterPipeThick; }
  float getOuterDz() const { return mOuterIpHLength; }

 private:
  void createMaterials();
  Alice3Pipe(const Alice3Pipe& orig) = default;
  Alice3Pipe& operator=(const Alice3Pipe&);

  float mBeInnerPipeRmax = 0.;  // inner diameter of the Be section
  float mBeInnerPipeThick = 0.; // inner section  thickness
  float mInnerIpHLength = 0.;   // half length of the inner beampipe around the IP

  float mBeOuterPipeRmax = 0.;  // outer diameter of the Be section
  float mBeOuterPipeThick = 0.; // outer section  thickness
  float mOuterIpHLength = 0.;   // half length of the outer beampipe around the IP

  ClassDefOverride(Alice3Pipe, 1);
};
} // namespace passive
} // namespace o2
#endif

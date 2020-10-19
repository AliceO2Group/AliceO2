// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_TRACKLETTRANSFORMER_H
#define O2_TRD_TRACKLETTRANSFORMER_H

#include "TRDBase/Geometry.h"

namespace o2
{
namespace trd
{

class TrackletTransformer
{
 public:
  TrackletTransformer();
  ~TrackletTransformer() = default;

  void loadPadPlane(int hcid);

  float calculateX();

  float calculateY(int hcid, int column, int position);

  float calculateZ(int padrow);

  float calculateDy(int slope);

  float calibrateX(double x, double t0Correction);

  float calibrateDy(double rawDy, double oldLorentzAngle, double lorentzAngle, double driftVRatio);

  std::array<float, 3> transformL2T(int hcid, std::array<double, 3> spacePoint);

 private:
  o2::trd::Geometry* mGeo;
  const o2::trd::PadPlane* mPadPlane;
};

} // namespace trd
} // namespace o2

#endif

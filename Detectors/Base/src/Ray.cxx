// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Ray.cxx
/// \brief Implementation of ray between start-end points for material budget estimate

#include "DetectorsBase/Ray.h"
#include "GPUCommonMath.h"

using namespace o2::base;
using namespace o2::gpu;

//______________________________________________________
GPUd() int Ray::crossLayer(const MatLayerCyl& lr)
{
  // Calculate parameters t of intersection with cyl.layer
  // Calculated as solution of equation for ray crossing with circles of r (rmin and rmax)
  // t^2*mDistXY2 +- sqrt( mXDxPlusYDy^2 - mDistXY2*(mR02 - r^2) )
  // Region of valid t is 0:1.
  // Straigh line may have 2 crossings with cyl. layer
  float detMax = mXDxPlusYDy2 - mDistXY2 * (mR02 - lr.getRMax2());
  if (detMax < 0)
    return 0; // does not reach outer R, hence inner also
  float detMaxRed = CAMath::Sqrt(detMax) * mDistXY2i;
  float tCross0Max = mXDxPlusYDyRed + detMaxRed; // largest possible t

  if (tCross0Max < 0) { // max t is outside of the limiting point -> other t's also
    return 0;
  }

  float tCross0Min = mXDxPlusYDyRed - detMaxRed; // smallest possible t
  if (tCross0Min > 1.f) {                        // min t is outside of the limiting point -> other t's also
    return 0;
  }
  float detMin = mXDxPlusYDy2 - mDistXY2 * (mR02 - lr.getRMin2());
  if (detMin < 0) { // does not reach inner R -> just 1 tangential crossing
    mCrossParams1[0] = tCross0Min > 0.f ? tCross0Min : 0.f;
    mCrossParams2[0] = tCross0Max < 1.f ? tCross0Max : 1.f;
    return validateZRange(mCrossParams1[0], mCrossParams2[0], lr);
  }
  int nCross = 0;
  float detMinRed = CAMath::Sqrt(detMin) * mDistXY2i;
  float tCross1Max = mXDxPlusYDyRed + detMinRed;
  float tCross1Min = mXDxPlusYDyRed - detMinRed;

  if (tCross1Max < 1.f) {
    mCrossParams1[0] = tCross0Max < 1.f ? tCross0Max : 1.f;
    mCrossParams2[0] = tCross1Max > 0.f ? tCross1Max : 0.f;
    if (validateZRange(mCrossParams1[nCross], mCrossParams2[nCross], lr)) {
      nCross++;
    }
  }

  if (tCross1Min > -0.f) {
    mCrossParams1[nCross] = tCross1Min < 1.f ? tCross1Min : 1.f;
    mCrossParams1[nCross] = tCross0Min > 0.f ? tCross0Min : 0.f;
    if (validateZRange(mCrossParams1[nCross], mCrossParams2[nCross], lr)) {
      nCross++;
    }
  }
  return nCross;
}

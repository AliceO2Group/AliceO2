// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CALIBRATEDTRACKLET_H
#define O2_TRD_CALIBRATEDTRACKLET_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"

namespace o2
{
namespace trd
{

// The CalibratedTracklet has been calibrated in x and dy according to a calculated Lorentz Angle and Drift Velocity.
// Tracklet positions in local z direction are reported at the center of the pad-row.
// Pad-tilting correction is performed after tracking.
class CalibratedTracklet
{
 public:
  GPUdDefault() CalibratedTracklet() = default;
  GPUd() CalibratedTracklet(float x, float y, float z, float dy)
    : mX(x), mY(y), mZ(z), mDy(dy){};
  GPUdDefault() ~CalibratedTracklet() = default;

  GPUd() float getX() const { return mX; }
  GPUd() float getY() const { return mY; }
  GPUd() float getZ() const { return mZ; }
  GPUd() float getDy() const { return mDy; }

  GPUd() void setX(float x) { mX = x; }
  GPUd() void setY(float y) { mY = y; }
  GPUd() void setZ(float z) { mZ = z; }
  GPUd() void setDy(float dy) { mDy = dy; }

 private:
  float mX;
  float mY;
  float mZ;
  float mDy;

  ClassDefNV(CalibratedTracklet, 1);
};

} // namespace trd
} // namespace o2

#endif

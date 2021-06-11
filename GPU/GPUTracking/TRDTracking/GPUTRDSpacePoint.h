// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDSpacePoint.h
/// \author Ole Schmidt, ole.schmidt@cern.ch
/// \brief Struct to hold the position/direction information of the tracklets transformed in sector coordinates

#ifndef GPUTRDSPACEPOINT_H
#define GPUTRDSPACEPOINT_H

#ifndef GPUCA_TPC_GEOMETRY_O2 // compatibility to Run 2 data types

namespace GPUCA_NAMESPACE
{
namespace gpu
{

// class to hold the information on the space points
class GPUTRDSpacePoint
{
 public:
  GPUd() GPUTRDSpacePoint(float x = 0, float y = 0, float z = 0, float dy = 0) : mX(x), mY(y), mZ(z), mDy(dy) {}
  GPUd() float getX() const { return mX; }
  GPUd() float getY() const { return mY; }
  GPUd() float getZ() const { return mZ; }
  GPUd() float getDy() const { return mDy; }
  GPUd() void setX(float x) { mX = x; }
  GPUd() void setY(float y) { mY = y; }
  GPUd() void setZ(float z) { mZ = z; }
  GPUd() void setDy(float dy) { mDy = dy; }

 private:
  float mX;  // x position (3.5 mm above anode wires) - radial offset due to t0 mis-calibration, measured -1 mm for run 245353
  float mY;  // y position (sector coordinates)
  float mZ;  // z position (sector coordinates)
  float mDy; // deflection over drift length
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else // compatibility with Run 3 data types

#include "DataFormatsTRD/CalibratedTracklet.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTRDSpacePoint : public o2::trd::CalibratedTracklet
{
};

#ifdef GPUCA_NOCOMPAT
static_assert(sizeof(GPUTRDSpacePoint) == sizeof(o2::trd::CalibratedTracklet), "Incorrect memory layout");
#endif

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUCA_TPC_GEOMETRY_O2

#endif // GPUTRDSPACEPOINT_H

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

#include "DataFormatsTRD/Tracklet64.h"

namespace o2
{
namespace trd
{

// The CalibratedTracklet has been calibrated in x and dy according to a calculated Lorentz Angle and Drift Velocity.
// Tracklet positions in local z direction are reported at the center of the pad-row.
// Pad-tilting correction is performed after tracking.
class CalibratedTracklet : public Tracklet64
{
 public:
  CalibratedTracklet() = default;
  CalibratedTracklet(uint64_t trackletWord, float x, float y, float z, float dy)
    : Tracklet64(trackletWord), mx(x), my(y), mz(z), mdy(dy){};
  ~CalibratedTracklet() = default;

  float getX() const { return mx; }
  float getY() const { return my; }
  float getZ() const { return mz; }
  float getDy() const { return mdy; }

  void setX(float x) { mx = x; }
  void setY(float y) { my = y; }
  void setZ(float z) { mz = z; }
  void setDy(float dy) { mdy = dy; }

 private:
  float mx;
  float my;
  float mz;
  float mdy;
};

} // namespace trd
} // namespace o2

#endif

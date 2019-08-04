// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_HIT_H
#define O2_MCH_SIMULATION_HIT_H

#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace mch
{

class Hit : public ::o2::BasicXYZEHit<float>
{

 public:
  Hit(int trackId = 0, short detElemId = 0, Point3D<float> entrancePoint = {}, const Point3D<float> exitPoint = {},
      float eloss = 0.0, float length = 0.0, float tof = 0.0)
    : ::o2::BasicXYZEHit<float>(entrancePoint.x(), entrancePoint.y(), entrancePoint.z(), tof, eloss, trackId, detElemId), mLength{length}, mExitPoint(exitPoint)
  {
  }

  Point3D<float> entrancePoint() const { return GetPos(); }
  Point3D<float> exitPoint() const { return mExitPoint; }

  short detElemId() const { return GetDetectorID(); }

 private:
  float mLength = {};
  Point3D<float> mExitPoint = {};
  ClassDefNV(Hit, 1);
};

} // namespace mch
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::mch::Hit> : public o2::utils::ShmAllocator<o2::mch::Hit>
{
};
} // namespace std
#endif

#endif

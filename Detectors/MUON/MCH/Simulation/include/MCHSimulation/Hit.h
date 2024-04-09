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
  Hit(int trackId = 0, short detElemId = 0, math_utils::Point3D<float> entrancePoint = {}, const math_utils::Point3D<float> exitPoint = {},
      float eloss = 0.0, float length = 0.0, float entranceTof = 0.0, float exitTof = 0.0)
    : ::o2::BasicXYZEHit<float>(entrancePoint.x(), entrancePoint.y(), entrancePoint.z(), entranceTof, eloss, trackId, detElemId), mLength{length}, mExitPoint(exitPoint), mExitTof{exitTof}
  {
  }

  math_utils::Point3D<float> entrancePoint() const { return GetPos(); }
  math_utils::Point3D<float> exitPoint() const { return mExitPoint; }

  //time in ns 
  float entranceTof() const { return GetTime(); }
  float exitTof() const { return mExitTof; }

  short detElemId() const { return GetDetectorID(); }

 private:
  float mLength = {};
  math_utils::Point3D<float> mExitPoint = {};
  float mExitTof = {};
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

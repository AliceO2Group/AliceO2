// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/Hit.h
/// \brief  Hit for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   10 July 2018

#ifndef O2_MID_SIMULATION_HIT_H
#define O2_MID_SIMULATION_HIT_H

#include <ostream>
#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace mid
{

class Hit : public ::o2::BasicXYZEHit<float>
{

 public:
  Hit(int trackId = 0, short deId = 0, Point3D<float> entrancePoint = {}, Point3D<float> exitPoint = {},
      float eloss = 0.0, float length = 0.0, float tof = 0.0);

  Point3D<float> entrancePoint() const { return GetPos(); }
  Point3D<float> exitPoint() const { return mExitPoint; }
  Point3D<float> middlePoint() const;

  short detElemId() const { return GetDetectorID(); }

 private:
  float mLength = {};
  Point3D<float> mExitPoint = {};
  ClassDefNV(Hit, 1);
};

inline std::ostream& operator<<(std::ostream& stream, const Hit& hit)
{
  stream << "Track = " << hit.GetTrackID() << ", DE = " << hit.GetDetectorID() << ", entrancePoint = " << hit.entrancePoint() << ", exitPoint = " << hit.exitPoint();
  return stream;
}

} // namespace mid
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::mid::Hit> : public o2::utils::ShmAllocator<o2::mid::Hit>
{
};
} // namespace std
#endif

#endif

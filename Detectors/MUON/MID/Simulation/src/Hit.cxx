// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/Hit.cxx
/// \brief  Implementation of hit for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   10 July 2018

#include "MIDSimulation/Hit.h"

ClassImp(o2::mid::Hit);

namespace o2
{
namespace mid
{
Hit::Hit(int trackId, short deId, Point3D<float> entrancePoint, Point3D<float> exitPoint,
         float eloss, float length, float tof) : o2::BasicXYZEHit<float>(entrancePoint.x(), entrancePoint.y(), entrancePoint.z(), tof, eloss, trackId, deId), mLength{length}, mExitPoint(exitPoint)
{
}

Point3D<float> Hit::middlePoint() const
{
  /// Returns the point in between the entrance and exit
  Point3D<float> middle(0.5 * (entrancePoint().x() + exitPoint().x()), 0.5 * (entrancePoint().y() + exitPoint().y()), 0.5 * (entrancePoint().z() + exitPoint().z()));
  return std::move(middle);
}

} // namespace mid
} // namespace o2

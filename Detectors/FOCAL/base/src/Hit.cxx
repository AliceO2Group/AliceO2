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
#include <iostream>
#include "FOCALBase/Hit.h"

using namespace o2::focal;

Hit::Hit(int primary, int trackID, int detID, Subsystem_t subsystem, double initialEnergy, const math_utils::Point3D<float>& pos,
         double tof, double eLoss) : o2::BasicXYZEHit<float>(pos.X(), pos.Y(), pos.Z(), tof, eLoss, trackID, detID),
                                     mSubSystem(subsystem),
                                     mInitialEnergy(initialEnergy)

{
}

bool Hit::operator==(const Hit& other) const
{
  return (GetDetectorID() == other.GetDetectorID()) && (GetTrackID() == other.GetTrackID() && mSubSystem == other.mSubSystem);
}

bool Hit::operator<(const Hit& other) const
{
  if (GetTrackID() != other.GetTrackID()) {
    return GetTrackID() < other.GetTrackID();
  }
  if (mSubSystem != other.mSubSystem) {
    return mSubSystem < other.mSubSystem;
  }
  return GetDetectorID() < other.GetDetectorID();
}

Hit& Hit::operator+=(const Hit& other)
{
  SetEnergyLoss(GetEnergyLoss() + other.GetEnergyLoss());
}

void Hit::printStream(std::ostream& stream) const
{
  stream << "FOCAL point: Track " << GetTrackID() << " in detector segment " << GetDetectorID()
         << " at position (" << GetX() << "|" << GetY() << "|" << GetZ() << "), energy loss " << GetEnergyLoss()
         << ", initial (parent) energy " << mInitialEnergy << " from primary " << mPrimary;
}

Hit o2::focal::operator+(const Hit& lhs, const Hit& rhs)
{
  Hit summed(lhs);
  summed += rhs;
  return summed;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const o2::focal::Hit& point)
{
  point.printStream(stream);
  return stream;
}
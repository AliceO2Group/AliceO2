// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSBase/Hit.h"

using namespace o2::phos;

ClassImp(o2::phos::Hit);

void Hit::PrintStream(std::ostream& stream) const
{
  stream << "PHOS point: Track " << GetTrackID() << " in detector segment " << GetDetectorID() << " at position ("
         << GetX() << "|" << GetY() << "|" << GetZ() << "), energy loss " << GetEnergyLoss() << " total energy "
         << mInitialEnergy << std::endl;
}
// Hit& Hit::operator=(const Hit& rh)
// {
//   mTrackID = rh.mTrackID;
//   mPos = rh.mPos;
//   mTime = rh.mTime;
//   mELoss = rh.mEloss;
//   mDetectorID = rh.mDetectorID;
//   mPvector = rh.mPvector ;
//   mInitialEnergy = rh.mInitialEnergy ;
// }

Bool_t Hit::operator<(const Hit& rhs) const
{
  if (GetDetectorID() == rhs.GetDetectorID())
    return GetTrackID() < rhs.GetTrackID();
  return GetDetectorID() < rhs.GetDetectorID();
}

Bool_t Hit::operator==(const Hit& rhs) const
{
  return ((GetDetectorID() == rhs.GetDetectorID()) && (GetTrackID() == rhs.GetTrackID()));
}

Hit& Hit::operator+=(const Hit& rhs)
{
  if (rhs.GetEnergyLoss() > GetEnergyLoss())
    SetTime(rhs.GetTime());
  SetEnergyLoss(GetEnergyLoss() + rhs.GetEnergyLoss());
  return *this;
}

Hit Hit::operator+(const Hit& rhs) const
{
  Hit result(*this);
  if (rhs.GetEnergyLoss() > result.GetEnergyLoss())
    result.SetTime(rhs.GetTime());
  result.SetEnergyLoss(result.GetEnergyLoss() + rhs.GetEnergyLoss());
  return *this;
}

std::ostream& operator<<(std::ostream& stream, const Hit& p)
{
  p.PrintStream(stream);
  return stream;
}

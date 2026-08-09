// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsTRKFT3/Hit.h"

#include <cstdio>

ClassImp(o2::trkft3::Hit);

namespace o2::trkft3
{

Hit::Hit(int trackID, unsigned short detID, const TVector3& startPos, const TVector3& endPos, const TVector3& startMom,
         double startE, double endTime, double eLoss, unsigned char startStatus, unsigned char endStatus)
  : BasicXYZEHit(endPos.X(), endPos.Y(), endPos.Z(), endTime, eLoss, trackID, detID),
    mMomentum(startMom.Px(), startMom.Py(), startMom.Pz()),
    mPosStart(startPos.X(), startPos.Y(), startPos.Z()),
    mE(startE),
    mTrackStatusEnd(endStatus),
    mTrackStatusStart(startStatus)
{
}

void Hit::Print(const Option_t* opt) const
{
  printf(
    "Det: %5d Track: %6d E.loss: %.3e P: %+.3e %+.3e %+.3e\n"
    "PosIn: %+.3e %+.3e %+.3e PosOut: %+.3e %+.3e %+.3e\n",
    GetDetectorID(), GetTrackID(), GetEnergyLoss(), GetPx(), GetPy(), GetPz(),
    GetStartX(), GetStartY(), GetStartZ(), GetX(), GetY(), GetZ());
}

} // namespace o2::trkft3

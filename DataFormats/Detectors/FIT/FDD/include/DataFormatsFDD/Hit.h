// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Hit.h
/// \Definition of the FDD Hit class we need number of photons on the top of base hit

#ifndef ALICEO2_FDD_HIT_H_
#define ALICEO2_FDD_HIT_H_

#include "SimulationDataFormat/BaseHits.h" // for BasicXYZEHit
#include "CommonUtils/ShmAllocator.h"
#include "TVector3.h"

namespace o2
{
namespace fdd
{

class Hit : public o2::BasicXYZEHit<Float_t, Float_t>
{

 public:
  //Default constructor
  Hit() = default;
  inline Hit(int trackID, unsigned short detID, const TVector3& Pos, double Time, double eLoss, int nPhot);

  Int_t GetNphot() const { return mNphot; }

 private:
  Int_t mNphot; // Number of photons created by current hit

  ClassDefNV(Hit, 1)
};

Hit::Hit(int trackID, unsigned short detID, const TVector3& Pos, double Time, double eLoss, int nPhot)
  : BasicXYZEHit(Pos.X(), Pos.Y(), Pos.Z(), Time, eLoss, trackID, detID),
    mNphot(nPhot)
{
}

} // namespace fdd
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::fdd::Hit> : public o2::utils::ShmAllocator<o2::fdd::Hit>
{
};
} // namespace std
#endif
#endif

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

#ifndef ALICEO2_DATAFORMATSTRKFT3_HIT_H
#define ALICEO2_DATAFORMATSTRKFT3_HIT_H

#include <ostream>
#include <vector>

#include "CommonUtils/ShmAllocator.h"
#include "SimulationDataFormat/BaseHits.h"
#include "Rtypes.h"
#include "TVector3.h"

namespace o2::trkft3
{

class Hit : public o2::BasicXYZEHit<Float_t, Float_t>
{
 public:
  enum HitStatus_t {
    kTrackEntering = 0x1,
    kTrackInside = 0x1 << 1,
    kTrackExiting = 0x1 << 2,
    kTrackOut = 0x1 << 3,
    kTrackStopped = 0x1 << 4,
    kTrackAlive = 0x1 << 5
  };

  Hit() = default;

  Hit(int trackID, unsigned short detID, const TVector3& startPos, const TVector3& endPos, const TVector3& startMom,
      double startE, double endTime, double eLoss, unsigned char startStatus, unsigned char endStatus);

  math_utils::Point3D<Float_t> GetPosStart() const { return mPosStart; }
  Float_t GetStartX() const { return mPosStart.X(); }
  Float_t GetStartY() const { return mPosStart.Y(); }
  Float_t GetStartZ() const { return mPosStart.Z(); }
  template <typename F>
  void GetStartPosition(F& x, F& y, F& z) const
  {
    x = GetStartX();
    y = GetStartY();
    z = GetStartZ();
  }

  math_utils::Vector3D<Float_t> GetMomentum() const { return mMomentum; }
  math_utils::Vector3D<Float_t>& GetMomentum() { return mMomentum; }
  Float_t GetPx() const { return mMomentum.X(); }
  Float_t GetPy() const { return mMomentum.Y(); }
  Float_t GetPz() const { return mMomentum.Z(); }
  Float_t GetE() const { return mE; }
  Float_t GetTotalEnergy() const { return GetE(); }

  UChar_t GetStatusEnd() const { return mTrackStatusEnd; }
  UChar_t GetStatusStart() const { return mTrackStatusStart; }

  Bool_t IsEntering() const { return mTrackStatusEnd & kTrackEntering; }
  Bool_t IsInside() const { return mTrackStatusEnd & kTrackInside; }
  Bool_t IsExiting() const { return mTrackStatusEnd & kTrackExiting; }
  Bool_t IsOut() const { return mTrackStatusEnd & kTrackOut; }
  Bool_t IsStopped() const { return mTrackStatusEnd & kTrackStopped; }
  Bool_t IsAlive() const { return mTrackStatusEnd & kTrackAlive; }

  Bool_t IsEnteringStart() const { return mTrackStatusStart & kTrackEntering; }
  Bool_t IsInsideStart() const { return mTrackStatusStart & kTrackInside; }
  Bool_t IsExitingStart() const { return mTrackStatusStart & kTrackExiting; }
  Bool_t IsOutStart() const { return mTrackStatusStart & kTrackOut; }
  Bool_t IsStoppedStart() const { return mTrackStatusStart & kTrackStopped; }
  Bool_t IsAliveStart() const { return mTrackStatusStart & kTrackAlive; }

  void SetPosStart(const math_utils::Point3D<Float_t>& p) { mPosStart = p; }

  void Print(const Option_t* opt) const;
  friend std::ostream& operator<<(std::ostream& of, const Hit& point)
  {
    of << "-I- Hit: O2 trkft3 point for track " << point.GetTrackID() << " in detector " << point.GetDetectorID() << std::endl;
    return of;
  }

 private:
  math_utils::Vector3D<Float_t> mMomentum; ///< momentum at entrance
  math_utils::Point3D<Float_t> mPosStart;  ///< position at entrance, base position is at exit
  Float_t mE = 0.f;                        ///< total energy at entrance
  UChar_t mTrackStatusEnd = 0;             ///< MC status flag at exit
  UChar_t mTrackStatusStart = 0;           ///< MC status at starting point

  ClassDefNV(Hit, 3);
};

} // namespace o2::trkft3

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::trkft3::Hit> : public o2::utils::ShmAllocator<o2::trkft3::Hit>
{
};
} // namespace std
#endif

#endif

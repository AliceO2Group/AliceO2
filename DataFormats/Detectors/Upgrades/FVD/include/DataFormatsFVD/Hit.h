// Copyright 2019-2024 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FVD_HIT_H_
#define ALICEO2_FVD_HIT_H_

#include "SimulationDataFormat/BaseHits.h" // for BasicXYZEHit
#include "CommonUtils/ShmAllocator.h"
#include "TVector3.h"

namespace o2
{
namespace fvd
{

class Hit : public o2::BasicXYZEHit<Float_t, Float_t>
{

 public:
  Hit() = default;

  inline Hit(int trackID,
             int cellID,
             const math_utils::Point3D<float>& startPos,
             const math_utils::Point3D<float>& endPos,
             const math_utils::Vector3D<float>& startMom,
             double startE,
             double endTime,
             double eLoss,
             int particlePdg);

  // Entrance position getters
  math_utils::Point3D<Float_t> const& GetPosStart() const { return mPositionStart; }
  Float_t GetStartX() const { return mPositionStart.X(); }
  Float_t GetStartY() const { return mPositionStart.Y(); }
  Float_t GetStartZ() const { return mPositionStart.Z(); }
  template <typename F>
  void GetStartPosition(F& x, F& y, F& z) const
  {
    x = GetStartX();
    y = GetStartY();
    z = GetStartZ();
  }

  // Momentum getters
  math_utils::Vector3D<Float_t> const& GetMomentum() const { return mMomentumStart; }

  math_utils::Vector3D<Float_t>& GetMomentum() { return mMomentumStart; }
  Float_t GetPx() const { return mMomentumStart.X(); }
  Float_t GetPy() const { return mMomentumStart.Y(); }
  Float_t GetPz() const { return mMomentumStart.Z(); }
  Float_t GetE() const { return mEnergyStart; }

  Float_t GetTotalEnergyAtEntrance() const { return GetE(); }

  int GetParticlePdg() const {return mParticlePdg;}

  void Print(const Option_t* opt) const;

 private:
  int mParticlePdg;
  float mEnergyStart;
  math_utils::Vector3D<float> mMomentumStart; ///< momentum at entrance
  math_utils::Point3D<float> mPositionStart; 
  ClassDefNV(Hit, 1);

};

Hit::Hit(int trackID,
         int detID,
         const math_utils::Point3D<float>& startPos,
         const math_utils::Point3D<float>& endPos,
         const math_utils::Vector3D<float>& startMom,
         double startE,
         double endTime,
         double eLoss,
         Int_t particlePdg)
  : BasicXYZEHit(endPos.X(),
                 endPos.Y(),
                 endPos.Z(),
                 endTime,
                 eLoss,
                 trackID,
                 detID)
{
}

} // namespace fvd
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::fvd::Hit> : public o2::utils::ShmAllocator<o2::fvd::Hit>
{
};
} // namespace std
#endif
#endif

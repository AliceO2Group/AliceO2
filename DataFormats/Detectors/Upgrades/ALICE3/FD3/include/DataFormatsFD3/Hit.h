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

/// \file Hit.h
/// \brief Definition of the FD3 Hit class (based on ITSMFT and FV0)

#ifndef ALICEO2_FVD_HIT_H_
#define ALICEO2_FVD_HIT_H_

#include <iosfwd>
#include "SimulationDataFormat/BaseHits.h" // for BasicXYZEHit
#include "Rtypes.h"                        // for Bool_t, Double_t, int, Double32_t, etc
#include "TVector3.h"                      // for TVector3
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace fd3
{

class Hit : public o2::BasicXYZEHit<float, float>
{
 public:
  /// Default constructor
  Hit() = default;

  /// Class Constructor
  /// \param trackID Index of MCTrack
  /// \param cellID Cell ID
  /// \param startPos Coordinates at entrance to active volume [cm]
  /// \param endPos Coordinates to active volume [cm]
  /// \param startMom Momentum of track at entrance [GeV]
  /// \param startE Energy of track at entrance [GeV]
  /// \param endTime Final time [ns]
  /// \param eLoss Energy deposit [GeV]
  /// \param particlePdg PDG code of the partcile associated with the track
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
  math_utils::Point3D<float> const& GetPosStart() const { return mPositionStart; }
  float GetStartX() const { return mPositionStart.X(); }
  float GetStartY() const { return mPositionStart.Y(); }
  float GetStartZ() const { return mPositionStart.Z(); }
  template <typename F>
  void GetStartPosition(F& x, F& y, F& z) const
  {
    x = GetStartX();
    y = GetStartY();
    z = GetStartZ();
  }

  // Momentum getters
  math_utils::Vector3D<float> const& GetMomentum() const { return mMomentumStart; }
  math_utils::Vector3D<float>& GetMomentum() { return mMomentumStart; }
  float GetPx() const { return mMomentumStart.X(); }
  float GetPy() const { return mMomentumStart.Y(); }
  float GetPz() const { return mMomentumStart.Z(); }
  float GetE() const { return mEnergyStart; }
  float GetTotalEnergyAtEntrance() const { return GetE(); }
  int GetParticlePdg() const { return mParticlePdg; }

  void Print(const Option_t* opt) const;

 private:
  math_utils::Vector3D<float> mMomentumStart; ///< momentum at entrance
  math_utils::Point3D<float> mPositionStart;  ///< position at entrance (base mPos give position on exit)
  float mEnergyStart;                         ///< total energy at entrance
  int mParticlePdg;                           ///< PDG code of the particle associated with this track

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
         int particlePdg)
  : BasicXYZEHit(endPos.X(),
                 endPos.Y(),
                 endPos.Z(),
                 endTime,
                 eLoss,
                 trackID,
                 detID),
    mMomentumStart(startMom.X(), startMom.Y(), startMom.Z()),
    mPositionStart(startPos.X(), startPos.Y(), startPos.Z()),
    mEnergyStart(startE),
    mParticlePdg(particlePdg)
{
}

} // namespace fd3
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::fd3::Hit> : public o2::utils::ShmAllocator<o2::fd3::Hit>
{
};

} // namespace std
#endif /* USESHM */
#endif /* ALICEO2_FD3_HIT_H_ */

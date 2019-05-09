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
/// \brief Definition of the FIT-V0 Hit class (based on ITSMFT)

#ifndef ALICEO2_FIT_V0_HIT_H_
#define ALICEO2_FIT_V0_HIT_H_

#include "SimulationDataFormat/BaseHits.h" // for BasicXYZEHit
#include "Rtypes.h"                        // for Bool_t, Double_t, Int_t, Double32_t, etc
#include "TVector3.h"                      // for TVector3
#include <iosfwd>
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace v0
{

class Hit : public o2::BasicXYZEHit<Float_t, Float_t>
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
  inline Hit(int trackID, int cellID, const TVector3& startPos, const TVector3& endPos,
             const TVector3& startMom, double startE, double endTime, double eLoss,
             Int_t particlePdg);

  // Entrance position getters
  Point3D<Float_t> GetPosStart() const { return mPositionStart; }
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
  Vector3D<Float_t> GetMomentum() const { return mMomentumStart; }
  Vector3D<Float_t>& GetMomentum() { return mMomentumStart; }
  Float_t GetPx() const { return mMomentumStart.X(); }
  Float_t GetPy() const { return mMomentumStart.Y(); }
  Float_t GetPz() const { return mMomentumStart.Z(); }
  Float_t GetE() const { return mEnergyStart; }
  Float_t GetTotalEnergyAtEntrance() const { return GetE(); }

  void Print(const Option_t* opt) const;

 private:
  Vector3D<Float_t> mMomentumStart; ///< momentum at entrance
  Point3D<Float_t> mPositionStart;  ///< position at entrance (base mPos give position on exit)
  Float_t mEnergyStart;             ///< total energy at entrance
  Int_t mParticlePdg;               ///< PDG code of the particle associated with this track

  ClassDefNV(Hit, 1)
};

Hit::Hit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
         const TVector3& startMom, double startE, double endTime, double eLoss,
         Int_t particlePdg)
  : BasicXYZEHit(endPos.X(), endPos.Y(), endPos.Z(), endTime, eLoss, trackID, detID),
    mMomentumStart(startMom.Px(), startMom.Py(), startMom.Pz()),
    mPositionStart(startPos.X(), startPos.Y(), startPos.Z()),
    mEnergyStart(startE),
    mParticlePdg(particlePdg)
{
}

} // namespace v0
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::v0::Hit> : public o2::utils::ShmAllocator<o2::v0::Hit>
{
};

} // namespace std
#endif /* USESHM */
#endif /* ALICEO2_FIT_V0_HIT_H_ */

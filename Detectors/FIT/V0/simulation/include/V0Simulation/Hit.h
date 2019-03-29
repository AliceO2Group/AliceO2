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
  /// \param pos Coordinates to active volume [cm]
  /// \param mom Momentum of track at entrance [GeV]
  /// \param endTime Time at entrance [ns]
  /// \param time Time since event start [ns]
  /// \param eLoss Energy deposit [GeV]
  /// \param startStatus: status at entrance
  /// \param endStatus: status at exit
  inline Hit(int trackID, int cellID, const TVector3& startPos, const TVector3& endPos,
             const TVector3& startMom, double startE, double endTime, double eLoss,
             float eTot, float eDep); // last two are for testing only

  // Entrance position getters
  Point3D<Float_t> GetPosStart() const { return mPosStart; }
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
  // momentum getters
  Vector3D<Float_t> GetMomentum() const { return mMomentum; }
  Vector3D<Float_t>& GetMomentum() { return mMomentum; }
  Float_t GetPx() const { return mMomentum.X(); }
  Float_t GetPy() const { return mMomentum.Y(); }
  Float_t GetPz() const { return mMomentum.Z(); }
  Float_t GetE() const { return mE; }
  Float_t GetTotalEnergy() const { return GetE(); }
  /*
    UChar_t GetStatusEnd()   const  { return mTrackStatusEnd; }
    UChar_t GetStatusStart() const  { return mTrackStatusStart; }

    Bool_t IsEntering()      const  { return mTrackStatusEnd & kTrackEntering; }
    Bool_t IsInside()        const  { return mTrackStatusEnd & kTrackInside; }
    Bool_t IsExiting()       const  { return mTrackStatusEnd & kTrackExiting; }
    Bool_t IsOut()           const  { return mTrackStatusEnd & kTrackOut; }
    Bool_t IsStopped()       const  { return mTrackStatusEnd & kTrackStopped; }
    Bool_t IsAlive()         const  { return mTrackStatusEnd & kTrackAlive; }

    Bool_t IsEnteringStart() const  { return mTrackStatusStart & kTrackEntering; }
    Bool_t IsInsideStart()   const  { return mTrackStatusStart & kTrackInside; }
    Bool_t IsExitingStart()  const  { return mTrackStatusStart & kTrackExiting; }
    Bool_t IsOutStart()      const  { return mTrackStatusStart & kTrackOut; }
    Bool_t IsStoppedStart()  const  { return mTrackStatusStart & kTrackStopped; }
    Bool_t IsAliveStart()    const  { return mTrackStatusStart & kTrackAlive; }

    /// Output to screen
    friend std::ostream &operator<<(std::ostream &of, const Hit &point)
    {
      of << "-I- Hit: O2its point for track " << point.GetTrackID() << " in detector " << point.GetDetectorID() << std::endl;
      //of << "    Position (" << point.fX << ", " << point.fY << ", " << point.fZ << ") cm" << std::endl;
      //of << "    Momentum (" << point.fPx << ", " << point.fPy << ", " << point.fPz << ") GeV" << std::endl;
      //of << "    Time " << point.fTime << " ns,  Length " << point.fLength << " cm,  Energy loss "
      //<< point.fELoss * 1.0e06 << " keV" << std::endl;
      return of;
    }
*/
  void Print(const Option_t* opt) const;

 private:
  Vector3D<Float_t> mMomentum; ///< momentum at entrance
  Point3D<Float_t> mPosStart;  ///< position at entrance (base mPos give position on exit)
  Float_t mE;                  ///< total energy at entrance

  ClassDefNV(Hit, 1)
};

Hit::Hit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
         const TVector3& startMom, double startE, double endTime, double eLoss,
         float eTot, float eDep)
  : BasicXYZEHit(endPos.X(), endPos.Y(), endPos.Z(), endTime, eLoss, trackID, detID),
    mMomentum(startMom.Px(), startMom.Py(), startMom.Pz()),
    mPosStart(startPos.X(), startPos.Y(), startPos.Z()),
    mE(startE)
// TODO: may require adding initialization to eTot and eDep and corresponding fields in the Hit.h header
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

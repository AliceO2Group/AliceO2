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
/// \brief Generic hit type for externally injected (CAD-derived) sensitive detectors
///
/// o2::ext::Hit is a deliberately rich, detector-agnostic hit. A single external
/// hit type lets an arbitrary number of o2::ext::ExternalDetector instances (each
/// tied to a different DetID) share one wire format, so that the o2-sim hit merger
/// only ever has to know how to (de)serialize this one type, independently of how
/// many external detectors are configured or what their sensitive action does.
///
/// It stores entrance and exit position, the momentum, energy and energy loss, the
/// time, the track length in the volume, the PDG code and the MC status flags at
/// entrance/exit, so that most information an external sensitive action might want
/// to keep is available downstream.

#ifndef ALICEO2_EXT_HIT_H
#define ALICEO2_EXT_HIT_H

#include "SimulationDataFormat/BaseHits.h" // for BasicXYZEHit
#include "CommonUtils/ShmAllocator.h"
#include "Rtypes.h"
#include "TVector3.h"
#include <iosfwd>

namespace o2::ext
{

class Hit : public o2::BasicXYZEHit<float, float>
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

  /// \param trackID    index of the MCTrack
  /// \param sensorID   index of the sensitive volume (per-detector running id)
  /// \param startPos   coordinates at entrance to the active volume [cm]
  /// \param endPos     coordinates at exit of the active volume [cm]
  /// \param startMom   momentum of the track at entrance [GeV]
  /// \param startE     total energy at entrance [GeV]
  /// \param endTime    time at exit [ns]
  /// \param eLoss      energy deposited in the volume [GeV]
  /// \param startStatus MC status flags at entrance
  /// \param endStatus   MC status flags at exit
  /// \param pdg        PDG code of the track (optional)
  /// \param length     track length inside the volume [cm] (optional)
  Hit(int trackID, unsigned short sensorID, const TVector3& startPos, const TVector3& endPos,
      const TVector3& startMom, double startE, double endTime, double eLoss,
      unsigned char startStatus, unsigned char endStatus, int pdg = 0, float length = 0.f)
    : BasicXYZEHit(endPos.X(), endPos.Y(), endPos.Z(), endTime, eLoss, trackID, sensorID),
      mMomentum(startMom.Px(), startMom.Py(), startMom.Pz()),
      mPosStart(startPos.X(), startPos.Y(), startPos.Z()),
      mE(startE),
      mLength(length),
      mPdg(pdg),
      mTrackStatusEnd(endStatus),
      mTrackStatusStart(startStatus)
  {
  }

  // entrance position
  math_utils::Point3D<float> GetPosStart() const { return mPosStart; }
  float GetStartX() const { return mPosStart.X(); }
  float GetStartY() const { return mPosStart.Y(); }
  float GetStartZ() const { return mPosStart.Z(); }
  void SetPosStart(const math_utils::Point3D<float>& p) { mPosStart = p; }

  // momentum / energy
  math_utils::Vector3D<float> GetMomentum() const { return mMomentum; }
  math_utils::Vector3D<float>& GetMomentum() { return mMomentum; }
  float GetPx() const { return mMomentum.X(); }
  float GetPy() const { return mMomentum.Y(); }
  float GetPz() const { return mMomentum.Z(); }
  float GetE() const { return mE; }
  float GetTotalEnergy() const { return mE; }

  // extra bookkeeping
  float GetLength() const { return mLength; }
  void SetLength(float l) { mLength = l; }
  int GetPdg() const { return mPdg; }
  void SetPdg(int pdg) { mPdg = pdg; }

  // status flags
  unsigned char GetStatusStart() const { return mTrackStatusStart; }
  unsigned char GetStatusEnd() const { return mTrackStatusEnd; }
  bool IsEntering() const { return mTrackStatusEnd & kTrackEntering; }
  bool IsInside() const { return mTrackStatusEnd & kTrackInside; }
  bool IsExiting() const { return mTrackStatusEnd & kTrackExiting; }
  bool IsOut() const { return mTrackStatusEnd & kTrackOut; }
  bool IsStopped() const { return mTrackStatusEnd & kTrackStopped; }
  bool IsAlive() const { return mTrackStatusEnd & kTrackAlive; }

  friend std::ostream& operator<<(std::ostream& of, const Hit& point)
  {
    of << "-I- o2::ext::Hit for track " << point.GetTrackID() << " in sensor " << point.GetDetectorID();
    return of;
  }

 private:
  math_utils::Vector3D<float> mMomentum; ///< momentum at entrance
  math_utils::Point3D<float> mPosStart;  ///< position at entrance (base mPos holds the exit position)
  float mE;                              ///< total energy at entrance
  float mLength;                         ///< track length inside the volume
  int mPdg;                              ///< PDG code of the track
  unsigned char mTrackStatusEnd;         ///< MC status flag at exit
  unsigned char mTrackStatusStart;       ///< MC status flag at entrance

  ClassDefNV(Hit, 1);
};

} // namespace o2::ext

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::ext::Hit> : public o2::utils::ShmAllocator<o2::ext::Hit>
{
};
} // namespace std
#endif

#endif // ALICEO2_EXT_HIT_H

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_BASE_TRACKREFERENCE_H_
#define ALICEO2_BASE_TRACKREFERENCE_H_

#include <TVirtualMC.h>
#include <ostream>
#include "Rtypes.h" // for TrackReference::Class, ClassDef, etc
#include "TMath.h"  // for Pi, Sqrt, ATan2, Cos, Sin, ACos

namespace o2
{

// class encoding sim track status
struct SimTrackStatus {
 public:
  enum SimTrackStatus_Enum {
    kTrackEntering = 0x1,
    kTrackInside = 0x1 << 1,
    kTrackExiting = 0x1 << 2,
    kTrackOut = 0x1 << 3,
    kTrackStopped = 0x1 << 4,
    kTrackAlive = 0x1 << 5,
    kTrackNew = 0x1 << 6
  };
  SimTrackStatus() = default;
  SimTrackStatus(const TVirtualMC& vmc)
  {
    // This is quite annoying since every single call
    // is virtual
    if (vmc.IsTrackEntering()) {
      mStatus |= kTrackEntering;
    }
    if (vmc.IsTrackExiting()) {
      mStatus |= kTrackExiting;
    }
    if (vmc.IsTrackInside()) {
      mStatus |= kTrackInside;
    }
    if (vmc.IsTrackOut()) {
      mStatus |= kTrackOut;
    }
    if (vmc.IsTrackStop()) {
      mStatus |= kTrackAlive;
    }
    if (vmc.IsNewTrack()) {
      mStatus |= kTrackNew;
    }
  }
  bool isEntering() const { return mStatus & kTrackEntering; }
  bool isInside() const { return mStatus & kTrackInside; }
  bool isExiting() const { return mStatus & kTrackExiting; }
  bool isOut() const { return mStatus & kTrackOut; }
  bool isStopped() const { return mStatus & kTrackStopped; }
  bool isAlive() const { return mStatus & kTrackAlive; }
  bool isNew() const { return mStatus & kTrackNew; }
  unsigned char getStatusWord() const { return mStatus; }

  friend std::ostream& operator<<(std::ostream&, const SimTrackStatus&);

 private:
  unsigned char mStatus = 0;
  ClassDefNV(SimTrackStatus, 1)
};

/// Track Reference object is created every time particle is
/// crossing detector bounds.
/// It is a snapshot of the track during propagation.

/// The class stores the following informations:
/// track label,
/// track position: X,Y,X
/// track momentum px, py, pz
/// track length and time of fligth: both in cm
/// status bits from Monte Carlo

// NOTE: This track shares a lot of functionality with MC track and other tracks
// which should be factored into a common base class
class TrackReference
{
 public:
  /// Default Constructor
  TrackReference() = default;

  TrackReference(float x, float y, float z, float px, float py, float pz, float length, float tof, int trackID,
                 int detlabel);
  TrackReference(const TVirtualMC& vmc, int detlabel);

  /// Default Destructor
  ~TrackReference() = default;

  Int_t getTrackID() const { return mTrackNumber; }
  void setTrackID(Int_t track) { mTrackNumber = track; }
  void setLength(float length) { mTrackLength = length; }
  void setTime(float time) { mTof = time; }
  float getLength() const { return mTrackLength; }
  float getTime() const { return mTof; }
  float R() const { return TMath::Sqrt(mX * mX + mY * mY); }

  float Pt() const { return TMath::Sqrt(mPX * mPX + mPY * mPY); }
  float PhiPos() const { return TMath::ATan2(mY, mX); }
  float Phi() const { return TMath::ATan2(mPY, mPX); }
  float Theta() const { return TMath::ACos(mPZ / P()); }
  float X() const { return mX; }
  float Y() const { return mY; }
  float Z() const { return mZ; }
  float Px() const { return mPX; }
  float Py() const { return mPY; }
  float Pz() const { return mPZ; }
  float P() const { return TMath::Sqrt(mPX * mPX + mPY * mPY + mPZ * mPZ); }
  Int_t getUserId() const { return mUserId; }
  Int_t getDetectorId() const { return mDetectorId; }
  void setDetectorId(Int_t id) { mDetectorId = id; }

  void setPosition(float x, float y, float z)
  {
    mX = x;
    mY = y;
    mZ = z;
  }

  void setMomentum(float px, float py, float pz)
  {
    mPX = px;
    mPY = py;
    mPZ = pz;
  }

  void setUserId(Int_t userId) { mUserId = userId; }

  // Methods to get position of the track reference in
  // in the TPC/TRD/TOF Tracking coordinate system
  float phiPosition() const { return TMath::Pi() + TMath::ATan2(-mY, -mX); }

  float Alpha() const { return TMath::Pi() * (20 * ((((Int_t)(phiPosition() * 180 / TMath::Pi())) / 20)) + 10) / 180.; }

  float LocalX() const
  {
    auto alpha = Alpha();
    return mX * TMath::Cos(-alpha) - mY * TMath::Sin(-alpha);
  }

  float LocalY() const
  {
    auto alpha = Alpha();
    return mX * TMath::Sin(-alpha) + mY * TMath::Cos(-alpha);
  }

  const SimTrackStatus& getTrackStatus() const { return mStatus; }

 private:
  Int_t mTrackNumber = 0; ///< Track number
  float mX = 0;           ///< X reference position of the track
  float mY = 0;           ///< Y reference position of the track
  float mZ = 0;           ///< Z reference position of the track
  float mPX = 0;          ///< momentum
  float mPY = 0;          ///< momentum
  float mPZ = 0;          ///< momentum
  float mTrackLength = 0; ///< track length from its origin in cm
  float mTof = 0;         ///< time of flight in cm
  Int_t mUserId = 0;      ///< optional Id defined by user
  Int_t mDetectorId = 0;  ///< Detector Id
  SimTrackStatus mStatus; ///< encoding the track status

  friend std::ostream& operator<<(std::ostream&, const TrackReference&);

  ClassDefNV(TrackReference, 1) // Base class for all Alice track references
};

// this is the preferred constructor as it might reuse variables
// already fetched from VMC
inline TrackReference::TrackReference(float x, float y, float z, float px, float py, float pz, float l, float tof,
                                      int trackID, int detlabel)
  : mX(x),
    mY(y),
    mZ(z),
    mPX(px),
    mPY(py),
    mPZ(pz),
    mTrackLength(l),
    mTof(tof),
    mTrackNumber(trackID),
    mDetectorId(detlabel)
{
}

// constructor fetching everything from vmc instance
// less performant than other constructor since
// potentially duplicated virtual function calls (already used in the
// stepping functions)
inline TrackReference::TrackReference(TVirtualMC const& vmc, int detlabel) : mStatus(vmc)
{
  float x, y, z;
  float px, py, pz, e;
  vmc.TrackPosition(x, y, z);
  vmc.TrackMomentum(px, py, pz, e);
  mX = x;
  mY = y;
  mZ = z;
  mPX = px;
  mPY = py;
  mPZ = pz;
  mTrackLength = vmc.TrackLength();
  mTof = vmc.TrackTime();
  mDetectorId = detlabel;
}

inline std::ostream& operator<<(std::ostream& os, const TrackReference& a)
{
  os << "TrackRef (" << a.mTrackNumber << "): X[" << a.mX << " , " << a.mY << " , " << a.mZ << "]"
     << "; P[ " << a.mPX << " , " << a.mPY << " , " << a.mPZ << " ] "
     << "; Length = " << a.mTrackLength << " ; TOF = " << a.mTof << " ; DetID = " << a.mDetectorId
     << "; Status = " << a.mStatus;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const SimTrackStatus& status)
{
  os << status.mStatus;
  return os;
}
}

#endif

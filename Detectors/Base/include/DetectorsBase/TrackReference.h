// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackReference.h
/// \brief Definition of the TrackReference class
/// \author Sylwester Radomski (S.Radomski@gsi.de) GSI, Jan 31, 2003

#ifndef ALICEO2_BASE_TRACKREFERENCE_H_
#define ALICEO2_BASE_TRACKREFERENCE_H_

#include "Rtypes.h"      // for TrackReference::Class, ClassDef, etc
#include "TMath.h"       // for Pi, Sqrt, ATan2, Cos, Sin, ACos
#include "TObject.h"     // for TObject

namespace o2 {
namespace Base {

/// Track Reference object is created every time particle is
/// crossing detector bounds. The object is created by Step Manager
/// The class stores the following informations:
/// track label,
/// track position: X,Y,X
/// track momentum px, py, pz
/// track length and time of fligth: both in cm
/// status bits from Monte Carlo
class TrackReference : public TObject
{

  public:
    enum constants
    {
        kDisappeared = -1,
        kITS = 0,
        kTPC = 1,
        kFRAME = 2,
        kTRD = 3,
        kTOF = 4,
        kMUON = 5,
        kHMPID = 6,
        kFIT = 7,
        kEMCAL = 8,
        kPMD = 10,
        kFMD = 12,
        kVZERO = 14,
        kMFT = 16,
        kHALL = 17
    };

    /// Default Constructor
    TrackReference();

    TrackReference(Int_t label, Int_t id = -999);

    TrackReference(const TrackReference &tr);

    /// Default Destructor
    ~TrackReference()
    override = default;

    // static AliExternalTrackParam * MakeTrack(const TrackReference *ref, Double_t mass);
    virtual Int_t GetTrack() const
    {
      return mTrackNumber;
    }

    virtual void SetTrack(Int_t track)
    {
      mTrackNumber = track;
    }

    virtual void SetLength(Float_t length)
    {
      mTrackLength = length;
    }

    virtual void SetTime(Float_t time)
    {
      mTof = time;
    }

    virtual Float_t GetLength() const
    {
      return mTrackLength;
    }

    virtual Float_t GetTime() const
    {
      return mTof;
    }

    virtual Int_t Label() const
    {
      return mTrackNumber;
    }

    virtual void SetLabel(Int_t track)
    {
      mTrackNumber = track;
    }

    virtual Float_t R() const
    {
      return TMath::Sqrt(mReferencePositionX * mReferencePositionX + mReferencePositionY * mReferencePositionY);
    }

    virtual Float_t Pt() const
    {
      return TMath::Sqrt(mMomentumX * mMomentumX + mMomentumY * mMomentumY);
    }

    virtual Float_t Phi() const
    {
      return TMath::Pi() + TMath::ATan2(-mMomentumY, -mMomentumX);
    }

    virtual Float_t Theta() const
    {
      return (mMomentumZ == 0) ? TMath::Pi() / 2 : TMath::ACos(mMomentumZ / P());
    }

    virtual Float_t X() const
    {
      return mReferencePositionX;
    }

    virtual Float_t Y() const
    {
      return mReferencePositionY;
    }

    virtual Float_t Z() const
    {
      return mReferencePositionZ;
    }

    virtual Float_t Px() const
    {
      return mMomentumX;
    }

    virtual Float_t Py() const
    {
      return mMomentumY;
    }

    virtual Float_t Pz() const
    {
      return mMomentumZ;
    }

    virtual Float_t P() const
    {
      return TMath::Sqrt(mMomentumX * mMomentumX + mMomentumY * mMomentumY + mMomentumZ * mMomentumZ);
    }

    virtual Int_t UserId() const
    {
      return mUserId;
    }

    virtual Int_t DetectorId() const
    {
      return mDetectorId;
    }

    virtual void SetDetectorId(Int_t id)
    {
      mDetectorId = id;
    }

    virtual void setPosition(Float_t x, Float_t y, Float_t z)
    {
      mReferencePositionX = x;
      mReferencePositionY = y;
      mReferencePositionZ = z;
    }

    virtual void SetMomentum(Float_t px, Float_t py, Float_t pz)
    {
      mMomentumX = px;
      mMomentumY = py;
      mMomentumZ = pz;
    }

    virtual void setUserId(Int_t userId)
    {
      mUserId = userId;
    }

    // Methods to get position of the track reference in
    // in the TPC/TRD/TOF Tracking coordinate system

    virtual Float_t phiPosition() const
    {
      return TMath::Pi() + TMath::ATan2(-mReferencePositionY, -mReferencePositionX);
    }

    virtual Float_t Alpha() const
    {
      return TMath::Pi() * (20 * ((((Int_t) (phiPosition() * 180 / TMath::Pi())) / 20)) + 10) / 180.;
    }

    virtual Float_t LocalX() const
    {
      return mReferencePositionX * TMath::Cos(-Alpha()) - mReferencePositionY * TMath::Sin(-Alpha());
    }

    virtual Float_t LocalY() const
    {
      return mReferencePositionX * TMath::Sin(-Alpha()) + mReferencePositionY * TMath::Cos(-Alpha());
    }

    Bool_t isSortable() const
    {
      return kTRUE;
    }

    Int_t Compare(const TObject *obj) const override
    {
      Int_t ll = ((TrackReference *) obj)->GetTrack();
      if (ll < mTrackNumber) {
        return 1;
      }
      if (ll > mTrackNumber) {
        return -1;
      }
      return 0;
    }

    void Print(Option_t *opt = "") const override;

  private:
    Int_t mTrackNumber;          ///< Track number
    Float_t mReferencePositionX; ///< X reference position of the track
    Float_t mReferencePositionY; ///< Y reference position of the track
    Float_t mReferencePositionZ; ///< Z reference position of the track
    Float_t mMomentumX;          ///< momentum
    Float_t mMomentumY;          ///< momentum
    Float_t mMomentumZ;          ///< momentum
    Float_t mTrackLength;        ///< track length from its origin in cm
    Float_t mTof;                ///< time of flight in cm
    Int_t mUserId;               ///< optional Id defined by user
    Int_t mDetectorId;           ///< Detector Id
  ClassDefOverride(TrackReference, 1)  // Base class for all Alice track references
};
}
}

#endif

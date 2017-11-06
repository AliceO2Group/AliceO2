// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MCTrack.h
/// \brief Definition of the MCTrack class
/// \author M. Al-Turany - June 2014

#ifndef ALICEO2_DATA_MCTRACK_H_
#define ALICEO2_DATA_MCTRACK_H_

#include "Rtypes.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TVector3.h"
#include "DetectorsBase/DetID.h"

class TParticle;

namespace o2 {

/// Data class for storing Monte Carlo tracks processed by the Stack.
/// An MCTrack can be a primary track put into the simulation or a
/// secondary one produced by the transport through decay or interaction.
/// This is a light weight particle class that is saved to disk
/// instead of saving the TParticle class. It is also used for filtering the stack
class MCTrack
{

  public:
    ///  Default constructor
    MCTrack();

    ///  Standard constructor
    MCTrack(Int_t pdgCode, Int_t motherID, Double_t px, Double_t py, Double_t pz, Double_t x, Double_t y, Double_t z,
            Double_t t, Int_t nPoints);

    ///  Copy constructor
    MCTrack(const MCTrack &track);

    ///  Constructor from TParticle
    MCTrack(TParticle *particle);

    ///  Destructor
    ~MCTrack();

    ///  Output to screen
    void Print(Int_t iTrack = 0) const;

    ///  Accessors
    Int_t GetPdgCode() const
    {
      return mPdgCode;
    }

    Int_t getMotherTrackId() const
    {
      return mMotherTrackId;
    }

    Double_t GetStartVertexMomentumX() const
    {
      return mStartVertexMomentumX;
    }

    Double_t GetStartVertexMomentumY() const
    {
      return mStartVertexMomentumY;
    }

    Double_t GetStartVertexMomentumZ() const
    {
      return mStartVertexMomentumZ;
    }

    Double_t GetStartVertexCoordinatesX() const
    {
      return mStartVertexCoordinatesX;
    }

    Double_t GetStartVertexCoordinatesY() const
    {
      return mStartVertexCoordinatesY;
    }

    Double_t GetStartVertexCoordinatesZ() const
    {
      return mStartVertexCoordinatesZ;
    }

    Double_t GetStartVertexCoordinatesT() const
    {
      return mStartVertexCoordinatesT;
    }

    Double_t GetMass() const;

    Double_t GetEnergy() const;

    Double_t GetPt() const
    {
      return TMath::Sqrt(mStartVertexMomentumX * mStartVertexMomentumX + mStartVertexMomentumY * mStartVertexMomentumY);
    }

    Double_t GetP() const
    {
      return TMath::Sqrt(mStartVertexMomentumX * mStartVertexMomentumX + mStartVertexMomentumY * mStartVertexMomentumY +
                         mStartVertexMomentumZ * mStartVertexMomentumZ);
    }

    Double_t GetRapidity() const;

    void GetMomentum(TVector3 &momentum);

    void Get4Momentum(TLorentzVector &momentum);

    void GetStartVertex(TVector3 &vertex);

    /// Accessors to the hit mask
    Int_t getHitMask() const { return mHitMask; }
    void setHitMask(Int_t m) { mHitMask = m; }

    ///  Modifiers
    void SetMotherTrackId(Int_t id)
    {
      mMotherTrackId = id;
    }

    static void setHit(Int_t iDet, int& encoding) { encoding |= 1 << iDet; }

    // set bit indicating that this track
    // left a hit in detector with id iDet
    void setHit(Int_t iDet) {
      assert(0<=iDet && iDet < o2::Base::DetID::nDetectors);
      MCTrack::setHit(iDet, mHitMask);
    }

    // did detector iDet see this track?
    bool leftTrace(Int_t iDet) const {
      return (mHitMask & ( 1 << iDet )) > 0;
    }

    // determine how many detectors "saw" this track
    int getNumDet() const
    {
      int count = 0;
      for (auto i = o2::Base::DetID::First; i < o2::Base::DetID::nDetectors; ++i) {
        if (leftTrace(i))
          count++;
      }
      return count;
    }

   private:
    ///  PDG particle code
    Int_t mPdgCode;

    ///  Index of mother track. -1 for primary particles.
    Int_t mMotherTrackId;

    /// Momentum components at start vertex [GeV]
    Double32_t mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ;

    /// Coordinates of start vertex [cm, ns]
    Double32_t mStartVertexCoordinatesX, mStartVertexCoordinatesY, mStartVertexCoordinatesZ, mStartVertexCoordinatesT;

    // hitmask stored as an int
    // if bit i is set it means that this track left a trace in detector i
    // we should have sizeof(int) < o2::Base::DetId::nDetectors
    Int_t mHitMask = 0;

  ClassDefNV(MCTrack, 1);
};

inline Double_t MCTrack::GetEnergy() const
{
  Double_t mass = GetMass();
  return TMath::Sqrt(mass * mass + mStartVertexMomentumX * mStartVertexMomentumX +
                     mStartVertexMomentumY * mStartVertexMomentumY + mStartVertexMomentumZ * mStartVertexMomentumZ);
}

inline void MCTrack::GetMomentum(TVector3 &momentum)
{
  momentum.SetXYZ(mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ);
}

inline void MCTrack::Get4Momentum(TLorentzVector &momentum)
{
  momentum.SetXYZT(mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ, GetEnergy());
}

inline void MCTrack::GetStartVertex(TVector3 &vertex)
{
  vertex.SetXYZ(mStartVertexCoordinatesX, mStartVertexCoordinatesY, mStartVertexCoordinatesZ);
}

}
 
#endif

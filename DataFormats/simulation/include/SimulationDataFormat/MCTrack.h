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
/// \author M. Al-Turany - June 2014; S. Wenzel - October 2017

#ifndef ALICEO2_DATA_MCTRACK_H_
#define ALICEO2_DATA_MCTRACK_H_

#include "Rtypes.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TVector3.h"
#include "DetectorsBase/DetID.h"
#include "TDatabasePDG.h"
#include "TParticle.h"
#include "TParticlePDG.h"
#include "TMCProcess.h"

namespace o2 {

/// Data class for storing Monte Carlo tracks processed by the Stack.
/// An MCTrack can be a primary track put into the simulation or a
/// secondary one produced by the transport through decay or interaction.
/// This is a light weight particle class that is saved to disk
/// instead of saving the TParticle class. It is also used for filtering the stack
template <class T>
class MCTrackT
{

  public:
    ///  Default constructor
    MCTrackT();

    ///  Standard constructor
    MCTrackT(Int_t pdgCode, Int_t motherID, Double_t px, Double_t py, Double_t pz, Double_t x, Double_t y, Double_t z,
            Double_t t, Int_t nPoints);

    ///  Copy constructor
    MCTrackT(const MCTrackT &track) = default;

    ///  Constructor from TParticle
    MCTrackT(const TParticle &particle);

    ///  Destructor
    ~MCTrackT() = default;

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
      double mx(mStartVertexMomentumX);
      double my(mStartVertexMomentumY);
      return std::sqrt(mx * mx + my * my); 
    }

    Double_t GetP() const
    {
      double mx(mStartVertexMomentumX);
      double my(mStartVertexMomentumY);
      double mz(mStartVertexMomentumZ);
      return std::sqrt(mx*mx + my*my + mz*mz);
    }

    Double_t GetRapidity() const;

    void GetMomentum(TVector3 &momentum);

    void Get4Momentum(TLorentzVector &momentum);

    void GetStartVertex(TVector3 &vertex);

    /// Accessors to the hit mask
    Int_t getHitMask() const { return ((PropEncoding)mProp).hitmask; }
    void setHitMask(Int_t m) { ((PropEncoding)mProp).hitmask = m; }

    ///  Modifiers
    void SetMotherTrackId(Int_t id)
    {
      mMotherTrackId = id;
    }

    // set bit indicating that this track
    // left a hit in detector with id iDet
    void setHit(Int_t iDet) {
      assert(0<=iDet && iDet < o2::Base::DetID::nDetectors);
      auto prop = ((PropEncoding)mProp);
      prop.hitmask |= 1 << iDet;
      mProp = prop.i;
    }

    // did detector iDet see this track?
    bool leftTrace(Int_t iDet) const {
      return (((PropEncoding)mProp).hitmask & ( 1 << iDet )) > 0;
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

    // keep track if this track will be persistet
    // using last bit in mHitMask to do so
    void setStore(bool f) { auto prop = ((PropEncoding)mProp); prop.storage = f; mProp=prop.i; }
    bool getStore() const { return ((PropEncoding)mProp).storage; }
    // determine if this track has hits
    bool hasHits() const { return ((PropEncoding)mProp).hitmask!=0; }  

    // set process property
    void setProcess(int proc) { auto prop = ((PropEncoding)mProp); prop.process = proc; mProp=prop.i; }
    int getProcess() const { return ((PropEncoding)mProp).process; }

 private:
    /// Momentum components at start vertex [GeV]
    T mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ;

    /// Coordinates of start vertex [cm, ns]
    T mStartVertexCoordinatesX, mStartVertexCoordinatesY, mStartVertexCoordinatesZ, mStartVertexCoordinatesT;

    ///  PDG particle code
    Int_t mPdgCode;

    ///  Index of mother track. -1 for primary particles.
    Int_t mMotherTrackId;
    
    // hitmask stored as an int
    // if bit i is set it means that this track left a trace in detector i
    // we should have sizeof(int) < o2::Base::DetId::nDetectors
    Int_t mProp = 0;

    // internal structure to allow convenient manipulation
    // of properties as bits on an int
    union PropEncoding {
      PropEncoding(int a) : i(a) {}
      int i;
      struct {
        int storage : 1; // encoding whether to store this track to the output
        int process : 6; // encoding process that created this track (enough to store TMCProcess from ROOT)
        int hitmask : 25; // encoding hits per detector
      };
    };

    ClassDefNV(MCTrackT, 1);
};

template <typename T>
inline Double_t MCTrackT<T>::GetEnergy() const
{
  const auto mass = GetMass();
  return std::sqrt(mass * mass + mStartVertexMomentumX * mStartVertexMomentumX +
                   mStartVertexMomentumY * mStartVertexMomentumY + mStartVertexMomentumZ * mStartVertexMomentumZ);
}
 
template <typename T>
inline void MCTrackT<T>::GetMomentum(TVector3 &momentum)
{
  momentum.SetXYZ(mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ);
}
 
template <typename T>
inline void MCTrackT<T>::Get4Momentum(TLorentzVector &momentum)
{
  momentum.SetXYZT(mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ, GetEnergy());
}

 template <typename T>
inline void MCTrackT<T>::GetStartVertex(TVector3 &vertex)
{
  vertex.SetXYZ(mStartVertexCoordinatesX, mStartVertexCoordinatesY, mStartVertexCoordinatesZ);
}

template <typename T>
inline MCTrackT<T>::MCTrackT()
  : mPdgCode(0),
    mMotherTrackId(-1),
    mStartVertexMomentumX(0.),
    mStartVertexMomentumY(0.),
    mStartVertexMomentumZ(0.),
    mStartVertexCoordinatesX(0.),
    mStartVertexCoordinatesY(0.),
    mStartVertexCoordinatesZ(0.),
    mStartVertexCoordinatesT(0.),
    mProp(0)
{
}

template <typename T>
inline MCTrackT<T>::MCTrackT(Int_t pdgCode, Int_t motherId, Double_t px, Double_t py, Double_t pz, Double_t x,
                             Double_t y, Double_t z, Double_t t, Int_t mask)
  : mPdgCode(pdgCode),
    mMotherTrackId(motherId),
    mStartVertexMomentumX(px),
    mStartVertexMomentumY(py),
    mStartVertexMomentumZ(pz),
    mStartVertexCoordinatesX(x),
    mStartVertexCoordinatesY(y),
    mStartVertexCoordinatesZ(z),
    mStartVertexCoordinatesT(t),
    mProp(mask)
{
}

template <typename T>
inline MCTrackT<T>::MCTrackT(const TParticle& part)
  : mPdgCode(part.GetPdgCode()),
    mMotherTrackId(part.GetMother(0)),
    mStartVertexMomentumX(part.Px()),
    mStartVertexMomentumY(part.Py()),
    mStartVertexMomentumZ(part.Pz()),
    mStartVertexCoordinatesX(part.Vx()),
    mStartVertexCoordinatesY(part.Vy()),
    mStartVertexCoordinatesZ(part.Vz()),
    mStartVertexCoordinatesT(part.T() * 1e09),
    mProp(0)
{
}

template <typename T>
inline
void MCTrackT<T>::Print(Int_t trackId) const
{
  //LOG(DEBUG) << "Track " << trackId << ", mother : " << mMotherTrackId << ", Type " << mPdgCode << ", momentum ("
  //           << mStartVertexMomentumX << ", " << mStartVertexMomentumY << ", " << mStartVertexMomentumZ << ") GeV"
  //           << FairLogger::endl;
}

template <typename T>
inline Double_t MCTrackT<T>::GetMass() const
{
  if (TDatabasePDG::Instance()) {
    TParticlePDG* particle = TDatabasePDG::Instance()->GetParticle(mPdgCode);
    if (particle) {
      return particle->Mass();
    } else {
      return 0.;
    }
  }
  return 0.;
}

template <typename T>
inline Double_t MCTrackT<T>::GetRapidity() const
{
  const auto e = GetEnergy();
  Double_t y =
    0.5 * std::log((e + static_cast<double>(mStartVertexMomentumZ)) / (e - static_cast<double>(mStartVertexMomentumZ)));
  return y;
}

using MCTrack = MCTrackT<float>;

}
 
#endif

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

/// \file MCTrack.h
/// \brief Definition of the MCTrack class
/// \author M. Al-Turany - June 2014; S. Wenzel - October 2017

#ifndef ALICEO2_DATA_MCTRACK_H_
#define ALICEO2_DATA_MCTRACK_H_

#include "SimulationDataFormat/ParticleStatus.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Rtypes.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "TLorentzVector.h"
#include "TMCProcess.h"
#include "TMath.h"
#include "TParticle.h"
#include "TParticlePDG.h"
#include "TVector3.h"

namespace o2
{

namespace MCTrackHelper
{
void printMassError(int pdg);
};

/// Data class for storing Monte Carlo tracks processed by the Stack.
/// An MCTrack can be a primary track put into the simulation or a
/// secondary one produced by the transport through decay or interaction.
/// This is a light weight particle class that is saved to disk
/// instead of saving the TParticle class. It is also used for filtering the stack
template <class _T>
class MCTrackT
{
 public:
  ///  Default constructor
  MCTrackT();

  ///  Standard constructor
  MCTrackT(Int_t pdgCode, Int_t motherID, Int_t secondMotherID, Int_t firstDaughterID, Int_t lastDaughterID,
           Double_t px, Double_t py, Double_t pz, Double_t x, Double_t y, Double_t z, Double_t t,
           Int_t nPoints);

  ///  Copy constructor
  MCTrackT(const MCTrackT& track) = default;

  ///  Constructor from TParticle
  MCTrackT(const TParticle& particle);

  ///  Destructor
  ~MCTrackT() = default;

  ///  Output to screen
  void Print(Int_t iTrack = 0) const;

  ///  Accessors
  Int_t GetPdgCode() const { return mPdgCode; }
  Int_t getMotherTrackId() const { return mMotherTrackId; }
  Int_t getSecondMotherTrackId() const { return mSecondMotherTrackId; }
  bool isPrimary() const { return getProcess() == TMCProcess::kPPrimary; }
  bool isSecondary() const { return !isPrimary(); }
  Int_t getFirstDaughterTrackId() const { return mFirstDaughterTrackId; }
  Int_t getLastDaughterTrackId() const { return mLastDaughterTrackId; }
  Double_t GetStartVertexMomentumX() const { return mStartVertexMomentumX; }
  Double_t GetStartVertexMomentumY() const { return mStartVertexMomentumY; }
  Double_t GetStartVertexMomentumZ() const { return mStartVertexMomentumZ; }
  Double_t GetStartVertexCoordinatesX() const { return mStartVertexCoordinatesX; }
  Double_t GetStartVertexCoordinatesY() const { return mStartVertexCoordinatesY; }
  Double_t GetStartVertexCoordinatesZ() const { return mStartVertexCoordinatesZ; }
  Double_t GetStartVertexCoordinatesT() const { return mStartVertexCoordinatesT; }

  /// return mass from PDG Database if known (print message in case cannot look up)
  Double_t GetMass() const;

  /// return particle weight
  _T getWeight() const { return mWeight; }

  Double_t GetEnergy() const;

  // Alternative accessors with TParticle like shorter names
  Double_t Px() const { return mStartVertexMomentumX; }
  Double_t Py() const { return mStartVertexMomentumY; }
  Double_t Pz() const { return mStartVertexMomentumZ; }
  Double_t Vx() const { return mStartVertexCoordinatesX; }
  Double_t Vy() const { return mStartVertexCoordinatesY; }
  Double_t Vz() const { return mStartVertexCoordinatesZ; }
  Double_t T() const { return mStartVertexCoordinatesT; }

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
    return std::sqrt(mx * mx + my * my + mz * mz);
  }

  Double_t GetPhi() const
  {
    double mx(mStartVertexMomentumX);
    double my(mStartVertexMomentumY);
    return (TMath::Pi() + TMath::ATan2(-my, -mx));
  }

  Double_t GetEta() const
  {
    double_t pmom = GetP();
    double mz(mStartVertexMomentumZ);
    if (pmom != TMath::Abs(mz)) {
      return 0.5 * std::log((pmom + mz) / (pmom - mz));
    } else {
      return 1.e30;
    }
  }

  Double_t GetTheta() const
  {
    double mz(mStartVertexMomentumZ);
    return (mz == 0) ? TMath::PiOver2() : TMath::ACos(mz / GetP());
  }

  Double_t GetRapidity() const;

  void GetMomentum(TVector3& momentum) const;

  void Get4Momentum(TLorentzVector& momentum) const;

  void GetStartVertex(TVector3& vertex) const;

  /// Accessors to the hit mask
  Int_t getHitMask() const { return ((PropEncoding)mProp).hitmask; }
  void setHitMask(Int_t m) { ((PropEncoding)mProp).hitmask = m; }
  ///  Modifiers
  void SetMotherTrackId(Int_t id) { mMotherTrackId = id; }
  void SetSecondMotherTrackId(Int_t id) { mSecondMotherTrackId = id; }
  void SetFirstDaughterTrackId(Int_t id) { mFirstDaughterTrackId = id; }
  void SetLastDaughterTrackId(Int_t id) { mLastDaughterTrackId = id; }
  // set bit indicating that this track
  // left a hit in detector with id iDet
  void setHit(Int_t iDet)
  {
    assert(0 <= iDet && iDet < o2::detectors::DetID::nDetectors);
    auto prop = ((PropEncoding)mProp);
    prop.hitmask |= 1 << iDet;
    mProp = prop.i;
  }

  // did detector iDet see this track?
  bool leftTrace(Int_t iDet) const { return (((PropEncoding)mProp).hitmask & (1 << iDet)) > 0; }
  // determine how many detectors "saw" this track
  int getNumDet() const
  {
    int count = 0;
    for (auto i = o2::detectors::DetID::First; i < o2::detectors::DetID::nDetectors; ++i) {
      if (leftTrace(i)) {
        count++;
      }
    }
    return count;
  }

  // keep track if this track will be persistet
  // using last bit in mHitMask to do so
  void setStore(bool f)
  {
    auto prop = ((PropEncoding)mProp);
    prop.storage = f;
    mProp = prop.i;
  }
  bool getStore() const { return ((PropEncoding)mProp).storage; }
  /// determine if this track has hits
  bool hasHits() const { return ((PropEncoding)mProp).hitmask != 0; }
  /// set process property
  void setProcess(int proc)
  {
    auto prop = ((PropEncoding)mProp);
    prop.process = proc;
    mProp = prop.i;
  }

  /// get the production process (id) of this track
  int getProcess() const { return ((PropEncoding)mProp).process; }

  /// get generator status code
  o2::mcgenstatus::MCGenStatusEncoding getStatusCode() const { return ((o2::mcgenstatus::MCGenStatusEncoding)mStatusCode); }

  void setToBeDone(bool f)
  {
    auto prop = ((PropEncoding)mProp);
    prop.toBeDone = f;
    mProp = prop.i;
  }
  bool getToBeDone() const { return ((PropEncoding)mProp).toBeDone; }

  void setInhibited(bool f)
  {
    auto prop = ((PropEncoding)mProp);
    prop.inhibited = f;
    mProp = prop.i;
  }
  bool getInhibited() const { return ((PropEncoding)mProp).inhibited; }

  bool isTransported() const { return getToBeDone() && !getInhibited(); };

  /// get the string representation of the production process
  const char* getProdProcessAsString() const;

 private:
  /// Momentum components at start vertex [GeV]
  _T mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ;

  /// Coordinates of start vertex [cm, ns]
  _T mStartVertexCoordinatesX, mStartVertexCoordinatesY, mStartVertexCoordinatesZ, mStartVertexCoordinatesT;

  /// particle weight
  _T mWeight;

  ///  PDG particle code
  Int_t mPdgCode;

  ///  Index of mother tracks
  Int_t mMotherTrackId = -1;
  Int_t mSecondMotherTrackId = -1;

  Int_t mFirstDaughterTrackId = -1;
  Int_t mLastDaughterTrackId = -1;
  // hitmask stored as an int
  // if bit i is set it means that this track left a trace in detector i
  // we should have sizeof(int) < o2::base::DetId::nDetectors
  Int_t mProp = 0;

  // internal structure to allow convenient manipulation
  // of properties as bits on an int
  union PropEncoding {
    PropEncoding(int a) : i(a) {}
    int i;
    struct {
      int storage : 1;  // encoding whether to store this track to the output
      unsigned int process : 6; // encoding process that created this track (enough to store TMCProcess from ROOT)
      int hitmask : 22;         // encoding hits per detector
      int reserved1 : 1;        // bit reserved for possible future purposes
      int inhibited : 1; // whether tracking of this was inhibited
      int toBeDone : 1; // whether this (still) needs tracking --> we might more complete information to cover full ParticleStatus space
    };
  };

  // Additional status codes for MC generator information.
  // NOTE: This additional memory cost might be reduced by using bits elsewhere
  // such as part of mProp (process) or mPDG
  Int_t mStatusCode = 0;

  ClassDefNV(MCTrackT, 8);
};

template <typename T>
inline Double_t MCTrackT<T>::GetEnergy() const
{
  const auto mass = GetMass();
  return std::sqrt(mass * mass + mStartVertexMomentumX * mStartVertexMomentumX +
                   mStartVertexMomentumY * mStartVertexMomentumY + mStartVertexMomentumZ * mStartVertexMomentumZ);
}

template <typename T>
inline void MCTrackT<T>::GetMomentum(TVector3& momentum) const
{
  momentum.SetXYZ(mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ);
}

template <typename T>
inline void MCTrackT<T>::Get4Momentum(TLorentzVector& momentum) const
{
  momentum.SetXYZT(mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ, GetEnergy());
}

template <typename T>
inline void MCTrackT<T>::GetStartVertex(TVector3& vertex) const
{
  vertex.SetXYZ(mStartVertexCoordinatesX, mStartVertexCoordinatesY, mStartVertexCoordinatesZ);
}

template <typename T>
inline MCTrackT<T>::MCTrackT()
  : mPdgCode(0),
    mMotherTrackId(-1),
    mSecondMotherTrackId(-1),
    mFirstDaughterTrackId(-1),
    mLastDaughterTrackId(-1),
    mStartVertexMomentumX(0.),
    mStartVertexMomentumY(0.),
    mStartVertexMomentumZ(0.),
    mStartVertexCoordinatesX(0.),
    mStartVertexCoordinatesY(0.),
    mStartVertexCoordinatesZ(0.),
    mStartVertexCoordinatesT(0.),
    mProp(0),
    mWeight(0)
{
}

template <typename T>
inline MCTrackT<T>::MCTrackT(Int_t pdgCode, Int_t motherId, Int_t secondMotherId, Int_t firstDaughterId, Int_t lastDaughterId,
                             Double_t px, Double_t py, Double_t pz, Double_t x,
                             Double_t y, Double_t z, Double_t t, Int_t mask)
  : mPdgCode(pdgCode),
    mMotherTrackId(motherId),
    mSecondMotherTrackId(secondMotherId),
    mFirstDaughterTrackId(firstDaughterId),
    mLastDaughterTrackId(lastDaughterId),
    mStartVertexMomentumX(px),
    mStartVertexMomentumY(py),
    mStartVertexMomentumZ(pz),
    mStartVertexCoordinatesX(x),
    mStartVertexCoordinatesY(y),
    mStartVertexCoordinatesZ(z),
    mStartVertexCoordinatesT(t),
    mProp(mask),
    mWeight(0)
{
}

template <typename T>
inline MCTrackT<T>::MCTrackT(const TParticle& part)
  : mPdgCode(part.GetPdgCode()),
    mMotherTrackId(part.GetMother(0)),
    mSecondMotherTrackId(part.GetMother(1)),
    mFirstDaughterTrackId(part.GetFirstDaughter()),
    mLastDaughterTrackId(part.GetLastDaughter()),
    mStartVertexMomentumX(part.Px()),
    mStartVertexMomentumY(part.Py()),
    mStartVertexMomentumZ(part.Pz()),
    mStartVertexCoordinatesX(part.Vx()),
    mStartVertexCoordinatesY(part.Vy()),
    mStartVertexCoordinatesZ(part.Vz()),
    mStartVertexCoordinatesT(part.T() * 1e09),
    mWeight(part.GetWeight()),
    mProp(0),
    mStatusCode(0)
{
  // our convention is to communicate the process as (part) of the unique ID
  setProcess(part.GetUniqueID());
  // extract storage flag
  setStore(part.TestBit(ParticleStatus::kKeep));
  // extract toBeDone flag
  setToBeDone(part.TestBit(ParticleStatus::kToBeDone));
  // extract inhibited flag
  if (part.TestBit(ParticleStatus::kInhibited)) {
    setToBeDone(true); // if inhibited, it had to be done: restore flag
    setInhibited(true);
  }
  // set MC generator status code only for primaries
  mStatusCode = part.TestBit(ParticleStatus::kPrimary) ? part.GetStatusCode() : -1;
}

template <typename T>
inline void MCTrackT<T>::Print(Int_t trackId) const
{
  // LOG(debug) << "Track " << trackId << ", mother : " << mMotherTrackId << ", Type " << mPdgCode << ", momentum ("
  //           << mStartVertexMomentumX << ", " << mStartVertexMomentumY << ", " << mStartVertexMomentumZ << ") GeV"
  //          ;
}

template <typename T>
inline Double_t MCTrackT<T>::GetMass() const
{
  bool success{};
  auto mass = O2DatabasePDG::Mass(mPdgCode, success);
  if (!success) {
    // coming here is a mistake which should not happen
    MCTrackHelper::printMassError(mPdgCode);
  }
  return mass;
}

template <typename T>
inline Double_t MCTrackT<T>::GetRapidity() const
{
  const auto e = GetEnergy();
  Double_t y =
    0.5 * std::log((e + static_cast<double>(mStartVertexMomentumZ)) / (e - static_cast<double>(mStartVertexMomentumZ)));
  return y;
}

template <typename T>
inline const char* MCTrackT<T>::getProdProcessAsString() const
{
  auto procID = getProcess();
  if (procID >= 0) {
    return TMCProcessName[procID];
  } else {
    return TMCProcessName[TMCProcess::kPNoProcess];
  }
}

using MCTrack = MCTrackT<float>;
} // end namespace o2

#endif

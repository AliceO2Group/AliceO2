/// \file MCTrack.h
/// \brief Definition of the MCTrack class
/// \author M. Al-Turany - June 2014

#ifndef ALICEO2_DATA_MCTRACK_H_
#define ALICEO2_DATA_MCTRACK_H_

#include "TObject.h"
#include "Rtypes.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TVector3.h"

#include "SimulationDataFormat/DetectorList.h"

class TParticle;

/// Data class for storing Monte Carlo tracks processed by the Stack.
/// An MCTrack can be a primary track put into the simulation or a
/// secondary one produced by the transport through decay or interaction.
/// This is a light weight particle class that is saved to disk
/// instead of saving the TParticle class. It is also used for filtering the stack
class MCTrack : public TObject
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
    ~MCTrack() override;

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

    /// Accessors to the number of MCPoints in the detectors
    Int_t getNumberOfPoints(DetectorId detId) const;

    ///  Modifiers
    void SetMotherTrackId(Int_t id)
    {
      mMotherTrackId = id;
    }

    void setNumberOfPoints(Int_t iDet, Int_t np);

  private:
    ///  PDG particle code
    Int_t mPdgCode;

    ///  Index of mother track. -1 for primary particles.
    Int_t mMotherTrackId;

    /// Momentum components at start vertex [GeV]
    Double32_t mStartVertexMomentumX, mStartVertexMomentumY, mStartVertexMomentumZ;

    /// Coordinates of start vertex [cm, ns]
    Double32_t mStartVertexCoordinatesX, mStartVertexCoordinatesY, mStartVertexCoordinatesZ, mStartVertexCoordinatesT;

    /// Bitvector representing the number of MCPoints for this track in
    /// each subdetector. The detectors can be represented by (example from CBM)
    /// REF:         Bit  0      (1 bit,  max. value  1)
    /// MVD:         Bit  1 -  3 (3 bits, max. value  7)
    /// STS:         Bit  4 -  8 (5 bits, max. value 31)
    /// RICH:        Bit  9      (1 bit,  max. value  1)
    /// MUCH:        Bit 10 - 14 (5 bits, max. value 31)
    /// TRD:         Bit 15 - 19 (5 bits, max. value 31)
    /// TOF:         Bit 20 - 23 (4 bits, max. value 15)
    /// ECAL:        Bit 24      (1 bit,  max. value  1)
    /// ZDC:         Bit 25      (1 bit,  max. value  1)
    /// The respective point numbers can be accessed and modified
    /// with the inline functions.
    /// Bits 26-31 are spare for potential additional detectors.

    Int_t mNumberOfPoints;

  ClassDefOverride(MCTrack, 1);
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

#endif

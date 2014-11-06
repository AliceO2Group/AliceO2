/// \file MCTrack.cxx
/// \brief Implementation of the MCTrack class
/// \author M. Al-Turany - June 2014

#include "MCTrack.h"

#include "FairLogger.h"
#include "TDatabasePDG.h"
#include "TParticle.h"
#include "TParticlePDG.h"

MCTrack::MCTrack()
  : TObject(),
    mPdgCode(0),
    mMotherTrackId(-1),
    mStartVertexMomentumX(0.),
    mStartVertexMomentumY(0.),
    mStartVertexMomentumZ(0.),
    mStartVertexCoordinatesX(0.),
    mStartVertexCoordinatesY(0.),
    mStartVertexCoordinatesZ(0.),
    mStartVertexCoordinatesT(0.),
    fNPoints(0)
{
}

MCTrack::MCTrack(Int_t pdgCode, Int_t motherId, Double_t px, Double_t py, Double_t pz, Double_t x,
                 Double_t y, Double_t z, Double_t t, Int_t nPoints = 0)
  : TObject(),
    mPdgCode(pdgCode),
    mMotherTrackId(motherId),
    mStartVertexMomentumX(px),
    mStartVertexMomentumY(py),
    mStartVertexMomentumZ(pz),
    mStartVertexCoordinatesX(x),
    mStartVertexCoordinatesY(y),
    mStartVertexCoordinatesZ(z),
    mStartVertexCoordinatesT(t),
    fNPoints(nPoints)
{
}

MCTrack::MCTrack(const MCTrack& track)
  : TObject(track),
    mPdgCode(track.mPdgCode),
    mMotherTrackId(track.mMotherTrackId),
    mStartVertexMomentumX(track.mStartVertexMomentumX),
    mStartVertexMomentumY(track.mStartVertexMomentumY),
    mStartVertexMomentumZ(track.mStartVertexMomentumZ),
    mStartVertexCoordinatesX(track.mStartVertexCoordinatesX),
    mStartVertexCoordinatesY(track.mStartVertexCoordinatesY),
    mStartVertexCoordinatesZ(track.mStartVertexCoordinatesZ),
    mStartVertexCoordinatesT(track.mStartVertexCoordinatesT),
    fNPoints(track.fNPoints)
{
}

MCTrack::MCTrack(TParticle* part)
  : TObject(),
    mPdgCode(part->GetPdgCode()),
    mMotherTrackId(part->GetMother(0)),
    mStartVertexMomentumX(part->Px()),
    mStartVertexMomentumY(part->Py()),
    mStartVertexMomentumZ(part->Pz()),
    mStartVertexCoordinatesX(part->Vx()),
    mStartVertexCoordinatesY(part->Vy()),
    mStartVertexCoordinatesZ(part->Vz()),
    mStartVertexCoordinatesT(part->T() * 1e09),
    fNPoints(0)
{
}

MCTrack::~MCTrack()
{
}

void MCTrack::Print(Int_t trackId) const
{
  LOG(DEBUG) << "Track " << trackId << ", mother : " << mMotherTrackId << ", Type " << mPdgCode
             << ", momentum (" << mStartVertexMomentumX << ", " << mStartVertexMomentumY << ", "
             << mStartVertexMomentumZ << ") GeV" << FairLogger::endl;
  // LOG(DEBUG2) << "       Ref " << GetNumberOfPoints(kREF)
  //           << ", TutDet " << GetNumberOfPoints(kTutDet)
  //           << ", Rutherford " << GetNumberOfPoints(kFairRutherford)
  //           << FairLogger::endl;
  //
}

Double_t MCTrack::GetMass() const
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

Double_t MCTrack::GetRapidity() const
{
  Double_t e = GetEnergy();
  Double_t y = 0.5 * TMath::Log((e + mStartVertexMomentumZ) / (e - mStartVertexMomentumZ));
  return y;
}

Int_t MCTrack::GetNumberOfPoints(DetectorId detId) const
{
  //   // TODO: Where does this come from
  // if      ( detId == kREF  ) { return (  fNPoints &   1); }
  // else if ( detId == kTutDet  ) { return ( (fNPoints & ( 7 <<  1) ) >>  1); }
  // else if ( detId == kFairRutherford ) { return ( (fNPoints & (31 <<  4) ) >>  4); }
  // else {
  //   LOG(ERROR) << "Unknown detector ID "
  //              << detId << FairLogger::endl;
  //   return 0;
  // }
  //
}

void MCTrack::SetNumberOfPoints(Int_t iDet, Int_t nPoints)
{
  //
  // if ( iDet == kREF ) {
  //   if      ( nPoints < 0 ) { nPoints = 0; }
  //   else if ( nPoints > 1 ) { nPoints = 1; }
  //   fNPoints = ( fNPoints & ( ~ 1 ) )  |  nPoints;
  // }

  // else if ( iDet == kTutDet ) {
  //   if      ( nPoints < 0 ) { nPoints = 0; }
  //   else if ( nPoints > 7 ) { nPoints = 7; }
  //   fNPoints = ( fNPoints & ( ~ (  7 <<  1 ) ) )  |  ( nPoints <<  1 );
  // }

  // else if ( iDet == kFairRutherford ) {
  //   if      ( nPoints <  0 ) { nPoints =  0; }
  //   else if ( nPoints > 31 ) { nPoints = 31; }
  //   fNPoints = ( fNPoints & ( ~ ( 31 <<  4 ) ) )  |  ( nPoints <<  4 );
  // }

  // else LOG(ERROR) << "Unknown detector ID "
  //                   << iDet << FairLogger::endl;
  //
}

ClassImp(MCTrack)

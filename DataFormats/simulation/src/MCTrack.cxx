// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MCTrack.cxx
/// \brief Implementation of the MCTrack class
/// \author M. Al-Turany, S. Wenzel - June 2014

#include "SimulationDataFormat/MCTrack.h"

#include "FairLogger.h"
#include "TDatabasePDG.h"
#include "TParticle.h"
#include "TParticlePDG.h"

ClassImp(o2::MCTrack);

namespace o2 {

MCTrack::MCTrack()
  : mPdgCode(0),
    mMotherTrackId(-1),
    mStartVertexMomentumX(0.),
    mStartVertexMomentumY(0.),
    mStartVertexMomentumZ(0.),
    mStartVertexCoordinatesX(0.),
    mStartVertexCoordinatesY(0.),
    mStartVertexCoordinatesZ(0.),
    mStartVertexCoordinatesT(0.),
    mHitMask(0)
{
}

MCTrack::MCTrack(Int_t pdgCode, Int_t motherId, Double_t px, Double_t py, Double_t pz, Double_t x, Double_t y,
                 Double_t z, Double_t t, Int_t nPoints = 0)
  : mPdgCode(pdgCode),
    mMotherTrackId(motherId),
    mStartVertexMomentumX(px),
    mStartVertexMomentumY(py),
    mStartVertexMomentumZ(pz),
    mStartVertexCoordinatesX(x),
    mStartVertexCoordinatesY(y),
    mStartVertexCoordinatesZ(z),
    mStartVertexCoordinatesT(t),
    mHitMask(nPoints)
{
}

MCTrack::MCTrack(const MCTrack &track)
  = default;

MCTrack::MCTrack(TParticle *part)
  : mPdgCode(part->GetPdgCode()),
    mMotherTrackId(part->GetMother(0)),
    mStartVertexMomentumX(part->Px()),
    mStartVertexMomentumY(part->Py()),
    mStartVertexMomentumZ(part->Pz()),
    mStartVertexCoordinatesX(part->Vx()),
    mStartVertexCoordinatesY(part->Vy()),
    mStartVertexCoordinatesZ(part->Vz()),
    mStartVertexCoordinatesT(part->T() * 1e09),
    mHitMask(0)
{
}

MCTrack::~MCTrack()
= default;

void MCTrack::Print(Int_t trackId) const
{
  LOG(DEBUG) << "Track " << trackId << ", mother : " << mMotherTrackId << ", Type " << mPdgCode << ", momentum ("
             << mStartVertexMomentumX << ", " << mStartVertexMomentumY << ", " << mStartVertexMomentumZ << ") GeV"
             << FairLogger::endl;
}

Double_t MCTrack::GetMass() const
{
  if (TDatabasePDG::Instance()) {
    TParticlePDG *particle = TDatabasePDG::Instance()->GetParticle(mPdgCode);
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

}

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

/// \file GPUTPCMCTrack.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCMCTrack.h"
#include "GPUCommonMath.h"
#include "TDatabasePDG.h"
#include "TParticle.h"

GPUTPCMCTrack::GPUTPCMCTrack() : fPDG(0), fP(0), fPt(0), mNHits(0), fNMCPoints(0), fFirstMCPointID(0), fNReconstructed(0), fSet(0), fNTurns(0)
{
  //* Default constructor
  for (int32_t i = 0; i < 7; i++) {
    fPar[i] = 0;
    fTPCPar[i] = 0;
  }
}

GPUTPCMCTrack::GPUTPCMCTrack(const TParticle* part) : fPDG(0), fP(0), fPt(0), mNHits(0), fNMCPoints(0), fFirstMCPointID(0), fNReconstructed(0), fSet(0), fNTurns(0)
{
  //* Constructor from TParticle

  for (int32_t i = 0; i < 7; i++) {
    fPar[i] = 0;
  }
  for (int32_t i = 0; i < 7; i++) {
    fTPCPar[i] = 0;
  }
  fP = 0;
  fPt = 0;

  if (!part) {
    return;
  }
  TLorentzVector mom, vtx;
  part->ProductionVertex(vtx);
  part->Momentum(mom);
  fPar[0] = part->Vx();
  fPar[1] = part->Vy();
  fPar[2] = part->Vz();
  fP = part->P();
  fPt = part->Pt();
  double pi = (fP > 1.e-4) ? 1. / fP : 0;
  fPar[3] = part->Px() * pi;
  fPar[4] = part->Py() * pi;
  fPar[5] = part->Pz() * pi;
  fPar[6] = 0;
  fPDG = part->GetPdgCode();
  if (CAMath::Abs(fPDG) < 100000) {
    TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(fPDG);
    if (pPDG) {
      fPar[6] = pPDG->Charge() / 3.0 * pi;
    }
  }
}

void GPUTPCMCTrack::SetTPCPar(float X, float Y, float Z, float Px, float Py, float Pz)
{
  //* Set parameters at TPC entrance

  for (int32_t i = 0; i < 7; i++) {
    fTPCPar[i] = 0;
  }

  fTPCPar[0] = X;
  fTPCPar[1] = Y;
  fTPCPar[2] = Z;
  double p = CAMath::Sqrt(Px * Px + Py * Py + Pz * Pz);
  double pi = (p > 1.e-4) ? 1. / p : 0;
  fTPCPar[3] = Px * pi;
  fTPCPar[4] = Py * pi;
  fTPCPar[5] = Pz * pi;
  fTPCPar[6] = 0;
  if (CAMath::Abs(fPDG) < 100000) {
    TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(fPDG);
    if (pPDG) {
      fTPCPar[6] = pPDG->Charge() / 3.0 * pi;
    }
  }
}

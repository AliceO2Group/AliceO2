//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAMCTRACK_H
#define ALIHLTTPCCAMCTRACK_H

#include "AliHLTTPCCADef.h"

class TParticle;


/**
 * @class AliHLTTPCCAMCTrack
 * store MC track information for AliHLTTPCCAPerformance
 */
class AliHLTTPCCAMCTrack
{
 public:

  AliHLTTPCCAMCTrack();
  AliHLTTPCCAMCTrack( const TParticle *part );

  void SetTPCPar( Float_t X, Float_t Y, Float_t Z, Float_t Px, Float_t Py, Float_t Pz );

  Int_t     PDG()            const { return fPDG;}
  const Double_t *Par()            const { return fPar; }
  const Double_t *TPCPar()         const { return fTPCPar; }
  Double_t  P()              const { return fP; }
  Double_t  Pt()             const { return fPt; }
  
  Int_t     NHits()          const { return fNHits;}
  Int_t     NMCPoints()      const { return fNMCPoints;}
  Int_t     FirstMCPointID() const { return fFirstMCPointID;}
  Int_t     NReconstructed() const { return fNReconstructed; }
  Int_t     Set()            const { return fSet; }
  Int_t     NTurns()         const { return fNTurns; }

  void SetP ( Float_t v )          { fP = v; }
  void SetPt( Float_t v )          { fPt = v; }
  void SetPDG( Int_t v )         { fPDG = v; }
  void SetPar( Int_t i, Double_t v)             { fPar[i] = v; }
  void SetTPCPar( Int_t i, Double_t v)          { fTPCPar[i] = v; }
  void SetNHits( Int_t v )         { fNHits = v; }
  void SetNMCPoints( Int_t v)      { fNMCPoints = v; }
  void SetFirstMCPointID( Int_t v ){ fFirstMCPointID = v;}
  void SetNReconstructed( Int_t v ){ fNReconstructed = v; }
  void SetSet( Int_t v )           { fSet = v; }
  void SetNTurns( Int_t v )        { fNTurns = v; }
  
 protected:

  Int_t    fPDG;            //* particle pdg code
  Double_t fPar[7];         //* x,y,z,ex,ey,ez,q/p
  Double_t fTPCPar[7];      //* x,y,z,ex,ey,ez,q/p at TPC entrance (x=y=0 means no information)
  Double_t fP, fPt;         //* momentum and transverse momentum
  Int_t    fNHits;          //* N TPC clusters
  Int_t    fNMCPoints;      //* N MC points 
  Int_t    fFirstMCPointID; //* id of the first MC point in the points array
  Int_t    fNReconstructed; //* how many times is reconstructed
  Int_t    fSet;            //* set of tracks 0-OutSet, 1-ExtraSet, 2-RefSet 
  Int_t    fNTurns;         //* N of turns in the current sector
  
};

#endif

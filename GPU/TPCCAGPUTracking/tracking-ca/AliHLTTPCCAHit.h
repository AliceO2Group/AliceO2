//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAHIT_H
#define ALIHLTTPCCAHIT_H

#include "Rtypes.h"

/**
 * @class AliHLTTPCCAHit
 *
 * The AliHLTTPCCAHit class is the internal representation
 * of the TPC clusters for the AliHLTTPCCATracker algorithm.
 *
 */
class AliHLTTPCCAHit
{
 public:
  AliHLTTPCCAHit(): fY(0),fZ(0),fErrY(0),fErrZ(0),fID(0){;}
  virtual ~AliHLTTPCCAHit(){}

  Float_t &Y(){ return fY; }
  Float_t &Z(){ return fZ; }
  Float_t &ErrY(){ return fErrY; } 
  Float_t &ErrZ(){ return fErrZ; }
  
  Int_t &ID(){ return fID; }
 
  void Set( Int_t HitID, Double_t HitY, Double_t HitZ, 
	    Double_t HitErrY, Double_t HitErrZ  );

 protected:

  Float_t fY, fZ;       // Y and Z position of the TPC cluster
  Float_t fErrY, fErrZ; // position errors
  Int_t fID;            // external unique ID of this hit, 
                        // used as cluster index in track->hit reference array
 private:

  void Dummy(); // to make rulechecker happy by having something in .cxx file

  ClassDef(AliHLTTPCCAHit,1);
};



inline void AliHLTTPCCAHit::Set( Int_t HitID, Double_t HitY, Double_t HitZ, 
				 Double_t HitErrY, Double_t HitErrZ  )
{
  //* set parameters
  fID = HitID;
  fY = HitY;
  fZ = HitZ;
  fErrY = HitErrY;
  fErrZ = HitErrZ;
}


#endif

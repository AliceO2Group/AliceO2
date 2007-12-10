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
 */
class AliHLTTPCCAHit
{
 public:
  AliHLTTPCCAHit(): fY(0),fZ(0),fErrY(0),fErrZ(0),fID(0){;}

  Float_t &Y(){ return fY; }
  Float_t &Z(){ return fZ; }
  Float_t &ErrY(){ return fErrY; } 
  Float_t &ErrZ(){ return fErrZ; }
  
  Int_t &ID(){ return fID; }
 
  void Set( Int_t ID, Double_t Y, Double_t Z, 
	    Double_t ErrY, Double_t ErrZ  );


 protected:
  Float_t fY, fZ;       // Y and Z position of the TPC cluster
  Float_t fErrY, fErrZ; // position errors
  Int_t fID;            // external ID of this hit, 
                        // used as cluster index in track->hit reference array
  

  ClassDef(AliHLTTPCCAHit,1);
};


#endif

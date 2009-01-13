//-*- Mode: C++ -*-

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAMCPOINT_H
#define ALIHLTTPCCAMCPOINT_H

#include "AliHLTTPCCADef.h"


/**
 * @class AliHLTTPCCAMCPoint
 * store MC point information for AliHLTTPCCAPerformance
 */
class AliHLTTPCCAMCPoint
{
 public:

  AliHLTTPCCAMCPoint();

  Float_t  &X()           { return fX; }
  Float_t  &Y()           { return fY; }
  Float_t  &Z()           { return fZ; }
  Float_t  &Sx()          { return fSx; }
  Float_t  &Sy()          { return fSy; }
  Float_t  &Sz()          { return fSz; }
  Float_t  &Time()        { return fTime; }
  Int_t    &ISlice()      { return fISlice; }
  Int_t    &TrackID()     { return fTrackID; }

  static Bool_t Compare( const AliHLTTPCCAMCPoint &p1, const AliHLTTPCCAMCPoint &p2 )
    {
      return (p1.fTrackID < p2.fTrackID);
    }
  
 protected:

  Float_t fX;         //* global X position
  Float_t fY;         //* global Y position
  Float_t fZ;         //* global Z position
  Float_t fSx;        //* slice X position
  Float_t fSy;        //* slice Y position
  Float_t fSz;        //* slice Z position
  Float_t fTime;      //* time 
  Int_t   fISlice;    //* slice number
  Int_t   fTrackID;   //* mc track number
};

#endif

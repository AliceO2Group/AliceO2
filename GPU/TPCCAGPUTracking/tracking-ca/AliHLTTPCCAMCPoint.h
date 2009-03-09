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

  Float_t  X()           const { return fX; }
  Float_t  Y()           const { return fY; }
  Float_t  Z()           const { return fZ; }
  Float_t  Sx()          const { return fSx; }
  Float_t  Sy()          const { return fSy; }
  Float_t  Sz()          const { return fSz; }
  Float_t  Time()        const { return fTime; }
  Int_t    ISlice()      const { return fISlice; }
  Int_t    TrackID()     const { return fTrackID; }

  void SetX( Float_t v )           { fX=v; }
  void SetY( Float_t v )           { fY=v; }
  void SetZ( Float_t v )           { fZ=v; }
  void SetSx( Float_t v )          { fSx=v; }
  void SetSy( Float_t v )          { fSy=v; }
  void SetSz( Float_t v )          { fSz=v; }
  void SetTime( Float_t v )        { fTime=v; }
  void SetISlice( Int_t v )      { fISlice=v; }
  void SetTrackID( Int_t v )     { fTrackID=v; }

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

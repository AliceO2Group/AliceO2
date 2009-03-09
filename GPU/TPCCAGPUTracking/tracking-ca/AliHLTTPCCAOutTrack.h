//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAOUTTRACK_H
#define ALIHLTTPCCAOUTTRACK_H

#include "AliHLTTPCCATrackParam.h"

/**
 * @class AliHLTTPCCAOutTrack
 * AliHLTTPCCAOutTrack class is used to store the final
 * reconstructed tracks which will be then readed
 * by the AliHLTTPCCATrackerComponent
 *
 * The class contains no temporary variables, etc.
 *
 */
class AliHLTTPCCAOutTrack
{
 public:

  AliHLTTPCCAOutTrack():fFirstHitRef(0),fNHits(0),fStartPoint(),fEndPoint(),fOrigTrackID(0){}
  virtual ~AliHLTTPCCAOutTrack(){}

  GPUhd() Int_t NHits()               const { return fNHits; }
  GPUhd() Int_t FirstHitRef()         const { return fFirstHitRef; }

  GPUhd() const AliHLTTPCCATrackParam &StartPoint() const { return fStartPoint; }
  GPUhd() const AliHLTTPCCATrackParam &EndPoint()   const { return fEndPoint; }
  GPUhd() Int_t OrigTrackID()                const { return fOrigTrackID; }

  GPUhd() void SetNHits( Int_t v )               { fNHits = v; }
  GPUhd() void SetFirstHitRef( Int_t v)         { fFirstHitRef = v; }

  GPUhd() void SetStartPoint( const AliHLTTPCCATrackParam &v) { fStartPoint = v; }
  GPUhd() void SetEndPoint( const AliHLTTPCCATrackParam &v)   { fEndPoint = v; }
  GPUhd() void SetOrigTrackID( Int_t v)                { fOrigTrackID = v; }

 protected:
  
  Int_t fFirstHitRef;   //* index of the first hit reference in track->hit reference array
  Int_t fNHits;         //* number of track hits
  AliHLTTPCCATrackParam fStartPoint; //* fitted track parameters at the start point
  AliHLTTPCCATrackParam fEndPoint;   //* fitted track parameters at the start point
  Int_t fOrigTrackID;                //* index of the original slice track

 private:

  void Dummy(); // to make rulechecker happy by having something in .cxx file

  ClassDef(AliHLTTPCCAOutTrack,1)
};


#endif

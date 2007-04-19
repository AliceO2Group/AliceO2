//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAOUTTRACK_H
#define ALIHLTTPCCAOUTTRACK_H

#include "Rtypes.h"
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

  Int_t &NHits()               { return fNHits; }
  Int_t &FirstHitRef()         { return fFirstHitRef; }

  AliHLTTPCCATrackParam &StartPoint() { return fStartPoint; }
  AliHLTTPCCATrackParam &EndPoint()   { return fEndPoint; }
  Int_t &OrigTrackID()                { return fOrigTrackID; }

 protected:
  
  Int_t fFirstHitRef;   //* index of the first hit reference in track->hit reference array
  Int_t fNHits;         //* number of track hits
  AliHLTTPCCATrackParam fStartPoint; //* fitted track parameters at the start point
  AliHLTTPCCATrackParam fEndPoint;   //* fitted track parameters at the start point
  Int_t fOrigTrackID;                //* index of the original slice track

 private:

  void Dummy(); // to make rulechecker happy by having something in .cxx file

  ClassDef(AliHLTTPCCAOutTrack,1);
};


#endif

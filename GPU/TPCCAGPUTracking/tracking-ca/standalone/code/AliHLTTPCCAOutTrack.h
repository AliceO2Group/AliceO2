//-*- Mode: C++ -*-
// @(#) $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

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

    AliHLTTPCCAOutTrack(): fFirstHitRef( 0 ), fNHits( 0 ), fStartPoint(), fEndPoint(), fOrigTrackID( 0 ) {}
    virtual ~AliHLTTPCCAOutTrack() {}

    GPUhd() int NHits()               const { return fNHits; }
    GPUhd() int FirstHitRef()         const { return fFirstHitRef; }

    GPUhd() const AliHLTTPCCATrackParam &StartPoint() const { return fStartPoint; }
    GPUhd() const AliHLTTPCCATrackParam &EndPoint()   const { return fEndPoint; }
    GPUhd() int OrigTrackID()                const { return fOrigTrackID; }

    GPUhd() void SetNHits( int v )               { fNHits = v; }
    GPUhd() void SetFirstHitRef( int v )         { fFirstHitRef = v; }

    GPUhd() void SetStartPoint( const AliHLTTPCCATrackParam &v ) { fStartPoint = v; }
    GPUhd() void SetEndPoint( const AliHLTTPCCATrackParam &v )   { fEndPoint = v; }
    GPUhd() void SetOrigTrackID( int v )                { fOrigTrackID = v; }

  protected:

    int fFirstHitRef;   //* index of the first hit reference in track->hit reference array
    int fNHits;         //* number of track hits
    AliHLTTPCCATrackParam fStartPoint; //* fitted track parameters at the start point
    AliHLTTPCCATrackParam fEndPoint;   //* fitted track parameters at the start point
    int fOrigTrackID;                //* index of the original slice track

  private:

    void Dummy() const; // to make rulechecker happy by having something in .cxx file

    ClassDef( AliHLTTPCCAOutTrack, 1 )
};


#endif //ALIHLTTPCCAOUTTRACK_H

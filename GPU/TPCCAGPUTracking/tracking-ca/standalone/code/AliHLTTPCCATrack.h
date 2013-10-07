//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCATrack.h 51203 2011-08-21 19:48:33Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACK_H
#define ALIHLTTPCCATRACK_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCABaseTrackParam.h"

/**
 * @class ALIHLTTPCCAtrack
 *
 * The class describes the [partially] reconstructed TPC track [candidate].
 * The class is dedicated for internal use by the AliHLTTPCCATracker algorithm.
 * The track parameters at both ends are stored separately in the AliHLTTPCCAEndPoint class
 */
 MEM_CLASS_PRE() class AliHLTTPCCATrack
{
  public:
#if !defined(HLTCA_GPUCODE)
    AliHLTTPCCATrack() : fAlive( 0 ), fFirstHitID( 0 ), fNHits( 0 ), fLocalTrackId( -1 ), fParam() {}
    ~AliHLTTPCCATrack() {}
#endif //!HLTCA_GPUCODE

    GPUhd() bool Alive()               const { return fAlive; }
    GPUhd() int  NHits()               const { return fNHits; }
	GPUhd() int  LocalTrackId()        const { return fLocalTrackId; }
    GPUhd() int  FirstHitID()          const { return fFirstHitID; }
    GPUhd() MakeType(const MEM_LG(AliHLTTPCCABaseTrackParam)&) Param() const { return fParam; }

    GPUhd() void SetAlive( bool v )              { fAlive = v; }
    GPUhd() void SetNHits( int v )               { fNHits = v; }
    GPUhd() void SetLocalTrackId( int v )        { fLocalTrackId = v; }
    GPUhd() void SetFirstHitID( int v )          { fFirstHitID = v; }

    MEM_CLASS_PRE2() GPUhd() void SetParam( const MEM_LG2(AliHLTTPCCABaseTrackParam) &v ) { fParam = v; }

  private:
    bool fAlive;       // flag for mark tracks used by the track merger
    int  fFirstHitID; // index of the first track cell in the track->cell pointer array
    int  fNHits;      // number of track cells
	int  fLocalTrackId; //Id of local track this global track belongs to, index of this track itself if it is a local track
  MEM_LG(AliHLTTPCCABaseTrackParam) fParam; // track parameters

  private:
    //void Dummy(); // to make rulechecker happy by having something in .cxx file

    //ClassDef(AliHLTTPCCATrack,1)
};

#endif //ALIHLTTPCCATRACK_H

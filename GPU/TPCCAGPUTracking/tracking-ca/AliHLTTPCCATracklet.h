//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKLET_H
#define ALIHLTTPCCATRACKLET_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATrackParam.h"

/**
 * @class ALIHLTTPCCATracklet
 *
 * The class describes the reconstructed TPC track candidate.
 * The class is dedicated for internal use by the AliHLTTPCCATracker algorithm.
 */
class AliHLTTPCCATracklet
{
  public:

#if !defined(HLTCA_GPUCODE)
    AliHLTTPCCATracklet() : fStartHitID( 0 ), fNHits( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fParam() {};
    void Dummy() const ;
    ~AliHLTTPCCATracklet() {}
#endif

    GPUhd() int StartHitID()            const { return fStartHitID; }
    GPUhd() int  NHits()                const { return fNHits;      }
    GPUhd() int  FirstRow()             const { return fFirstRow;   }
    GPUhd() int  LastRow()              const { return fLastRow;    }
    GPUhd() const AliHLTTPCCATrackParam &Param() const { return fParam;      }
    GPUhd() int  RowHit( int i )   const { return fRowHits[i];    }

    GPUhd() void SetStartHitID( int v )           { fStartHitID = v; }
    GPUhd() void SetNHits( int v )               {  fNHits = v;      }
    GPUhd() void SetFirstRow( int v )            {  fFirstRow = v;   }
    GPUhd() void SetLastRow( int v )             {  fLastRow = v;    }
    GPUhd() void SetParam( const AliHLTTPCCATrackParam &v ) { fParam = v;      }
    GPUhd() void SetRowHit( int irow, int ih )  { fRowHits[irow] = ih;    }


  private:

    int fStartHitID;            // ID of the starting hit
    int fNHits;                 // N hits
    int fFirstRow;              // first TPC row
    int fLastRow;               // last TPC row
    AliHLTTPCCATrackParam fParam; // tracklet parameters
    int fRowHits[160];          // hit index for each TPC row
};

#endif

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
#include "AliHLTTPCCAGPUConfig.h"

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
    GPUhd() const AliHLTTPCCATrackParam &Param() const { return fParam; }
#ifndef EXTERN_ROW_HITS
    GPUhd() int  RowHit( int i )   const { return fRowHits[i];    }
	GPUhd() int* RowHits()				{ return(fRowHits); }
#endif

    GPUhd() void SetStartHitID( int v )           { fStartHitID = v; }
    GPUhd() void SetNHits( int v )               {  fNHits = v;      }
    GPUhd() void SetFirstRow( int v )            {  fFirstRow = v;   }
    GPUhd() void SetLastRow( int v )             {  fLastRow = v;    }
    GPUhd() void SetParam( const AliHLTTPCCATrackParam &v ) { fParam = v;      }
#ifndef EXTERN_ROW_HITS
    GPUhd() void SetRowHit( int irow, int ih )  { fRowHits[irow] = ih;    }
#endif

#ifndef CUDA_DEVICE_EMULATION
  private:
#endif

    int fStartHitID;            // ID of the starting hit
    int fNHits;                 // N hits
    int fFirstRow;              // first TPC row
    int fLastRow;               // last TPC row
    AliHLTTPCCATrackParam fParam; // tracklet parameters
#ifndef EXTERN_ROW_HITS
    int fRowHits[HLTCA_ROW_COUNT + 1];          // hit index for each TPC row
#endif
};

#endif

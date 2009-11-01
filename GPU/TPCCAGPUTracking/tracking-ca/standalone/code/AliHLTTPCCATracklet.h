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
#include "AliHLTTPCCATrackParam2.h"
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
    AliHLTTPCCATracklet() : fNHits( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fParam() {};
    void Dummy() const ;
    ~AliHLTTPCCATracklet() {}
#endif //!HLTCA_GPUCODE

    GPUhd() int  NHits()                const { return fNHits;      }
    GPUhd() int  FirstRow()             const { return fFirstRow;   }
    GPUhd() int  LastRow()              const { return fLastRow;    }
    GPUhd() const AliHLTTPCCATrackParam2 &Param() const { return fParam; }
#ifndef EXTERN_ROW_HITS
    GPUhd() int  RowHit( int i )   const { return fRowHits[i];    }
	GPUhd() const int* RowHits()	const			{ return(fRowHits); }
    GPUhd() void SetRowHit( int irow, int ih )  { fRowHits[irow] = ih;    }
#endif //EXTERN_ROW_HITS

    GPUhd() void SetNHits( int v )               {  fNHits = v;      }
    GPUhd() void SetFirstRow( int v )            {  fFirstRow = v;   }
    GPUhd() void SetLastRow( int v )             {  fLastRow = v;    }
    GPUhd() void SetParam( const AliHLTTPCCATrackParam2 &v ) { fParam = v;      }

  private:
    int fNHits;                 // N hits
    int fFirstRow;              // first TPC row
    int fLastRow;               // last TPC row
    AliHLTTPCCATrackParam2 fParam; // tracklet parameters
#ifndef EXTERN_ROW_HITS
    int fRowHits[HLTCA_ROW_COUNT + 1];          // hit index for each TPC row
#endif //EXTERN_ROW_HITS
};

#endif //ALIHLTTPCCATRACKLET_H

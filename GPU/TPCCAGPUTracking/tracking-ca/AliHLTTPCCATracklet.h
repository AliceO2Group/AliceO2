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
  AliHLTTPCCATracklet() : fStartHitID(0), fNHits(0), fFirstRow(0), fLastRow(0), fParam(){};
  void Dummy() const ;
  ~AliHLTTPCCATracklet(){}
#endif

  GPUhd() Int_t StartHitID()            const { return fStartHitID; }
  GPUhd() Int_t  NHits()                const { return fNHits;      }
  GPUhd() Int_t  FirstRow()             const { return fFirstRow;   }
  GPUhd() Int_t  LastRow()              const { return fLastRow;    }
  GPUhd() const AliHLTTPCCATrackParam &Param() const { return fParam;      }
  GPUhd() Int_t  RowHit(Int_t i)   const { return fRowHits[i];    }

  GPUhd() void SetStartHitID( Int_t v )           { fStartHitID = v; }
  GPUhd() void SetNHits( Int_t v )               {  fNHits = v;      }
  GPUhd() void SetFirstRow( Int_t v )            {  fFirstRow = v;   }
  GPUhd() void SetLastRow( Int_t v )             {  fLastRow = v;    }
  GPUhd() void SetParam( const AliHLTTPCCATrackParam &v ){ fParam = v;      }
  GPUhd() void SetRowHit( Int_t irow, Int_t ih)  { fRowHits[irow] = ih;    }


private:

  Int_t fStartHitID;            // ID of the starting hit
  Int_t fNHits;                 // N hits
  Int_t fFirstRow;              // first TPC row
  Int_t fLastRow;               // last TPC row
  AliHLTTPCCATrackParam fParam; // tracklet parameters
  Int_t fRowHits[160];          // hit index for each TPC row  
};

#endif

//-*- Mode: C++ -*-

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

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
  ~AliHLTTPCCATracklet(){}
#endif

  GPUhd() Int_t &StartHitID()           { return fStartHitID; }
  GPUhd() Int_t  &NHits()               { return fNHits;      }
  GPUhd() Int_t  &FirstRow()            { return fFirstRow;   }
  GPUhd() Int_t  &LastRow()             { return fLastRow;    }
  GPUhd() AliHLTTPCCATrackParam &Param(){ return fParam;      }
  GPUhd() Int_t  *RowHits()             { return fRowHits;    }

private:

  Int_t fStartHitID;            // ID of the starting hit
  Int_t fNHits;                 // N hits
  Int_t fFirstRow;              // first TPC row
  Int_t fLastRow;               // last TPC row
  AliHLTTPCCATrackParam fParam; // tracklet parameters
  Int_t fRowHits[160];          // hit index for each TPC row  
};

#endif

//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCANEIGHBOURSCLEANER_H
#define ALIHLTTPCCANEIGHBOURSCLEANER_H


#include "AliHLTTPCCADef.h"

class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCANeighboursCleaner
 * 
 */
class AliHLTTPCCANeighboursCleaner
{
 public:
  class AliHLTTPCCASharedMemory
    {
     friend class AliHLTTPCCANeighboursCleaner;
    public:
#if !defined(HLTCA_GPUCODE)
      AliHLTTPCCASharedMemory()
	:fIRow(0),fIRowUp(0),fIRowDn(0),fNRows(0),fNHits(0),fHitLinkDn(0),fHitLinkUp(0),fUpHitLinkDn(0),fDnHitLinkUp(0),fFirstHit(0){}
      AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/) 
	:fIRow(0),fIRowUp(0),fIRowDn(0),fNRows(0),fNHits(0),fHitLinkDn(0),fHitLinkUp(0),fUpHitLinkDn(0),fDnHitLinkUp(0),fFirstHit(0) {}
      AliHLTTPCCASharedMemory& operator=(const AliHLTTPCCASharedMemory& /*dummy*/){ return *this; }
#endif
    protected:
      Int_t fIRow; // current row index
      Int_t fIRowUp; // current row index
      Int_t fIRowDn; // current row index
      Int_t fNRows; // number of rows
      Int_t fNHits; // number of hits
      Short_t *fHitLinkDn; // links to the previous row
      Short_t *fHitLinkUp; // links to the next row
      Short_t *fUpHitLinkDn; // links from next row
      Short_t *fDnHitLinkUp; // links from previous row
      Int_t fFirstHit; // index of the first row hit in global arrays 
    };

  GPUd() static Int_t NThreadSyncPoints(){ return 1; }  

  GPUd() static void Thread( Int_t /*nBlocks*/, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
			     AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );
};


#endif

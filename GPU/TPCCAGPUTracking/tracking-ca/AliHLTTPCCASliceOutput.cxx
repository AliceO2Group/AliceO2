// @(#) $Id$
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "AliHLTTPCCASliceOutput.h"
#include "MemoryAssignmentHelpers.h"


GPUhd() int AliHLTTPCCASliceOutput::EstimateSize( int nOfTracks, int nOfTrackClusters )
{
  // calculate the amount of memory [bytes] needed for the event

  const int kClusterDataSize = sizeof(  int ) + sizeof( unsigned short ) + sizeof( float2 ) + sizeof( float ) + sizeof( UChar_t ) + sizeof( UChar_t );

  return sizeof( AliHLTTPCCASliceOutput ) + sizeof( AliHLTTPCCASliceTrack )*nOfTracks + kClusterDataSize*nOfTrackClusters;
}

#ifndef HLTCA_GPUCODE

template<typename T> inline void AssignNoAlignment( T *&dst, char *&mem, int count )
{
  // assign memory to the pointer dst
  dst = ( T* ) mem;
  mem = ( char * )( dst + count );
}

void AliHLTTPCCASliceOutput::SetPointers(int nTracks, int nTrackClusters, const outputControlStruct* outputControl)
{
  // set all pointers
	if (nTracks == -1) nTracks = fNTracks;
	if (nTrackClusters == -1) nTrackClusters = fNTrackClusters;

  char *mem = fMemory;

  if (outputControl == NULL || outputControl->fDefaultOutput)
  {
	  AssignNoAlignment( fTracks,            mem, nTracks );
	  AssignNoAlignment( fClusterUnpackedYZ, mem, nTrackClusters );
	  AssignNoAlignment( fClusterUnpackedX,  mem, nTrackClusters );
	  AssignNoAlignment( fClusterId,         mem, nTrackClusters );
	  AssignNoAlignment( fClusterRow,        mem, nTrackClusters );
  }

  if (outputControl == NULL || outputControl->fObsoleteOutput)
  {
	  // memory for output tracks
	  AssignMemory( fOutTracks, mem, nTracks );
	  // arrays for track hits
	  AssignMemory( fOutTrackHits, mem, nTrackClusters );
  }
  if ((size_t) (mem - fMemory) + sizeof(AliHLTTPCCASliceOutput) > fMemorySize)
  {
	  fMemorySize = NULL;
	  //printf("\nINTERNAL ERROR IN AliHLTTPCCASliceOutput MEMORY MANAGEMENT req: %d avail: %d\n", (int) ((size_t) (mem - fMemory) + sizeof(AliHLTTPCCASliceOutput)), (int) fMemorySize);
  }
}

void AliHLTTPCCASliceOutput::Allocate(AliHLTTPCCASliceOutput* &ptrOutput, int nTracks, int nTrackHits, outputControlStruct* outputControl)
{
	//Allocate All memory needed for slice output
  const int memsize = (outputControl->fDefaultOutput ? EstimateSize(nTracks, nTrackHits) : sizeof(AliHLTTPCCASliceOutput)) +
	  (outputControl->fObsoleteOutput? (nTracks * sizeof(AliHLTTPCCAOutTrack) + nTrackHits * sizeof(int)) : 0);
  if (outputControl->fOutputPtr)
  {
	  if (outputControl->fOutputMaxSize < memsize)
	  {
                  outputControl->fEndOfSpace = 1;
		  ptrOutput = NULL;
		  return;
	  }
	ptrOutput = (AliHLTTPCCASliceOutput*) outputControl->fOutputPtr;
	outputControl->fOutputPtr += memsize;
	outputControl->fOutputMaxSize -= memsize;
  }
  else
  {
    if (ptrOutput) free(ptrOutput);
	ptrOutput = (AliHLTTPCCASliceOutput*) malloc(memsize);
  }
  ptrOutput->SetMemorySize(memsize);
  ptrOutput->SetPointers(nTracks, nTrackHits, outputControl); // set pointers
}
#endif

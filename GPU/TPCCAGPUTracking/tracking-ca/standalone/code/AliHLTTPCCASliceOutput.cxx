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

void AliHLTTPCCASliceOutput::SetPointers(int nTracks, int nTrackClusters)
{
  // set all pointers
	if (nTracks == -1) nTracks = fNTracks;
	if (nTrackClusters == -1) nTrackClusters == fNTrackClusters;

  char *mem = fMemory;
  AssignNoAlignment( fTracks,            mem, nTracks );
  AssignNoAlignment( fClusterUnpackedYZ, mem, nTrackClusters );
  AssignNoAlignment( fClusterUnpackedX,  mem, nTrackClusters );
  AssignNoAlignment( fClusterId,         mem, nTrackClusters );
  AssignNoAlignment( fClusterPackedYZ,   mem, nTrackClusters );
  AssignNoAlignment( fClusterRow,        mem, nTrackClusters );
  AssignNoAlignment( fClusterPackedAmp,  mem, nTrackClusters );

  // memory for output tracks

  AssignMemory( fOutTracks, mem, nTracks );

  // arrays for track hits

  AssignMemory( fOutTrackHits, mem, nTrackClusters );


  fMemorySize = (mem - fMemory);
}

void AliHLTTPCCASliceOutput::Allocate(AliHLTTPCCASliceOutput* &ptrOutput, int nTracks, int nTrackHits)
{
	//Allocate All memory needed for slice output
  if (ptrOutput) free(ptrOutput);
  ptrOutput = (AliHLTTPCCASliceOutput*) malloc(EstimateSize(nTracks, nTrackHits) + nTracks * sizeof(AliHLTTPCCAOutTrack) + nTrackHits * sizeof(int) + 1024);
  ptrOutput->SetPointers(nTracks, nTrackHits); // set pointers
}
#endif

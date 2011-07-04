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

int AliHLTTPCCASliceOutput::EstimateSize( int nOfTracks, int nOfTrackClusters )
{
  // calculate the amount of memory [bytes] needed for the event

  return sizeof( AliHLTTPCCASliceOutput ) + sizeof( AliHLTTPCCASliceOutTrack )*nOfTracks + sizeof(AliHLTTPCCASliceOutCluster)*nOfTrackClusters;
}

#ifndef HLTCA_GPUCODE

inline void AssignNoAlignment( int &dst, int &size, int count )
{
  // assign memory to the pointer dst
  dst = size;
  size = dst + count ;
}


void AliHLTTPCCASliceOutput::Allocate(AliHLTTPCCASliceOutput* &ptrOutput, int nTracks, int nTrackHits, outputControlStruct* outputControl)
{
	//Allocate All memory needed for slice output
	const int memsize =  EstimateSize(nTracks, nTrackHits);

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
}
#endif

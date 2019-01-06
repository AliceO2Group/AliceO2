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

#include "AliGPUCAOutputControl.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliTPCCommonMath.h"

int AliHLTTPCCASliceOutput::EstimateSize(int nOfTracks, int nOfTrackClusters)
{
	// calculate the amount of memory [bytes] needed for the event
	return sizeof(AliHLTTPCCASliceOutput) + sizeof(AliHLTTPCCASliceOutTrack) * nOfTracks + sizeof(AliHLTTPCCASliceOutCluster) * nOfTrackClusters;
}

#ifndef HLTCA_GPUCODE

inline void AssignNoAlignment(int &dst, int &size, int count)
{
	// assign memory to the pointer dst
	dst = size;
	size = dst + count;
}

void AliHLTTPCCASliceOutput::Allocate(AliHLTTPCCASliceOutput* &ptrOutput, int nTracks, int nTrackHits, AliGPUCAOutputControl *outputControl, void* &internalMemory)
{
	//Allocate All memory needed for slice output
	const size_t memsize = EstimateSize(nTracks, nTrackHits);

	if (outputControl->OutputType != AliGPUCAOutputControl::AllocateInternal)
	{
		if (outputControl->OutputMaxSize - outputControl->Offset < memsize)
		{
			outputControl->EndOfSpace = 1;
			ptrOutput = NULL;
			return;
		}
		ptrOutput = (AliHLTTPCCASliceOutput *) (outputControl->OutputPtr + outputControl->Offset);
		outputControl->Offset += memsize;
	}
	else
	{
		if (internalMemory) free(internalMemory);
		internalMemory = malloc(memsize);
		ptrOutput = (AliHLTTPCCASliceOutput *) internalMemory;
	}
	ptrOutput->SetMemorySize(memsize);
}
#endif

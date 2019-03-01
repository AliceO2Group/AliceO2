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

#include "GPUOutputControl.h"
#include "GPUTPCSliceOutput.h"
#include "GPUCommonMath.h"

unsigned int GPUTPCSliceOutput::EstimateSize(unsigned int nOfTracks, unsigned int nOfTrackClusters)
{
	// calculate the amount of memory [bytes] needed for the event
	return sizeof(GPUTPCSliceOutput) + sizeof(GPUTPCSliceOutTrack) * nOfTracks + sizeof(GPUTPCSliceOutCluster) * nOfTrackClusters;
}

#ifndef GPUCA_GPUCODE
void GPUTPCSliceOutput::Allocate(GPUTPCSliceOutput* &ptrOutput, int nTracks, int nTrackHits, GPUOutputControl *outputControl, void* &internalMemory)
{
	//Allocate All memory needed for slice output
	const size_t memsize = EstimateSize(nTracks, nTrackHits);

	if (outputControl->OutputType != GPUOutputControl::AllocateInternal)
	{
		if (outputControl->OutputMaxSize - outputControl->Offset < memsize)
		{
			outputControl->EndOfSpace = 1;
			ptrOutput = NULL;
			return;
		}
		ptrOutput = (GPUTPCSliceOutput *) (outputControl->OutputPtr + outputControl->Offset);
		outputControl->Offset += memsize;
	}
	else
	{
		if (internalMemory) free(internalMemory);
		internalMemory = malloc(memsize);
		ptrOutput = (GPUTPCSliceOutput *) internalMemory;
	}
	ptrOutput->SetMemorySize(memsize);
}
#endif

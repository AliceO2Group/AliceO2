// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//					David Rohr <drohr@kip.uni-heidelberg.de>				*
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

//If not building GPU Code then build dummy functions to link against
#include "AliHLTTPCCAGPUTracker.h"

AliHLTTPCCAGPUTracker::AliHLTTPCCAGPUTracker() {}
AliHLTTPCCAGPUTracker::~AliHLTTPCCAGPUTracker() {}
int AliHLTTPCCAGPUTracker::InitGPU(int /*sliceCount*/, int /*forceDeviceID*/) { return(1); }
void AliHLTTPCCAGPUTracker::SetDebugLevel(int /*dwLevel*/, std::ostream* /*NewOutFile*/) {}
int AliHLTTPCCAGPUTracker::SetGPUTrackerOption(char* /*OptionName*/, int /*OptionValue*/) {return(1);}
int AliHLTTPCCAGPUTracker::Reconstruct(AliHLTTPCCASliceOutput** /*pTracker*/, AliHLTTPCCAClusterData* /*pClusterData*/, int /*fFirstSlice*/, int /*fSliceCount*/) {return(1);}
int AliHLTTPCCAGPUTracker::ExitGPU() {return(0);}
int AliHLTTPCCAGPUTracker::InitializeSliceParam(int /*iSlice*/, AliHLTTPCCAParam& /*param*/) { return 1; }
void AliHLTTPCCAGPUTracker::SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* /*val*/) {}
unsigned long long int* AliHLTTPCCAGPUTracker::PerfTimer(int /*iSlice*/, unsigned int /*i*/)
{
	//Just return some dummy adress the caller can access
	static unsigned long long int tmp;
	return(&tmp);
}
const AliHLTTPCCASliceOutput::outputControlStruct* AliHLTTPCCAGPUTracker::OutputControl() const { return NULL; }
int AliHLTTPCCAGPUTracker::GetSliceCount() const { return(0); }
int AliHLTTPCCAGPUTracker::RefitMergedTracks(AliHLTTPCGMMerger* /*Merger*/) { return(0); }
char* AliHLTTPCCAGPUTracker::MergerBaseMemory() {return(NULL);}

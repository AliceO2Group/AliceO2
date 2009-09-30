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

int AliHLTTPCCAGPUTracker::InitGPU(int /*sliceCount*/, int /*forceDeviceID*/) {return(1);}
void StandalonePerfTime(int /*iSlice*/, int /*i*/) {}
//template <class T> inline T* AliHLTTPCCAGPUTracker::alignPointer(T* ptr, int alignment) {return(NULL);}
//bool AliHLTTPCCAGPUTracker::CudaFailedMsg(cudaError_t error) {return(true);}
//int AliHLTTPCCAGPUTracker::CUDASync() {return(1);}
void AliHLTTPCCAGPUTracker::SetDebugLevel(int /*dwLevel*/, std::ostream* /*NewOutFile*/) {}
int AliHLTTPCCAGPUTracker::SetGPUTrackerOption(char* /*OptionName*/, int /*OptionValue*/) {return(1);}
int Reconstruct(AliHLTTPCCASliceOutput* /*pTracker*/, AliHLTTPCCAClusterData* /*pClusterData*/, int /*fFirstSlice*/, int /*fSliceCount*/) {return(1);}
int AliHLTTPCCAGPUTracker::ExitGPU() {return(0);}
int AliHLTTPCCAGPUTracker::InitializeSliceParam(int /*iSlice*/, AliHLTTPCCAParam& /*param*/) {}

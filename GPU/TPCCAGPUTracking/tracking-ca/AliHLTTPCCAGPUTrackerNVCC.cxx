// $Id$
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

//  @file   AliHLTTPCCAGPUTrackerNVCC.cxx
//  @author David Rohr, Sergey Gorbunov
//  @date   
//  @brief  TPC CA Tracker for the NVIDIA GPU
//  @note 

#include "AliHLTTPCCAGPUTrackerNVCC.h"

AliHLTTPCCAGPUTrackerNVCC::AliHLTTPCCAGPUTrackerNVCC() :
	fGpuTracker(NULL),
	fGPUMemory(NULL),
	fHostLockedMemory(NULL),
	fDebugLevel(0),
	fOutFile(NULL),
	fGPUMemSize(0),
	fpCudaStreams(NULL),
	fSliceCount(0),
	fOutputControl(NULL),
	fThreadId(0),
	fCudaInitialized(0)
	
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt
};

AliHLTTPCCAGPUTrackerNVCC::~AliHLTTPCCAGPUTrackerNVCC() 
{
  // see header file for class documentation
};

void AliHLTTPCCAGPUTrackerNVCC::ReleaseGlobalLock(void* sem)
{
  // see header file for class documentation
}

int AliHLTTPCCAGPUTrackerNVCC::CheckMemorySizes(int sliceCount)
{
  // see header file for class documentation
  return(0);
}

int AliHLTTPCCAGPUTrackerNVCC::InitGPU(int sliceCount, int forceDeviceID)
{
  // see header file for class documentation
  return(0);
}

template <class T> inline T* AliHLTTPCCAGPUTrackerNVCC::alignPointer(T* ptr, int alignment)
{
  // see header file for class documentation
	return((T*) NULL);
}

bool AliHLTTPCCAGPUTrackerNVCC::CudaFailedMsg(cudaError_t error)
{
  // see header file for class documentation
	return(true);
}

int AliHLTTPCCAGPUTrackerNVCC::CUDASync(char* state)
{
  // see header file for class documentation
	return(0);
}

void AliHLTTPCCAGPUTrackerNVCC::SetDebugLevel(const int dwLevel, std::ostream* const NewOutFile)
{
  // see header file for class documentation
}

int AliHLTTPCCAGPUTrackerNVCC::SetGPUTrackerOption(char* OptionName, int /*OptionValue*/)
{
	return(0);
}

void AliHLTTPCCAGPUTrackerNVCC::StandalonePerfTime(int /*iSlice*/, int /*i*/) 
{
  // see header file for class documentation
}

void AliHLTTPCCAGPUTrackerNVCC::DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check)
{
  // see header file for class documentation
}

int AliHLTTPCCAGPUTrackerNVCC::Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
  // see header file for class documentation
	return(0);
}

int AliHLTTPCCAGPUTrackerNVCC::InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param)
{
  // see header file for class documentation
	return(0);
}

int AliHLTTPCCAGPUTrackerNVCC::ExitGPU()
{
  // see header file for class documentation
	return(0);
}

void AliHLTTPCCAGPUTrackerNVCC::SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val)
{
  // see header file for class documentation
}

int AliHLTTPCCAGPUTrackerNVCC::GetThread()
{
  // see header file for class documentation
    return(0);
}

unsigned long long int* AliHLTTPCCAGPUTrackerNVCC::PerfTimer(int iSlice, unsigned int i)
{
  // see header file for class documentation
    static unsigned long long int tmp;
    return(&tmp);
}

const AliHLTTPCCASliceOutput::outputControlStruct* AliHLTTPCCAGPUTrackerNVCC::OutputControl() const
{
	//Return Pointer to Output Control Structure
	return fOutputControl;
}

int AliHLTTPCCAGPUTrackerNVCC::GetSliceCount() const
{
	//Return max slice count processable
	return(fSliceCount);
}

AliHLTTPCCAGPUTracker* AliHLTTPCCAGPUTrackerNVCCCreate()
{
  // see header file for class documentation
	return new AliHLTTPCCAGPUTrackerNVCC;
} 
void AliHLTTPCCAGPUTrackerNVCCDestroy(AliHLTTPCCAGPUTracker* ptr)
{
  // see header file for class documentation
	delete ptr;
}


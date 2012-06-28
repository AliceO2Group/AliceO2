// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAGPUTRACKER_H
#define ALIHLTTPCCAGPUTRACKER_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCASliceOutput.h"
#include <iostream>

class AliHLTTPCCAClusterData;
class AliHLTTPCCASliceOutput;
class AliHLTTPCCAParam;
class AliHLTTPCGMMerger;

//Abstract Interface for GPU Tracker class
class AliHLTTPCCAGPUTracker
{
public:
	AliHLTTPCCAGPUTracker();
	virtual ~AliHLTTPCCAGPUTracker();

	virtual int InitGPU(int sliceCount = -1, int forceDeviceID = -1);
	virtual int IsInitialized();
	virtual int Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ExitGPU();

	virtual void SetDebugLevel(const int dwLevel, std::ostream* const NewOutFile = NULL);
	virtual int SetGPUTrackerOption(char* OptionName, int OptionValue);

	virtual unsigned long long int* PerfTimer(int iSlice, unsigned int i);

	virtual int InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param);
	virtual void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val);

	virtual const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const;
	virtual int GetSliceCount() const;

	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger);
	virtual char* MergerBaseMemory();

private:
	// disable copy
	AliHLTTPCCAGPUTracker( const AliHLTTPCCAGPUTracker& );
	AliHLTTPCCAGPUTracker &operator=( const AliHLTTPCCAGPUTracker& );
};

#endif //ALIHLTTPCCAGPUTRACKER_H

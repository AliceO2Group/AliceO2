//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCATracker.h 33907 2009-07-23 13:52:49Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKERFRAMEWORK_H
#define ALIHLTTPCCATRACKERFRAMEWORK_H

#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGPUTracker.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTLogging.h"
#include <iostream>
#include <string.h>

class AliHLTTPCCASliceOutput;
class AliHLTTPCCAClusterData;

class AliHLTTPCCATrackerFramework : AliHLTLogging
{
public:
	AliHLTTPCCATrackerFramework(int allowGPU = 1, const char* GPU_Library = NULL, int GPUDeviceNum = -1);
	~AliHLTTPCCATrackerFramework();

	int InitGPU(int sliceCount = 1, int forceDeviceID = -1);
	int ExitGPU();
	void SetGPUDebugLevel(int Level, std::ostream *OutFile = NULL, std::ostream *GPUOutFile = NULL);
	int SetGPUTrackerOption(const char* OptionName, int OptionValue) {if (strcmp(OptionName, "GlobalTracking") == 0) fGlobalTracking = OptionValue;return(fGPUTracker->SetGPUTrackerOption(OptionName, OptionValue));}
	int SetGPUTracker(bool enable);

	int InitializeSliceParam(int iSlice, AliHLTTPCCAParam &param);
	void UpdateGPUSliceParam();

	GPUhd() const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const { return fOutputControl; }
	GPUhd() void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val);

	int ProcessSlices(int firstSlice, int sliceCount, AliHLTTPCCAClusterData* pClusterData, AliHLTTPCCASliceOutput** pOutput);
	double GetTimer(int iSlice, int iTimer);
	void ResetTimer(int iSlice, int iTimer);

	int MaxSliceCount() const { return(fUseGPUTracker ? (fGPUTrackerAvailable ? fGPUTracker->GetSliceCount() : 0) : fCPUSliceCount); }
	int GetGPUStatus() const { return(fGPUTrackerAvailable + fUseGPUTracker); }

	const AliHLTTPCCAParam& Param(int iSlice) const { return(fCPUTrackers[iSlice].Param()); }
	AliHLTTPCCAParam& GetParam(int iSlice) { return(*((AliHLTTPCCAParam*)fCPUTrackers[iSlice].pParam())); }
	const AliHLTTPCCARow& Row(int iSlice, int iRow) const { return(fCPUTrackers[iSlice].Row(iRow)); }  //TODO: Should be changed to return only row parameters

	void SetKeepData(bool v) {fKeepData = v;}

	AliHLTTPCCAGPUTracker* GetGPUTracker() {return(fGPUTracker);}
	AliHLTTPCCATracker& CPUTracker(int iSlice) {return(fCPUTrackers[iSlice]);}

private:
  static const int fgkNSlices = 36;       //* N slices

  char fGPULibAvailable;	//Is the Library with the GPU code available at all?
  char fGPUTrackerAvailable; // Is the GPU Tracker Available?
  char fUseGPUTracker; // use the GPU tracker 
  int fGPUDebugLevel;  // debug level for the GPU code
  AliHLTTPCCAGPUTracker* fGPUTracker;	//Pointer to GPU Tracker Object
  void* fGPULib;		//Pointer to GPU Library

  AliHLTTPCCASliceOutput::outputControlStruct* fOutputControl;

  AliHLTTPCCATracker fCPUTrackers[fgkNSlices];
  static const int fCPUSliceCount = 36;

  char fKeepData;		//Keep temporary data and do not free memory imediately, used for Standalone Debug Event Display
  char fGlobalTracking;	//Use global tracking

  AliHLTTPCCATrackerFramework( const AliHLTTPCCATrackerFramework& );
  AliHLTTPCCATrackerFramework &operator=( const AliHLTTPCCATrackerFramework& );

  ClassDef( AliHLTTPCCATrackerFramework, 0 )

};

#endif //ALIHLTTPCCATRACKERFRAMEWORK_H

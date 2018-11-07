//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCASTANDALONEFRAMEWORK_H
#define ALIHLTTPCCASTANDALONEFRAMEWORK_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCGMMerger.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCCAMCInfo.h"
class AliGPUReconstruction;
#include <iostream>
#include <fstream>

/**
 * @class AliHLTTPCCAStandaloneFramework
 *
 * The class to run the HLT TPC reconstruction (36 CA slice trackers + CA merger )
 * in a stand-alone mode.
 *
 */
class AliHLTTPCCAStandaloneFramework
{
  public:

    AliHLTTPCCAStandaloneFramework();
    ~AliHLTTPCCAStandaloneFramework();

    int Initialize(AliGPUReconstruction* rec);
    void Uninitialize();

    static AliHLTTPCCAStandaloneFramework &Instance();

	const AliGPUCAParam &Param ( int iSlice ) const { return(fTracker->Param(iSlice)); }
	const AliGPUCAParam &Param () const { return(fMerger.SliceParam()); }
	const AliHLTTPCCARow &Row ( int iSlice, int iRow ) const { return(fTracker->Row(iSlice, iRow)); }
    const AliHLTTPCCASliceOutput &Output( int iSlice ) const { return *fSliceOutput[iSlice]; }
    AliHLTTPCGMMerger  &Merger()  { return fMerger; }
    AliHLTTPCCAClusterData &ClusterData( int iSlice ) { return fClusterData[iSlice]; }
    AliHLTTPCCATrackerFramework &Tracker() {return *fTracker;}
    /**
     * prepare for reading of the event
     */
    void StartDataReading( int guessForNumberOfClusters = 256 );

    /**
     *  perform event reconstruction
     */
    int ProcessEvent(int forceSingleSlice = -1, bool resetTimers = true);


    int NSlices() const { return fgkNSlices; }

    double LastTime( int iTimer ) const { return fLastTime[iTimer]; }
    double StatTime( int iTimer ) const { return fStatTime[iTimer]; }
    int StatNEvents() const { return fStatNEvents; }

    void WriteEvent( std::ostream &out ) const;
    int ReadEvent( std::istream &in, bool ResetIds = false, bool addData = false, float shift = 0., float minZ = -1e6, float maxZ = -1e6, bool silent = false, bool doQA = true );

	void SetGPUDebugLevel(int Level, std::ostream *OutFile = NULL, std::ostream *GPUOutFile = NULL) { fDebugLevel = Level; fTracker->SetGPUDebugLevel(Level, OutFile, GPUOutFile); fMerger.SetDebugLevel(Level);}
	int SetGPUTrackerOption(const char* OptionName, int OptionValue) {return(fTracker->SetGPUTrackerOption(OptionName, OptionValue));}
	int SetGPUTracker(bool enable) { return(fTracker->SetGPUTracker(enable)); }
	int GetGPUStatus() const { return(fTracker->GetGPUStatus()); }
	int GetGPUMaxSliceCount() const { return(fTracker->MaxSliceCount()); }
	void SetEventDisplay(int v) {fEventDisplay = v;}
	void SetRunQA(int v) {fRunQA = v;}
	void SetRunMerger(int v) {fRunMerger = v;}
	void SetExternalClusterData(AliHLTTPCCAClusterData* v) {fClusterData = v;}

	int InitializeSliceParam(int iSlice, const AliGPUCAParam* param) { return(fTracker->InitializeSliceParam(iSlice, param)); }
	void SetOutputControl(char* ptr, size_t size) {fOutputControl.fOutputPtr = ptr;fOutputControl.fOutputMaxSize = size;}

	int GetNMCLabels() {return(fMCLabels.size());}
	int GetNMCInfo() {return(fMCInfo.size());}
	const AliHLTTPCClusterMCLabel* GetMCLabels() {return(fMCLabels.data());}
	const AliHLTTPCCAMCInfo* GetMCInfo() {return(fMCInfo.data());}
    void ResetMC() {fMCLabels.clear(); fMCInfo.clear();}

  private:

    static const int fgkNSlices = 36;       //* N slices

    AliHLTTPCCAStandaloneFramework( const AliHLTTPCCAStandaloneFramework& );
    const AliHLTTPCCAStandaloneFramework &operator=( const AliHLTTPCCAStandaloneFramework& ) const;

    AliHLTTPCGMMerger fMerger;  //* global merger
	AliHLTTPCCAClusterData* fClusterData;
    AliHLTTPCCAClusterData fInternalClusterData[fgkNSlices];
	AliHLTTPCCASliceOutput* fSliceOutput[fgkNSlices];
	AliHLTTPCCASliceOutput::outputControlStruct fOutputControl;

	AliHLTTPCCATrackerFramework* fTracker;

    double fLastTime[20]; //* timers
    double fStatTime[20]; //* timers
    int fStatNEvents;    //* n events proceed

	int fDebugLevel;	//Tracker Framework Debug Level
	int fEventDisplay;	//Display event in Standalone Event Display
	int fRunQA;         //Stun Standalone QA
	int fRunMerger;		//Run Track Merger
	std::vector<AliHLTTPCClusterMCLabel> fMCLabels;
	std::vector<AliHLTTPCCAMCInfo> fMCInfo;
};

#endif //ALIHLTTPCCASTANDALONEFRAMEWORK_H

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
#if defined(HLTCA_STANDALONE) & defined(HLTCA_STANDALONE_OLD_MERGER)
#include "AliHLTTPCCAMerger.h"
#define AliHLTTPCGMMerger AliHLTTPCCAMerger
#else
#include "AliHLTTPCGMMerger.h"
#endif
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCCAMCInfo.h"
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
#ifdef HLTCA_STANDALONE
	friend int DrawGLScene(bool DoAnimation);
#endif

  public:

    AliHLTTPCCAStandaloneFramework(int allowGPU = 1, const char* GPULibrary = NULL);
    ~AliHLTTPCCAStandaloneFramework();

    static AliHLTTPCCAStandaloneFramework &Instance(int allowGPU = 1, const char* GPULibrary = NULL);

	const AliHLTTPCCAParam &Param ( int iSlice ) const { return(fTracker.Param(iSlice)); }
	const AliHLTTPCCAParam &Param () const { return(fMerger.SliceParam()); }
	const AliHLTTPCCARow &Row ( int iSlice, int iRow ) const { return(fTracker.Row(iSlice, iRow)); }
    const AliHLTTPCCASliceOutput &Output( int iSlice ) const { return *fSliceOutput[iSlice]; }
    AliHLTTPCGMMerger  &Merger()  { return fMerger; }
    AliHLTTPCCAClusterData &ClusterData( int iSlice ) { return fClusterData[iSlice]; }

    /**
     * prepare for reading of the event
     */
    void StartDataReading( int guessForNumberOfClusters = 256 );

    /**
     *  read next cluster
     */
    void ReadCluster( int id, int iSlice, int iRow, float X, float Y, float Z, float Amp ) {
      fClusterData[iSlice].ReadCluster( id, iRow, X, Y, Z, Amp );
    }

    /**
     * finish reading of the event
     */
    void FinishDataReading();

    /**
     *  perform event reconstruction
     */
    int ProcessEvent(int forceSingleSlice = -1, bool resetTimers = true);


    int NSlices() const { return fgkNSlices; }

    double LastTime( int iTimer ) const { return fLastTime[iTimer]; }
    double StatTime( int iTimer ) const { return fStatTime[iTimer]; }
    int StatNEvents() const { return fStatNEvents; }

    void SetSettings(float solenoidBz, bool homemadeEvents, bool constBz);
    void WriteEvent( std::ostream &out ) const;
    int ReadEvent( std::istream &in, bool ResetIds = false, bool addData = false, float shift = 0., float minZ = -1e6, float maxZ = -1e6, bool silent = false, bool doQA = true );

	int InitGPU(int sliceCount = 1, int forceDeviceID = -1) { return(fTracker.InitGPU(sliceCount, forceDeviceID)); }
	int ExitGPU() { return(fTracker.ExitGPU()); }
	void SetGPUDebugLevel(int Level, std::ostream *OutFile = NULL, std::ostream *GPUOutFile = NULL) { fDebugLevel = Level; fTracker.SetGPUDebugLevel(Level, OutFile, GPUOutFile); fMerger.SetDebugLevel(Level);}
	int SetGPUTrackerOption(const char* OptionName, int OptionValue) {return(fTracker.SetGPUTrackerOption(OptionName, OptionValue));}
	int SetGPUTracker(bool enable) { return(fTracker.SetGPUTracker(enable)); }
	int GetGPUStatus() const { return(fTracker.GetGPUStatus()); }
	int GetGPUMaxSliceCount() const { return(fTracker.MaxSliceCount()); }
	void SetHighQPtForward(float v) { AliHLTTPCCAParam param = fMerger.SliceParam(); param.SetHighQPtForward(v); fMerger.SetSliceParam(param);}
	void SetNWays(int v) { AliHLTTPCCAParam param = fMerger.SliceParam(); param.SetNWays(v); fMerger.SetSliceParam(param);}
	void SetNWaysOuter(bool v) { AliHLTTPCCAParam param = fMerger.SliceParam(); param.SetNWaysOuter(v); fMerger.SetSliceParam(param);}
	void SetSearchWindowDZDR(float v) { AliHLTTPCCAParam param = fMerger.SliceParam(); param.SetSearchWindowDZDR(v); fMerger.SetSliceParam(param);for (int i = 0;i < fgkNSlices;i++) fTracker.GetParam(i).SetSearchWindowDZDR(v);}
	void SetContinuousTracking(bool v) { AliHLTTPCCAParam param = fMerger.SliceParam(); param.SetContinuousTracking(v); fMerger.SetSliceParam(param);for (int i = 0;i < fgkNSlices;i++) fTracker.GetParam(i).SetContinuousTracking(v);}
	void SetTrackReferenceX(float v) { AliHLTTPCCAParam param = fMerger.SliceParam(); param.SetTrackReferenceX(v); fMerger.SetSliceParam(param);}
	void UpdateGPUSliceParam() {fTracker.UpdateGPUSliceParam();}
	void SetEventDisplay(int v) {fEventDisplay = v;}
	void SetRunQA(int v) {fRunQA = v;}
	void SetRunMerger(int v) {fRunMerger = v;}
	void SetExternalClusterData(AliHLTTPCCAClusterData* v) {fClusterData = v;}

	int InitializeSliceParam(int iSlice, AliHLTTPCCAParam& param) { return(fTracker.InitializeSliceParam(iSlice, param)); }
	void SetOutputControl(char* ptr, size_t size) {fOutputControl.fOutputPtr = ptr;fOutputControl.fOutputMaxSize = size;}
	
	int GetNMCLabels() {return(fMCLabels.size());}
	int GetNMCInfo() {return(fMCInfo.size());}
	const AliHLTTPCClusterMCLabel* GetMCLabels() {return(fMCLabels.data());}
	const AliHLTTPCCAMCInfo* GetMCInfo() {return(fMCInfo.data());}

  private:

    static const int fgkNSlices = 36;       //* N slices

    AliHLTTPCCAStandaloneFramework( const AliHLTTPCCAStandaloneFramework& );
    const AliHLTTPCCAStandaloneFramework &operator=( const AliHLTTPCCAStandaloneFramework& ) const;

    AliHLTTPCGMMerger fMerger;  //* global merger
	AliHLTTPCCAClusterData* fClusterData;
    AliHLTTPCCAClusterData fInternalClusterData[fgkNSlices];
	AliHLTTPCCASliceOutput* fSliceOutput[fgkNSlices];
	AliHLTTPCCASliceOutput::outputControlStruct fOutputControl;

	AliHLTTPCCATrackerFramework fTracker;

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

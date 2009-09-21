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
#include "AliHLTTPCCAMerger.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCATrackerFramework.h"
#include <iostream>
#include <fstream>
#include "TStopwatch.h"

/**
 * @class AliHLTTPCCAStandaloneFramework
 *
 * The class to run the HLT TPC reconstruction (36 CA slice trackers + CA merger )
 * in a stand-alone mode.
 * Used by AliTPCtrackerCA, the CA event display, CA performance.
 *
 */
class AliHLTTPCCAStandaloneFramework
{

  public:

    AliHLTTPCCAStandaloneFramework();
    ~AliHLTTPCCAStandaloneFramework();

    static AliHLTTPCCAStandaloneFramework &Instance();

	const AliHLTTPCCAParam &Param ( int iSlice ) const { return(fTracker.Param(iSlice)); }
	const AliHLTTPCCARow &Row ( int iSlice, int iRow ) const { return(fTracker.Row(iSlice, iRow)); }
    AliHLTTPCCASliceOutput &Output( int iSlice )  { return fSliceOutput[iSlice]; }
    AliHLTTPCCAMerger  &Merger()  { return fMerger; }
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
    void ProcessEvent(int forceSingleSlice = -1);


    int NSlices() const { return fgkNSlices; }

    double LastTime( int iTimer ) const { return fLastTime[iTimer]; }
    double StatTime( int iTimer ) const { return fStatTime[iTimer]; }
    int StatNEvents() const { return fStatNEvents; }

    void WriteSettings( std::ostream &out ) const;
    void WriteEvent( std::ostream &out ) const;
    void WriteTracks( std::ostream &out ) const;

    void ReadSettings( std::istream &in );
    void ReadEvent( std::istream &in );
    void ReadTracks( std::istream &in );

	int InitGPU(int sliceCount = 1, int forceDeviceID = -1) { return(fTracker.InitGPU(sliceCount, forceDeviceID)); }
	int ExitGPU() { return(fTracker.ExitGPU()); }
	void SetGPUDebugLevel(int Level, std::ostream *OutFile = NULL, std::ostream *GPUOutFile = NULL) { fDebugLevel = Level; fTracker.SetGPUDebugLevel(Level, OutFile, GPUOutFile); }
	int SetGPUTrackerOption(char* OptionName, int OptionValue) {return(fTracker.SetGPUTrackerOption(OptionName, OptionValue));}
	int SetGPUTracker(bool enable) { return(fTracker.SetGPUTracker(enable)); }

	int InitializeSliceParam(int iSlice, AliHLTTPCCAParam& param) { return(fTracker.InitializeSliceParam(iSlice, param)); }

#ifdef HLTCA_STANDALONE
	static void StandaloneQueryTime(unsigned long long int *i)
	{
	#ifdef R__WIN32
		  QueryPerformanceCounter((LARGE_INTEGER*) i);
	#else
		  timespec t;
		  clock_gettime(CLOCK_REALTIME, &t);
		  *i = (unsigned long long int) t.tv_sec * (unsigned long long int) 1000000000 + (unsigned long long int) t.tv_nsec;
	#endif
	}

	static void StandaloneQueryFreq(unsigned long long int *i)
	{
	#ifdef R__WIN32
		  QueryPerformanceFrequency((LARGE_INTEGER*) i);
	#else
		*i = 1000000000;
	#endif
	}
#endif

  private:

    static const int fgkNSlices = 36;       //* N slices

    AliHLTTPCCAStandaloneFramework( const AliHLTTPCCAStandaloneFramework& );
    const AliHLTTPCCAStandaloneFramework &operator=( const AliHLTTPCCAStandaloneFramework& ) const;

    AliHLTTPCCAMerger fMerger;  //* global merger
    AliHLTTPCCAClusterData fClusterData[fgkNSlices];
	AliHLTTPCCASliceOutput fSliceOutput[fgkNSlices];

	AliHLTTPCCATrackerFramework fTracker;

    double fLastTime[20]; //* timers
    double fStatTime[20]; //* timers
    int fStatNEvents;    //* n events proceed

	int fDebugLevel;
};

#endif

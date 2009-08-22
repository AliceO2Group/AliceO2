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
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAMerger.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAGPUTracker.h"
#include <iostream>
#include <fstream>

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

    AliHLTTPCCATracker &SliceTracker( int iSlice )  { return fSliceTrackers[iSlice]; }
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

	int InitGPU(int sliceCount = 1, int forceDeviceID = -1);
	int ExitGPU();
	void SetGPUDebugLevel(int Level, std::ostream *OutFile = NULL, std::ostream *GPUOutFile = NULL);
	int SetGPUTrackerOption(char* OptionName, int OptionValue) {return(fGPUTracker.SetGPUTrackerOption(OptionName, OptionValue));}

  private:

    static const int fgkNSlices = 36;       //* N slices

    AliHLTTPCCAStandaloneFramework( const AliHLTTPCCAStandaloneFramework& );
    const AliHLTTPCCAStandaloneFramework &operator=( const AliHLTTPCCAStandaloneFramework& ) const;

    AliHLTTPCCATracker fSliceTrackers[fgkNSlices]; //* array of slice trackers
    AliHLTTPCCAMerger fMerger;  //* global merger
    AliHLTTPCCAClusterData fClusterData[fgkNSlices];

	AliHLTTPCCAGPUTracker fGPUTracker;

    double fLastTime[20]; //* timers
    double fStatTime[20]; //* timers
    int fStatNEvents;    //* n events proceed

  bool fUseGPUTracker; // use the GPU tracker 
  int fGPUDebugLevel;  // debug level for the GPU code
  int fGPUSliceCount;	//How many slices to process parallel
};

#endif

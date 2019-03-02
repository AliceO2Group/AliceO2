//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCASLICEOUTPUT_H
#define ALIHLTTPCCASLICEOUTPUT_H

#include "AliHLTTPCCADef.h"
#include <cstdlib>
#include "AliHLTTPCCASliceOutTrack.h"


/**
 * @class AliHLTTPCCASliceOutput
 *
 * AliHLTTPCCASliceOutput class is used to store the output of AliHLTTPCCATracker{Component}
 * and transport the output to AliHLTTPCCAGBMerger{Component}
 *
 * The class contains all the necessary information about TPC tracks, reconstructed in one slice.
 * This includes the reconstructed track parameters and some compressed information
 * about the assigned clusters: clusterId, position and amplitude.
 *
 */
class AliHLTTPCCASliceOutput
{
  public:

  struct outputControlStruct
  {
    outputControlStruct() :  fOutputPtr( NULL ), fOutputMaxSize ( 0 ), fEndOfSpace(0) {}
    char* fOutputPtr;		//Pointer to Output Space, NULL to allocate output space
    int fOutputMaxSize;		//Max Size of Output Data if Pointer to output space is given
    bool fEndOfSpace; // end of space flag 
  };

  GPUhd() int NTracks()                    const { return fNTracks;              }
  GPUhd() int NTrackClusters()             const { return fNTrackClusters;       }  
  GPUhd() const AliHLTTPCCASliceOutTrack *GetFirstTrack() const { return fMemory; }
  GPUhd() AliHLTTPCCASliceOutTrack *FirstTrack(){ return fMemory; }
  GPUhd() size_t Size() const { return(fMemorySize); }

  GPUhd() static int EstimateSize( int nOfTracks, int nOfTrackClusters );
  static void Allocate(AliHLTTPCCASliceOutput* &ptrOutput, int nTracks, int nTrackHits, outputControlStruct* outputControl);

  GPUhd() void SetNTracks       ( int v )  { fNTracks = v;        }
  GPUhd() void SetNTrackClusters( int v )  { fNTrackClusters = v; }

  private:

  AliHLTTPCCASliceOutput()
    : fNTracks( 0 ), fNTrackClusters( 0 ), fMemorySize( 0 ){}
  
  ~AliHLTTPCCASliceOutput() {}
  const AliHLTTPCCASliceOutput& operator=( const AliHLTTPCCASliceOutput& ) const { return *this; }
  AliHLTTPCCASliceOutput( const AliHLTTPCCASliceOutput& );

  GPUh() void SetMemorySize(size_t val) { fMemorySize = val; }

  int fNTracks;                   // number of reconstructed tracks
  int fNTrackClusters;            // total number of track clusters
  size_t fMemorySize;	       	// Amount of memory really used

  //Must be last element of this class, user has to make sure to allocate anough memory consecutive to class memory!
  //This way the whole Slice Output is one consecutive Memory Segment

  AliHLTTPCCASliceOutTrack fMemory[0]; // the memory where the pointers above point into

};

#endif 

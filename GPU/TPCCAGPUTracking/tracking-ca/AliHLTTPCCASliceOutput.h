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

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)

#include <cstdlib>
#ifndef HLTCA_GPUCODE
#include "AliHLTTPCCASliceOutTrack.h"
#else
class AliHLTTPCCASliceOutTrack;
#endif
#else
#define NULL 0
#endif


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
    char* volatile fOutputPtr;		//Pointer to Output Space, NULL to allocate output space
    volatile int fOutputMaxSize;		//Max Size of Output Data if Pointer to output space is given
    bool fEndOfSpace; // end of space flag 
  };

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
  GPUhd() int NTracks()                    const { return fNTracks;              }
  GPUhd() int NLocalTracks()               const { return fNLocalTracks;         }
  GPUhd() int NTrackClusters()             const { return fNTrackClusters;       }  
#ifndef HLTCA_GPUCODE
  GPUhd() const AliHLTTPCCASliceOutTrack *GetFirstTrack() const { return fMemory; }
  GPUhd() AliHLTTPCCASliceOutTrack *FirstTrack(){ return fMemory; }
#endif
  GPUhd() size_t Size() const { return(fMemorySize); }

  static int EstimateSize( int nOfTracks, int nOfTrackClusters );
  static void Allocate(AliHLTTPCCASliceOutput* &ptrOutput, int nTracks, int nTrackHits, outputControlStruct* outputControl);

  GPUhd() void SetNTracks       ( int v )  { fNTracks = v;        }
  GPUhd() void SetNLocalTracks  ( int v )  { fNLocalTracks = v;   }
  GPUhd() void SetNTrackClusters( int v )  { fNTrackClusters = v; }

  private:

  AliHLTTPCCASliceOutput()
    : fNTracks( 0 ), fNLocalTracks( 0 ), fNTrackClusters( 0 ), fMemorySize( 0 ){}
  
  ~AliHLTTPCCASliceOutput() {}
  AliHLTTPCCASliceOutput( const AliHLTTPCCASliceOutput& );
  AliHLTTPCCASliceOutput& operator=( const AliHLTTPCCASliceOutput& ) { return *this; }

  GPUh() void SetMemorySize(size_t val) { fMemorySize = val; }

  int fNTracks;                   // number of reconstructed tracks
  int fNLocalTracks;
  int fNTrackClusters;            // total number of track clusters
  size_t fMemorySize;	       	// Amount of memory really used

  //Must be last element of this class, user has to make sure to allocate anough memory consecutive to class memory!
  //This way the whole Slice Output is one consecutive Memory Segment

#ifndef HLTCA_GPUCODE
  AliHLTTPCCASliceOutTrack fMemory[0]; // the memory where the pointers above point into
#endif
#endif

};
#endif 

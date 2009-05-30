//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCAMERGEROUTPUT_H
#define ALIHLTTPCCAMERGEROUTPUT_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAMergedTrack.h"

/**
 * @class AliHLTTPCCAMergerOutput
 *
 * AliHLTTPCCAMergerOutput class is used to store the output of AliHLTTPCCATracker{Component}
 * and transport the output to AliHLTTPCCAMerger{Component}
 *
 * The class contains all the necessary information about TPC tracks, reconstructed in one slice.
 * This includes the reconstructed track parameters and some compressed information
 * about the assigned clusters: clusterId, position and amplitude.
 *
 */
class AliHLTTPCCAMergerOutput
{
  public:

    AliHLTTPCCAMergerOutput()
        : fNTracks( 0 ), fNTrackClusters( 0 ), fTracks( 0 ), fClusterId( 0 ), fClusterPackedAmp( 0 ) {}

    ~AliHLTTPCCAMergerOutput() {}


    GPUhd() int NTracks()                    const { return fNTracks;              }
    GPUhd() int NTrackClusters()             const { return fNTrackClusters;       }

    GPUhd() const AliHLTTPCCAMergedTrack &Track( int i ) const { return fTracks[i]; }
    GPUhd()  int   ClusterId     ( int i )  const { return fClusterId[i]; }
    GPUhd() UChar_t  ClusterPackedAmp( int i )  const { return fClusterPackedAmp[i]; }

    GPUhd() static int EstimateSize( int nOfTracks, int nOfTrackClusters );
    GPUhd() void SetPointers();

    GPUhd() void SetNTracks       ( int v )  { fNTracks = v;        }
    GPUhd() void SetNTrackClusters( int v )  { fNTrackClusters = v; }

    GPUhd() void SetTrack( int i, const AliHLTTPCCAMergedTrack &v ) {  fTracks[i] = v; }
    GPUhd() void SetClusterId( int i,  int v ) {  fClusterId[i] = v; }
    GPUhd() void SetClusterPackedAmp( int i, UChar_t v ) {  fClusterPackedAmp[i] = v; }

  private:

    AliHLTTPCCAMergerOutput( const AliHLTTPCCAMergerOutput & )
        : fNTracks( 0 ), fNTrackClusters( 0 ), fTracks( 0 ), fClusterId( 0 ), fClusterPackedAmp( 0 ) {}

    const AliHLTTPCCAMergerOutput& operator=( const AliHLTTPCCAMergerOutput &/*v*/ ) const {
      return *this;
    }

    int fNTracks;                 // number of reconstructed tracks
    int fNTrackClusters;          // total number of track clusters
    AliHLTTPCCAMergedTrack *fTracks; // pointer to reconstructed tracks
    int   *fClusterId;         // pointer to cluster IDs ( packed slice, patch, cluster )
    UChar_t  *fClusterPackedAmp;    // pointer to packed cluster amplitudes
};



GPUhd() inline int AliHLTTPCCAMergerOutput::EstimateSize( int nOfTracks, int nOfTrackClusters )
{
  // calculate the amount of memory [bytes] needed for the event

  const int kClusterDataSize = sizeof( int ) + sizeof( UChar_t );

  return sizeof( AliHLTTPCCAMergerOutput ) + sizeof( AliHLTTPCCAMergedTrack )*nOfTracks + kClusterDataSize*nOfTrackClusters;
}


GPUhd() inline void AliHLTTPCCAMergerOutput::SetPointers()
{
  // set all pointers

  fTracks            = ( AliHLTTPCCAMergedTrack* )( ( &fClusterPackedAmp ) + 1 );
  fClusterId         = ( int* )  ( fTracks            + fNTracks );
  fClusterPackedAmp  = ( UChar_t* ) ( fClusterId + fNTrackClusters );
}

#endif

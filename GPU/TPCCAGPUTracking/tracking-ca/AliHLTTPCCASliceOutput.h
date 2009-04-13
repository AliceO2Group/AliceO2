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

#include "AliHLTTPCCASliceTrack.h"

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

    GPUhd() int NTracks()                    const { return fNTracks;              }
    GPUhd() int NTrackClusters()             const { return fNTrackClusters;       }

    GPUhd() const AliHLTTPCCASliceTrack &Track( int i ) const { return fTracks[i]; }
    GPUhd() unsigned int   ClusterIDrc     ( int i )  const { return fClusterIDrc[i]; }
    GPUhd() unsigned short ClusterPackedYZ ( int i )  const { return fClusterPackedYZ[i]; }
    GPUhd() UChar_t  ClusterPackedAmp( int i )  const { return fClusterPackedAmp[i]; }
    GPUhd() float2   ClusterUnpackedYZ ( int i )  const { return fClusterUnpackedYZ[i]; }
    GPUhd() float    ClusterUnpackedX  ( int i )  const { return fClusterUnpackedX[i]; }

    GPUhd() static int EstimateSize( int nOfTracks, int nOfTrackClusters );
    GPUhd() void SetPointers();

    GPUhd() void SetNTracks       ( int v )  { fNTracks = v;        }
    GPUhd() void SetNTrackClusters( int v )  { fNTrackClusters = v; }

    GPUhd() void SetTrack( int i, const AliHLTTPCCASliceTrack &v ) {  fTracks[i] = v; }
    GPUhd() void SetClusterIDrc( int i, unsigned int v ) {  fClusterIDrc[i] = v; }
    GPUhd() void SetClusterPackedYZ( int i, unsigned short v ) {  fClusterPackedYZ[i] = v; }
    GPUhd() void SetClusterPackedAmp( int i, UChar_t v ) {  fClusterPackedAmp[i] = v; }
    GPUhd() void SetClusterUnpackedYZ( int i, float2 v ) {  fClusterUnpackedYZ[i] = v; }
    GPUhd() void SetClusterUnpackedX( int i, float v ) {  fClusterUnpackedX[i] = v; }

  private:

    AliHLTTPCCASliceOutput( const AliHLTTPCCASliceOutput& )
        : fNTracks( 0 ), fNTrackClusters( 0 ), fTracks( 0 ), fClusterIDrc( 0 ), fClusterPackedYZ( 0 ), fClusterUnpackedYZ( 0 ), fClusterUnpackedX( 0 ), fClusterPackedAmp( 0 ) {}

    const AliHLTTPCCASliceOutput& operator=( const AliHLTTPCCASliceOutput& ) const { return *this; }

    int fNTracks;                 // number of reconstructed tracks
    int fNTrackClusters;          // total number of track clusters
    AliHLTTPCCASliceTrack *fTracks; // pointer to reconstructed tracks
    unsigned int   *fClusterIDrc;         // pointer to cluster IDs ( packed IRow and ICluster)
    unsigned short *fClusterPackedYZ;     // pointer to packed cluster YZ coordinates
    float2   *fClusterUnpackedYZ;   // pointer to cluster coordinates (temporary data, for debug proposes)
    float    *fClusterUnpackedX;   // pointer to cluster coordinates (temporary data, for debug proposes)
    UChar_t  *fClusterPackedAmp;    // pointer to packed cluster amplitudes

};



GPUhd() inline int AliHLTTPCCASliceOutput::EstimateSize( int nOfTracks, int nOfTrackClusters )
{
  // calculate the amount of memory [bytes] needed for the event

  const int kClusterDataSize = sizeof( unsigned int ) + sizeof( unsigned short ) + sizeof( float2 ) + sizeof( float ) + sizeof( UChar_t );

  return sizeof( AliHLTTPCCASliceOutput ) + sizeof( AliHLTTPCCASliceTrack )*nOfTracks + kClusterDataSize*nOfTrackClusters;
}


GPUhd() inline void AliHLTTPCCASliceOutput::SetPointers()
{
  // set all pointers

  fTracks            = ( AliHLTTPCCASliceTrack* )( ( &fClusterPackedAmp ) + 1 );
  fClusterUnpackedYZ = ( float2* )  ( fTracks   + fNTracks );
  fClusterUnpackedX  = ( float* )   ( fClusterUnpackedYZ + fNTrackClusters );
  fClusterIDrc       = ( unsigned int* )  ( fClusterUnpackedX  + fNTrackClusters );
  fClusterPackedYZ   = ( unsigned short* )( fClusterIDrc       + fNTrackClusters );
  fClusterPackedAmp  = ( UChar_t* ) ( fClusterPackedYZ + fNTrackClusters );
}

#endif

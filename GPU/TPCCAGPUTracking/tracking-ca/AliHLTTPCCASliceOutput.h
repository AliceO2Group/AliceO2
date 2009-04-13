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

  GPUhd() Int_t NTracks()                    const { return fNTracks;              }
  GPUhd() Int_t NTrackClusters()             const { return fNTrackClusters;       }

  GPUhd() const AliHLTTPCCASliceTrack &Track( Int_t i ) const { return fTracks[i]; }
  GPUhd() UInt_t   ClusterIDrc     ( Int_t i )  const { return fClusterIDrc[i]; }
  GPUhd() UShort_t ClusterPackedYZ ( Int_t i )  const { return fClusterPackedYZ[i]; }
  GPUhd() UChar_t  ClusterPackedAmp( Int_t i )  const { return fClusterPackedAmp[i]; }
  GPUhd() float2   ClusterUnpackedYZ ( Int_t i )  const { return fClusterUnpackedYZ[i]; }
  GPUhd() float    ClusterUnpackedX  ( Int_t i )  const { return fClusterUnpackedX[i]; }

  GPUhd() static Int_t EstimateSize( Int_t nOfTracks, Int_t nOfTrackClusters );
  GPUhd() void SetPointers();

  GPUhd() void SetNTracks       ( Int_t v )  { fNTracks = v;        }
  GPUhd() void SetNTrackClusters( Int_t v )  { fNTrackClusters = v; }

  GPUhd() void SetTrack( Int_t i, const AliHLTTPCCASliceTrack &v ) {  fTracks[i] = v; }
  GPUhd() void SetClusterIDrc( Int_t i, UInt_t v ) {  fClusterIDrc[i] = v; }
  GPUhd() void SetClusterPackedYZ( Int_t i, UShort_t v ) {  fClusterPackedYZ[i] = v; }
  GPUhd() void SetClusterPackedAmp( Int_t i, UChar_t v ) {  fClusterPackedAmp[i] = v; }
  GPUhd() void SetClusterUnpackedYZ( Int_t i, float2 v ) {  fClusterUnpackedYZ[i] = v; }
  GPUhd() void SetClusterUnpackedX( Int_t i, float v ) {  fClusterUnpackedX[i] = v; }

 private:

  AliHLTTPCCASliceOutput( const AliHLTTPCCASliceOutput& )
    : fNTracks(0),fNTrackClusters(0),fTracks(0),fClusterIDrc(0), fClusterPackedYZ(0),fClusterUnpackedYZ(0),fClusterUnpackedX(0),fClusterPackedAmp(0){}

  const AliHLTTPCCASliceOutput& operator=( const AliHLTTPCCASliceOutput& ) const { return *this; }

  Int_t fNTracks;                 // number of reconstructed tracks
  Int_t fNTrackClusters;          // total number of track clusters
  AliHLTTPCCASliceTrack *fTracks; // pointer to reconstructed tracks
  UInt_t   *fClusterIDrc;         // pointer to cluster IDs ( packed IRow and ICluster)
  UShort_t *fClusterPackedYZ;     // pointer to packed cluster YZ coordinates 
  float2   *fClusterUnpackedYZ;   // pointer to cluster coordinates (temporary data, for debug proposes)
  float    *fClusterUnpackedX;   // pointer to cluster coordinates (temporary data, for debug proposes)
  UChar_t  *fClusterPackedAmp;    // pointer to packed cluster amplitudes

};



GPUhd() inline Int_t AliHLTTPCCASliceOutput::EstimateSize( Int_t nOfTracks, Int_t nOfTrackClusters )
{
  // calculate the amount of memory [bytes] needed for the event

  const Int_t kClusterDataSize = sizeof(UInt_t) + sizeof(UShort_t) + sizeof(float2) + sizeof(float)+ sizeof(UChar_t);

  return sizeof(AliHLTTPCCASliceOutput) + sizeof(AliHLTTPCCASliceTrack)*nOfTracks + kClusterDataSize*nOfTrackClusters;
}


GPUhd() inline void AliHLTTPCCASliceOutput::SetPointers()
{
  // set all pointers

  fTracks            = (AliHLTTPCCASliceTrack*)((&fClusterPackedAmp)+1);
  fClusterUnpackedYZ = (float2*)  ( fTracks   + fNTracks );
  fClusterUnpackedX  = (float*)   ( fClusterUnpackedYZ + fNTrackClusters );
  fClusterIDrc       = (UInt_t*)  ( fClusterUnpackedX  + fNTrackClusters );
  fClusterPackedYZ   = (UShort_t*)( fClusterIDrc       + fNTrackClusters );
  fClusterPackedAmp  = (UChar_t*) ( fClusterPackedYZ + fNTrackClusters );
}

#endif

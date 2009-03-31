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
    :fNTracks(0), fNTrackClusters(0),fTracks(0),fClusterIDsrc(0),fClusterPackedAmp(0) {}

  AliHLTTPCCAMergerOutput( const AliHLTTPCCAMergerOutput & )
    :fNTracks(0), fNTrackClusters(0),fTracks(0),fClusterIDsrc(0),fClusterPackedAmp(0) {}
  
  const AliHLTTPCCAMergerOutput& operator=( const AliHLTTPCCAMergerOutput &/*v*/ ) const { 
    return *this; 
  }

  ~AliHLTTPCCAMergerOutput(){}


  GPUhd() Int_t NTracks()                    const { return fNTracks;              }
  GPUhd() Int_t NTrackClusters()             const { return fNTrackClusters;       }

  GPUhd() const AliHLTTPCCAMergedTrack &Track( Int_t i ) const { return fTracks[i]; }
  GPUhd() UInt_t   ClusterIDsrc     ( Int_t i )  const { return fClusterIDsrc[i]; }
  GPUhd() UChar_t  ClusterPackedAmp( Int_t i )  const { return fClusterPackedAmp[i]; }

  GPUhd() static Int_t EstimateSize( Int_t nOfTracks, Int_t nOfTrackClusters );
  GPUhd() void SetPointers();

  GPUhd() void SetNTracks       ( Int_t v )  { fNTracks = v;        }
  GPUhd() void SetNTrackClusters( Int_t v )  { fNTrackClusters = v; }

  GPUhd() void SetTrack( Int_t i, const AliHLTTPCCAMergedTrack &v ) {  fTracks[i] = v; }
  GPUhd() void SetClusterIDsrc( Int_t i, UInt_t v ) {  fClusterIDsrc[i] = v; }
  GPUhd() void SetClusterPackedAmp( Int_t i, UChar_t v ) {  fClusterPackedAmp[i] = v; }

 private:
  
  Int_t fNTracks;                 // number of reconstructed tracks
  Int_t fNTrackClusters;          // total number of track clusters
  AliHLTTPCCAMergedTrack *fTracks; // pointer to reconstructed tracks
  UInt_t   *fClusterIDsrc;         // pointer to cluster IDs ( packed IRow and ICluster)
  UChar_t  *fClusterPackedAmp;    // pointer to packed cluster amplitudes

};



GPUhd() inline Int_t AliHLTTPCCAMergerOutput::EstimateSize( Int_t nOfTracks, Int_t nOfTrackClusters )
{
  // calculate the amount of memory [bytes] needed for the event

  const Int_t kClusterDataSize = sizeof(UInt_t) + sizeof(UChar_t);

  return sizeof(AliHLTTPCCAMergerOutput) + sizeof(AliHLTTPCCAMergedTrack)*nOfTracks + kClusterDataSize*nOfTrackClusters;
}


GPUhd() inline void AliHLTTPCCAMergerOutput::SetPointers()
{
  // set all pointers

  fTracks            = (AliHLTTPCCAMergedTrack*)((&fClusterPackedAmp)+1);
  fClusterIDsrc      = (UInt_t*)  ( fTracks            + fNTracks );
  fClusterPackedAmp  = (UChar_t*) ( fClusterIDsrc + fNTrackClusters );
}

#endif

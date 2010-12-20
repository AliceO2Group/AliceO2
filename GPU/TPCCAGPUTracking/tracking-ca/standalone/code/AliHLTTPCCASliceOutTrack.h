//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCASLICEOUTTRACK_H
#define ALIHLTTPCCASLICEOUTTRACK_H

#include "AliHLTTPCCABaseTrackParam.h"
#include "AliHLTTPCCASliceOutCluster.h"

/**
 * @class AliHLTTPCCASliceOutTrack
 * AliHLTTPCCASliceOutTrack class is used to store TPC tracks,
 * which are reconstructed by the TPCCATracker slice tracker.
 *
 * The class contains:
 * - fitted track parameters at its first row, the covariance matrix, \Chi^2, NDF (number of degrees of freedom )
 * - n of clusters assigned to the track
 * - clusters in corresponding cluster arrays
 *
 * The class is used to transport the data between AliHLTTPCCATracker{Component} and AliHLTTPCCAGBMerger{Component}
 *
 */
class AliHLTTPCCASliceOutTrack
{
  public:

    GPUhd() int NClusters()                    const { return fNClusters;       }
    GPUhd() const AliHLTTPCCABaseTrackParam &Param() const { return fParam;           }
    GPUhd() const AliHLTTPCCASliceOutCluster &Cluster( int i ) const { return fClusters[i];           }
    GPUhd() const AliHLTTPCCASliceOutCluster* Clusters() const { return fClusters;           }

    GPUhd() void SetNClusters( int v )                   { fNClusters = v;       }
    GPUhd() void SetParam( const AliHLTTPCCABaseTrackParam &v ) { fParam = v;           }
    GPUhd() void SetCluster( int i, const AliHLTTPCCASliceOutCluster &v ) { fClusters[i] = v;           }
    
    GPUhd() static int GetSize( int nClust )  { return sizeof(AliHLTTPCCASliceOutTrack)+nClust*sizeof(AliHLTTPCCASliceOutCluster) ;}

    GPUhd() AliHLTTPCCASliceOutTrack *NextTrack(){
      return ( AliHLTTPCCASliceOutTrack*)( ((char*)this) + GetSize( fNClusters ) );
    }

    GPUhd() const AliHLTTPCCASliceOutTrack *GetNextTrack() const{
      return ( AliHLTTPCCASliceOutTrack*)( ((char*)this) + GetSize( fNClusters ) );
    }

  private:

    AliHLTTPCCABaseTrackParam fParam; //* fitted track parameters at its innermost cluster
    int fNClusters;             //* number of track clusters
    AliHLTTPCCASliceOutCluster fClusters[0]; //* track clusters
};


#endif

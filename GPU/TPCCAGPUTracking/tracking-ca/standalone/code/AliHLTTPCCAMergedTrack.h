//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCAMERGEDTRACK_H
#define ALIHLTTPCCAMERGEDTRACK_H

#include "AliHLTTPCCATrackParam.h"

/**
 * @class AliHLTTPCCAMergedTrack
 * AliHLTTPCCAMergedTrack class is used to store TPC tracks,
 * which are reconstructed by the TPCCATracker slice tracker.
 *
 * The class contains:
 * - fitted track parameters at its first row, the covariance matrix, \Chi^2, NDF (number of degrees of freedom )
 * - n of clusters assigned to the track
 * - index of its first cluster in corresponding cluster arrays
 *
 * The class is used to transport the data between AliHLTTPCCATracker{Component} and AliHLTTPCCAGBMerger{Component}
 *
 */
class AliHLTTPCCAMergedTrack
{
  public:

    GPUhd() int NClusters()                         const { return fNClusters;       }
    GPUhd() int FirstClusterRef()                   const { return fFirstClusterRef; }
    GPUhd() const AliHLTTPCCATrackParam &InnerParam() const { return fInnerParam;      }
    GPUhd() const AliHLTTPCCATrackParam &OuterParam() const { return fOuterParam;      }
    GPUhd() float InnerAlpha()                      const { return fInnerAlpha;      }
    GPUhd() float OuterAlpha()                      const { return fOuterAlpha;      }

    GPUhd() void SetNClusters      ( int v )                  { fNClusters = v;       }
    GPUhd() void SetFirstClusterRef( int v )                  { fFirstClusterRef = v; }
    GPUhd() void SetInnerParam( const AliHLTTPCCATrackParam &v ) { fInnerParam = v;      }
    GPUhd() void SetOuterParam( const AliHLTTPCCATrackParam &v ) { fOuterParam = v;      }
    GPUhd() void SetInnerAlpha( float v )                       { fInnerAlpha = v;      }
    GPUhd() void SetOuterAlpha( float v )                       { fOuterAlpha = v;      }

  private:

    AliHLTTPCCATrackParam fInnerParam; //* fitted track parameters at the TPC inner radius
    AliHLTTPCCATrackParam fOuterParam; //* fitted track parameters at the TPC outer radius
    float fInnerAlpha;               //* alpha angle for the inner parameters
    float fOuterAlpha;               //* alpha angle for the outer parameters
    int fFirstClusterRef;            //* index of the first track cluster in corresponding cluster arrays
    int fNClusters;                  //* number of track clusters
};


#endif //ALIHLTTPCCAMERGEDTRACK_H

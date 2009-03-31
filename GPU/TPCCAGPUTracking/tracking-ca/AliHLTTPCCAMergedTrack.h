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

  GPUhd() Int_t NClusters()                         const { return fNClusters;       }
  GPUhd() Int_t FirstClusterRef()                   const { return fFirstClusterRef; }
  GPUhd() const AliHLTTPCCATrackParam &InnerParam() const { return fInnerParam;      }
  GPUhd() const AliHLTTPCCATrackParam &OuterParam() const { return fOuterParam;      }
  GPUhd() Float_t InnerAlpha()                      const { return fInnerAlpha;      }
  GPUhd() Float_t OuterAlpha()                      const { return fOuterAlpha;      }

  GPUhd() void SetNClusters      ( Int_t v )                  { fNClusters = v;       }
  GPUhd() void SetFirstClusterRef( Int_t v )                  { fFirstClusterRef = v; }
  GPUhd() void SetInnerParam( const AliHLTTPCCATrackParam &v) { fInnerParam = v;      }
  GPUhd() void SetOuterParam( const AliHLTTPCCATrackParam &v) { fOuterParam = v;      }
  GPUhd() void SetInnerAlpha( Float_t v )                       { fInnerAlpha = v;      }
  GPUhd() void SetOuterAlpha( Float_t v )                       { fOuterAlpha = v;      }

 private:
  
  AliHLTTPCCATrackParam fInnerParam; //* fitted track parameters at the TPC inner radius
  AliHLTTPCCATrackParam fOuterParam; //* fitted track parameters at the TPC outer radius
  Float_t fInnerAlpha;               //* alpha angle for the inner parameters
  Float_t fOuterAlpha;               //* alpha angle for the outer parameters
  Int_t fFirstClusterRef;            //* index of the first track cluster in corresponding cluster arrays
  Int_t fNClusters;                  //* number of track clusters
};


#endif

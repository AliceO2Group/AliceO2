//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCGMMERGEDTRACK_H
#define ALIHLTTPCGMMERGEDTRACK_H

#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCGMMergedTrackHit.h"

/**
 * @class AliHLTTPCGMMergedTrack
 * 
 * The class is used to store merged tracks in AliHLTTPCGMMerger
 */
class AliHLTTPCGMMergedTrack
{
 public:

  GPUd() int NClusters()                      const { return fNClusters;       }
  GPUd() int NClustersFitted()                const { return fNClustersFitted; }
  GPUd() int FirstClusterRef()                const { return fFirstClusterRef; }
  GPUd() const AliHLTTPCGMTrackParam &GetParam() const { return fParam;      }
  GPUd() float GetAlpha()                        const { return fAlpha;      }
  GPUd() AliHLTTPCGMTrackParam &Param() { return fParam;      }
  GPUd() float &Alpha()                 { return fAlpha;      }
  GPUd() float LastX()                        const { return fLastX; }
  GPUd() float LastY()                        const { return fLastY; }
  GPUd() float LastZ()                        const { return fLastZ; }
  GPUd() bool OK() const { return fOK; }
  GPUd() char CSide() const { return fCSide; }

  GPUd() void SetNClusters      ( int v )                { fNClusters = v;       }
  GPUd() void SetNClustersFitted( int v )                { fNClustersFitted = v; }
  GPUd() void SetFirstClusterRef( int v )                { fFirstClusterRef = v; }
  GPUd() void SetParam( const AliHLTTPCGMTrackParam &v ) { fParam = v;      }     
  GPUd() void SetAlpha( float v )                        { fAlpha = v;      }  
  GPUd() void SetLastX( float v )                        { fLastX = v; }
  GPUd() void SetLastY( float v )                        { fLastY = v; }
  GPUd() void SetLastZ( float v )                        { fLastZ = v; }
  GPUd() void SetOK( bool v ) {fOK = v;}
  GPUd() void SetCSide( char v ) {fCSide = v;}
  
 private:

  AliHLTTPCGMTrackParam fParam; //* fitted track parameters 

  float fAlpha;                 //* alpha angle 
  float fLastX; //* outer X
  float fLastY; //* outer Y
  float fLastZ; //* outer Z
  int fFirstClusterRef;         //* index of the first track cluster in corresponding cluster arrays
  int fNClusters;               //* number of track clusters
  int fNClustersFitted;         //* number of clusters used in fit
  bool fOK;//
  char fCSide;
};

#endif 

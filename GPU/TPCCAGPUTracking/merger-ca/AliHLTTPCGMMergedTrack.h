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

/**
 * @class AliHLTTPCGMMergedTrack
 * 
 * The class is used to store merged tracks in AliHLTTPCGMMerger
 */
class AliHLTTPCGMMergedTrack
{
 public:

  int NClusters()                      const { return fNClusters;       }
  int FirstClusterRef()                const { return fFirstClusterRef; }
  const AliHLTTPCGMTrackParam &GetParam() const { return fParam;      }
  float GetAlpha()                        const { return fAlpha;      }
  AliHLTTPCGMTrackParam &Param() { return fParam;      }
  float &Alpha()                 { return fAlpha;      }
  float LastX()                        const { return fLastX; }
  float LastY()                        const { return fLastY; }
  float LastZ()                        const { return fLastZ; }
  bool OK() const{ return fOK; }

  void SetNClusters      ( int v )                { fNClusters = v;       }
  void SetFirstClusterRef( int v )                { fFirstClusterRef = v; }
  void SetParam( const AliHLTTPCGMTrackParam &v ) { fParam = v;      }     
  void SetAlpha( float v )                        { fAlpha = v;      }  
  void SetLastX( float v )                        { fLastX = v; }
  void SetLastY( float v )                        { fLastY = v; }
  void SetLastZ( float v )                        { fLastZ = v; }
  void SetOK( bool v ) {fOK = v;}
 private:

  AliHLTTPCGMTrackParam fParam; //* fitted track parameters 

  float fAlpha;                 //* alpha angle 
  float fLastX; //* outer X
  float fLastY; //* outer Y
  float fLastZ; //* outer Z
  int fFirstClusterRef;         //* index of the first track cluster in corresponding cluster arrays
  int fNClusters;               //* number of track clusters
  bool fOK;//
};


#endif 

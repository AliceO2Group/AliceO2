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

	GPUd() int NClusters()                         const { return fNClusters;       }
	GPUd() int NClustersFitted()                   const { return fNClustersFitted; }
	GPUd() int FirstClusterRef()                   const { return fFirstClusterRef; }
	GPUd() const AliHLTTPCGMTrackParam &GetParam() const { return fParam;           }
	GPUd() float GetAlpha()                        const { return fAlpha;           }
	GPUd() AliHLTTPCGMTrackParam &Param()                { return fParam;           }
	GPUd() float &Alpha()                                { return fAlpha;           }
	GPUd() float LastX()                           const { return fLastX;           }
	GPUd() float LastY()                           const { return fLastY;           }
	GPUd() float LastZ()                           const { return fLastZ;           }
	GPUd() bool OK()                               const { return fFlags & 0x01;    }
	GPUd() bool Looper()                           const { return fFlags & 0x02;    }
	GPUd() bool CSide()                            const { return fFlags & 0x04;    }
	GPUd() bool CCE()                              const { return fFlags & 0x08;    }
  
	GPUd() void SetNClusters      ( int v )                { fNClusters = v;       }
	GPUd() void SetNClustersFitted( int v )                { fNClustersFitted = v; }
	GPUd() void SetFirstClusterRef( int v )                { fFirstClusterRef = v; }
	GPUd() void SetParam( const AliHLTTPCGMTrackParam &v ) { fParam = v;      }
	GPUd() void SetAlpha( float v )                        { fAlpha = v;      }
	GPUd() void SetLastX( float v )                        { fLastX = v; }
	GPUd() void SetLastY( float v )                        { fLastY = v; }
	GPUd() void SetLastZ( float v )                        { fLastZ = v; }
	GPUd() void SetOK( bool v )                            { if (v) fFlags |= 0x01; else fFlags &= 0xFE; }
	GPUd() void SetLooper( bool v )                        { if (v) fFlags |= 0x02; else fFlags &= 0xFD; }
	GPUd() void SetCSide( bool v )                         { if (v) fFlags |= 0x04; else fFlags &= 0xFB; }
	GPUd() void SetCCE( bool v )                           { if (v) fFlags |= 0x08; else fFlags &= 0xF7; }
	GPUd() void SetFlags ( unsigned char v )               { fFlags = v; }
  
	GPUd() const AliHLTTPCGMTrackParam::AliHLTTPCCAOuterParam& OuterParam() const {return fOuterParam;}
	GPUd() AliHLTTPCGMTrackParam::AliHLTTPCCAOuterParam& OuterParam() {return fOuterParam;}
  
 private:

	AliHLTTPCGMTrackParam fParam; //* fitted track parameters
	AliHLTTPCGMTrackParam::AliHLTTPCCAOuterParam fOuterParam; //* outer param

	float fAlpha;                 //* alpha angle
	float fLastX; //* outer X
	float fLastY; //* outer Y
	float fLastZ; //* outer Z
	int fFirstClusterRef;         //* index of the first track cluster in corresponding cluster arrays
	int fNClusters;               //* number of track clusters
	int fNClustersFitted;         //* number of clusters used in fit
	unsigned char fFlags;
};

#endif

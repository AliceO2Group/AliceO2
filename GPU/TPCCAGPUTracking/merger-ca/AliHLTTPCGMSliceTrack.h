//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCGMSLICETRACK_H
#define ALIHLTTPCGMSLICETRACK_H

#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCCASliceOutTrack.h"

/**
 * @class AliHLTTPCGMSliceTrack
 *
 * The class describes TPC slice tracks used in AliHLTTPCGMMerger
 */
class AliHLTTPCGMSliceTrack
{
  
 public:
  
  float Alpha()                      const { return fAlpha;      }
  int   NClusters()                  const { return fNClusters;       }
  int   PrevNeighbour()              const { return fPrevNeighbour;   }
  int   NextNeighbour()              const { return fNextNeighbour;   }
  int   SliceNeighbour()             const { return fSliceNeighbour; }
  int   Used()                       const { return fUsed;            }
  const AliHLTTPCCASliceOutTrack* OrigTrack() const { return fOrigTrack; }
  float X()                      const { return fX;      }
  float Y()                      const { return fY;      }
  float Z()                      const { return fZ;      }
  float SinPhi()                      const { return fSinPhi;      }
  float CosPhi()                      const { return fCosPhi;      }
  float SecPhi()                      const { return fSecPhi;      }
  float DzDs()                      const { return fDzDs;      }
  float QPt()                      const { return fQPt;      }

  int  LocalTrackId()        const { return fLocalTrackId; }
  void SetLocalTrackId( int v )        { fLocalTrackId = v; }
  int  GlobalTrackId(int n)        const { return fGlobalTrackIds[n]; }
  void SetGlobalTrackId( int n, int v )        { fGlobalTrackIds[n] = v; }


  void Set( const AliHLTTPCCASliceOutTrack *sliceTr, float alpha ){
    const AliHLTTPCCABaseTrackParam &t = sliceTr->Param();
    fOrigTrack = sliceTr;
    fX = t.GetX();
    fY = t.GetY();
    fZ = t.GetZ();
    fDzDs = t.GetDzDs();
    fSinPhi = t.GetSinPhi();
    fQPt = t.GetQPt();
    fCosPhi = sqrt(1.f - fSinPhi*fSinPhi);
    fSecPhi = 1.f / fCosPhi;
    fAlpha = alpha;
    fNClusters = sliceTr->NClusters();    
  }
  
  void SetNClusters ( int v )                        { fNClusters = v;       }
  void SetPrevNeighbour( int v )                     { fPrevNeighbour = v;   }
  void SetNextNeighbour( int v )                     { fNextNeighbour = v;   }
  void SetUsed( int v )                             { fUsed = v;            }
  void SetSliceNeighbour( int v )                    { fSliceNeighbour = v;            }


  void CopyParamFrom( const AliHLTTPCGMSliceTrack &t ){
    fX=t.fX; fY=t.fY; fZ=t.fZ;
    fSinPhi=t.fSinPhi; fDzDs=t.fDzDs; fQPt=t.fQPt; fCosPhi=t.fCosPhi, fSecPhi=t.fSecPhi;
    fAlpha = t.fAlpha;
  }

  bool FilterErrors( AliHLTTPCCAParam &param, float maxSinPhi =.999 );

  bool TransportToX( float x, float Bz, AliHLTTPCGMBorderTrack &b, float maxSinPhi, bool doCov = true ) const ;

  bool TransportToXAlpha( float x, float sinAlpha, float cosAlpha, float Bz, AliHLTTPCGMBorderTrack &b, float maxSinPhi ) const ;

 private:

  const AliHLTTPCCASliceOutTrack *fOrigTrack; // pointer to original slice track
  float fX, fY, fZ, fSinPhi, fDzDs, fQPt, fCosPhi, fSecPhi; // parameters
  float fC0, fC2, fC3, fC5, fC7, fC9, fC10, fC12, fC14; // covariances
  float fAlpha;           // alpha angle 
  int fNClusters;         // N clusters
  int fPrevNeighbour;     // neighbour in the previous slise
  int fNextNeighbour;     // neighbour in the next slise
  int fSliceNeighbour;    // next neighbour withing the same slice;
  int fUsed;              // is the slice track already merged
  int fLocalTrackId;	  // Corrected local track id in terms of GMSliceTracks array
  int fGlobalTrackIds[2]; // IDs of associated global tracks
};

#endif

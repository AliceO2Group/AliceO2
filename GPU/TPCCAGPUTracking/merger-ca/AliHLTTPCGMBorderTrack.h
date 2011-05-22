//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCGMBORDERTRACK_H
#define ALIHLTTPCGMBORDERTRACK_H

#include "AliHLTTPCCAMath.h"

/**
 * @class AliHLTTPCGMBorderTrack
 *
 * The class describes TPC slice tracks at sector borders. 
 * Used in AliHLTTPCGMMerger
 *
 */
class AliHLTTPCGMBorderTrack
{

 public:

  struct Range{
    int fId;
    float fMin, fMax;
    static bool CompMin(const Range &a, const Range &b)  { return a.fMin<b.fMin; }
    static bool CompMax(const Range &a, const Range &b)  { return a.fMax<b.fMax; }
  };


  int   TrackID()                    const { return fTrackID;   }
  int   NClusters()                  const { return fNClusters; }  
  const float *Par() const { return fP; }
  const float *Cov() const { return fC; }
  const float *CovD() const { return fD; }

  void SetTrackID   ( int v )                        { fTrackID   = v; }
  void SetNClusters ( int v )                        { fNClusters = v; }
  void SetPar( int i, float x ) { fP[i] = x; }
  void SetCov( int i, float x ) { fC[i] = x; }
  void SetCovD( int i, float x ) { fD[i] = x; }
 
  static bool CheckChi2( float x1, float y1, float cx1, float cxy1, float cy1,
			  float x2, float y2, float cx2, float cxy2, float cy2, float chi2cut  )
  {
    //* Calculate Chi2/ndf deviation
    float dx = x1 - x2;
    float dy = y1 - y2;
    float cx = cx1 + cx2;
    float cxy = cxy1 + cxy2;
    float cy = cy1 + cy2;
    float det = cx*cy - cxy*cxy ;
    return ( ( cy*dx - (cxy+cxy)*dy )*dx + cx*dy*dy < (det+det)*chi2cut );
  }

  bool CheckChi2Y( const AliHLTTPCGMBorderTrack &t, float chi2cut ) const {
    float d = fP[0]-t.fP[0];
    return ( d*d < chi2cut*(fC[0] + t.fC[0]) );
  }

  bool CheckChi2Z( const AliHLTTPCGMBorderTrack &t, float chi2cut ) const {
    float d = fP[1]-t.fP[1];
    return ( d*d < chi2cut *(fC[1] + t.fC[1]) );
  }

  bool CheckChi2QPt( const AliHLTTPCGMBorderTrack &t, float chi2cut  ) const {
    float d = fP[4]-t.fP[4];
    return ( d*d < chi2cut*(fC[4] + t.fC[4]) );
  }
  
  bool CheckChi2YS( const AliHLTTPCGMBorderTrack &t, float chi2cut ) const {
    return CheckChi2(   fP[0],   fP[2],   fC[0],   fD[0], fC[2],
		      t.fP[0], t.fP[2], t.fC[0], t.fD[0], t.fC[2], chi2cut );
  }
 
  bool CheckChi2ZT( const AliHLTTPCGMBorderTrack &t, float chi2cut ) const {
    return  CheckChi2(   fP[1],   fP[3],   fC[1],   fD[1], fC[3],
		       t.fP[1], t.fP[3], t.fC[1], t.fD[1], t.fC[3], chi2cut );
    
  }

 private:

  int   fTrackID;              // track index
  int   fNClusters;            // n clusters
  float fP[5];
  float fC[5];
  float fD[2];
};

#endif

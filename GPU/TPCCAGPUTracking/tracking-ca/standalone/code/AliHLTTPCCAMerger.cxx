// $Id: AliHLTTPCCAMerger.cxx 30732 2009-01-22 23:02:02Z sgorbuno $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include <stdio.h>
#include "AliHLTTPCCASliceTrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackParam.h"

#include "AliHLTTPCCAMerger.h"

#include "AliHLTTPCCAMath.h"
#include "TStopwatch.h"

#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCASliceTrack.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCAMergedTrack.h"
#include "AliHLTTPCCAMergerOutput.h"
#include "AliHLTTPCCADataCompressor.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackLinearisation.h"


class AliHLTTPCCAMerger::AliHLTTPCCASliceTrackInfo
{

  public:

    const AliHLTTPCCATrackParam &InnerParam() const { return fInnerParam;      }
    const AliHLTTPCCATrackParam &OuterParam() const { return fOuterParam;      }
    float InnerAlpha()                      const { return fInnerAlpha;      }
    float OuterAlpha()                      const { return fOuterAlpha;      }
    int   NClusters()                       const { return fNClusters;       }
    int   FirstClusterRef()                 const { return fFirstClusterRef; }
    int   PrevNeighbour()                   const { return fPrevNeighbour;   }
    int   NextNeighbour()                   const { return fNextNeighbour;   }
    bool  Used()                            const { return fUsed;            }

    void SetInnerParam( const AliHLTTPCCATrackParam &v ) { fInnerParam = v;      }
    void SetOuterParam( const AliHLTTPCCATrackParam &v ) { fOuterParam = v;      }
    void SetInnerAlpha( float v )                      { fInnerAlpha = v;      }
    void SetOuterAlpha( float v )                      { fOuterAlpha = v;      }
    void SetNClusters ( int v )                        { fNClusters = v;       }
    void SetFirstClusterRef( int v )                   { fFirstClusterRef = v; }
    void SetPrevNeighbour( int v )                     { fPrevNeighbour = v;   }
    void SetNextNeighbour( int v )                     { fNextNeighbour = v;   }
    void SetUsed( bool v )                             { fUsed = v;            }

  private:

    AliHLTTPCCATrackParam fInnerParam; // inner parameters
    AliHLTTPCCATrackParam fOuterParam; // outer parameters
    float fInnerAlpha;               // alpha angle for inner parameters
    float fOuterAlpha;               // alpha angle for outer parameters
    int fNClusters;                  // N clusters
    int fFirstClusterRef;  // index of the first track cluster in the global cluster array
    int fPrevNeighbour;    // neighbour in the previous slise
    int fNextNeighbour;    // neighbour in the next slise
    bool fUsed;            // is the slice track already merged

};


class AliHLTTPCCAMerger::AliHLTTPCCABorderTrack
{

  public:

    const AliHLTTPCCATrackParam &Param() const { return fParam;     }
    int   TrackID()                    const { return fTrackID;   }
    int   NClusters()                  const { return fNClusters; }
    int   IRow()                       const { return fIRow;      }
    float X()                          const { return fX;         }
    bool  OK()                         const { return fOK;        }

    void SetParam     ( const AliHLTTPCCATrackParam &v ) { fParam     = v; }
    void SetTrackID   ( int v )                        { fTrackID   = v; }
    void SetNClusters ( int v )                        { fNClusters = v; }
    void SetIRow      ( int v )                        { fIRow      = v; }
    void SetX         ( float v )                      { fX         = v; }
    void SetOK        ( bool v )                       { fOK        = v; }

  private:

    AliHLTTPCCATrackParam fParam;  // track parameters at the border
    int   fTrackID;              // track index
    int   fNClusters;            // n clusters
    int   fIRow;                 // row number of the closest cluster
    float fX;                    // X coordinate of the closest cluster
    bool  fOK;                   // is the track rotated and extrapolated correctly

};



AliHLTTPCCAMerger::AliHLTTPCCAMerger()
    :
    fSliceParam(),
    fOutput( 0 ),
    fTrackInfos( 0 ),
    fMaxTrackInfos( 0 ),
    fClusterInfos( 0 ),
    fMaxClusterInfos( 0 )
{
  //* constructor
  Clear();
}

/*
AliHLTTPCCAMerger::AliHLTTPCCAMerger(const AliHLTTPCCAMerger&)
  :
  fSliceParam(),
  fkSlices(0),
  fOutput(0),
  fTrackInfos(0),
  fMaxTrackInfos(0),
  fClusterInfos(0),
  fMaxClusterInfos(0)
{
}

const AliHLTTPCCAMerger &AliHLTTPCCAMerger::operator=(const AliHLTTPCCAMerger&) const
{
  return *this;
}
*/

AliHLTTPCCAMerger::~AliHLTTPCCAMerger()
{
  //* destructor
  if ( fTrackInfos ) delete[] fTrackInfos;
  if ( fClusterInfos ) delete[] fClusterInfos;
  if ( fOutput ) delete[] ( ( float2* )( fOutput ) );
}

void AliHLTTPCCAMerger::Clear()
{
  for ( int i = 0; i < fgkNSlices; ++i ) {
    fkSlices[i] = 0;
    fSliceNTrackInfos[ i ] = 0;
    fSliceTrackInfoStart[ i ] = 0;
  }
  if ( fOutput ) delete[] ( ( float2* )( fOutput ) );
  if ( fTrackInfos ) delete[] fTrackInfos;
  if ( fClusterInfos ) delete[] fClusterInfos;
  fOutput = 0;
  fTrackInfos = 0;
  fClusterInfos = 0;
  fMaxTrackInfos = 0;
  fMaxClusterInfos = 0;
}


void AliHLTTPCCAMerger::SetSliceData( int index, const AliHLTTPCCASliceOutput *SliceData )
{
  fkSlices[index] = SliceData;
}

void AliHLTTPCCAMerger::Reconstruct()
{
  //* main merging routine

  UnpackSlices();
  Merging();
}

void AliHLTTPCCAMerger::UnpackSlices()
{
  //* unpack the cluster information from the slice tracks and initialize track info array

  // get N tracks and N clusters in event

  int nTracksTotal = 0;
  int nTrackClustersTotal = 0;
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    if ( !fkSlices[iSlice] ) continue;
    nTracksTotal += fkSlices[iSlice]->NTracks();
    nTrackClustersTotal += fkSlices[iSlice]->NTrackClusters();
  }

  // book/clean memory if necessary
  {
    if ( nTracksTotal > fMaxTrackInfos || ( fMaxTrackInfos > 100 && nTracksTotal < 0.5*fMaxTrackInfos ) ) {
      if ( fTrackInfos ) delete[] fTrackInfos;
      fMaxTrackInfos = ( int ) ( nTracksTotal * 1.2 );
      fTrackInfos = new AliHLTTPCCASliceTrackInfo[fMaxTrackInfos];
    }

    if ( nTrackClustersTotal > fMaxClusterInfos || ( fMaxClusterInfos > 1000 && nTrackClustersTotal < 0.5*fMaxClusterInfos ) ) {
      if ( fClusterInfos ) delete[] fClusterInfos;
      fMaxClusterInfos = ( int ) ( nTrackClustersTotal * 1.2 );
      fClusterInfos = new AliHLTTPCCAClusterInfo [fMaxClusterInfos];
    }

    if ( fOutput ) delete[] ( ( float2* )( fOutput ) );
    int size = fOutput->EstimateSize( nTracksTotal, nTrackClustersTotal );
    fOutput = ( AliHLTTPCCAMergerOutput* )( new float2[size/sizeof( float2 )+1] );
  }

  // unpack track and cluster information

  int nTracksCurrent = 0;
  int nClustersCurrent = 0;

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

    fSliceTrackInfoStart[ iSlice ] = nTracksCurrent;
    fSliceNTrackInfos[ iSlice ] = 0;

    if ( !fkSlices[iSlice] ) continue;

    const AliHLTTPCCASliceOutput &slice = *( fkSlices[iSlice] );

    for ( int itr = 0; itr < slice.NTracks(); itr++ ) {

      const AliHLTTPCCASliceTrack &sTrack = slice.Track( itr );
      AliHLTTPCCATrackParam t0;
	  t0.InitParam();
	  t0.SetParam(sTrack.Param());
      int nCluNew = 0;

      for ( int iTrClu = 0; iTrClu < sTrack.NClusters(); iTrClu++ ) {

        // unpack cluster information

        AliHLTTPCCAClusterInfo &clu = fClusterInfos[nClustersCurrent + nCluNew];
        int ic = sTrack.FirstClusterRef() + iTrClu;

        clu.SetISlice( iSlice );
        clu.SetIRow( slice.ClusterRow( ic ) );
        clu.SetId( slice.ClusterId( ic ) );
        clu.SetPackedAmp( 0 );
        float2 yz = slice.ClusterUnpackedYZ( ic );
        clu.SetX( slice.ClusterUnpackedX( ic ) );
        clu.SetY( yz.x );
        clu.SetZ( yz.y );

        if ( !t0.TransportToX( clu.X(), fSliceParam.GetBz( t0 ), .999 ) ) continue;

        float err2Y, err2Z;
        fSliceParam.GetClusterErrors2( clu.IRow(), clu.Z(), t0.SinPhi(), t0.GetCosPhi(), t0.DzDs(), err2Y, err2Z );

        clu.SetErr2Y( err2Y );
        clu.SetErr2Z( err2Z );
        nCluNew++ ;
      }

      if ( nCluNew < .8*sTrack.NClusters() ) continue;

      // refit the track

      int hits[1000];
      int nHits = nCluNew;
      for ( int i = 0; i < nHits; i++ ) hits[i] = nClustersCurrent + i;

      AliHLTTPCCATrackParam startPoint;
	  startPoint.InitParam();
	  startPoint.SetParam(sTrack.Param());
      AliHLTTPCCATrackParam endPoint = startPoint;
      float startAlpha = fSliceParam.Alpha( iSlice );
      float endAlpha = startAlpha;

      if ( !FitTrack( endPoint, endAlpha, startPoint, startAlpha, hits, nHits, 0 ) ) continue;

      startPoint = endPoint;
      startAlpha = endAlpha;
      if ( !FitTrack( startPoint, startAlpha, endPoint, endAlpha, hits, nHits, 1 ) ) continue;

      if ( nHits < .8*sTrack.NClusters() ) continue;

      // store the track

      AliHLTTPCCASliceTrackInfo &track = fTrackInfos[nTracksCurrent];

      track.SetInnerParam( startPoint );
      track.SetInnerAlpha( startAlpha );
      track.SetOuterParam( endPoint );
      track.SetOuterAlpha( endAlpha );
      track.SetFirstClusterRef( nClustersCurrent );
      track.SetNClusters( nHits );
      track.SetPrevNeighbour( -1 );
      track.SetNextNeighbour( -1 );
      track.SetUsed( 0 );

      for ( int i = 0; i < nHits; i++ )
        fClusterInfos[nClustersCurrent + i] = fClusterInfos[hits[i]];
      nTracksCurrent++;
      fSliceNTrackInfos[ iSlice ]++;
      nClustersCurrent += nHits;
    }
    //std::cout<<"Unpack slice "<<iSlice<<": ntracks "<<slice.NTracks()<<"/"<<fSliceNTrackInfos[iSlice]<<std::endl;
  }
}



bool AliHLTTPCCAMerger::FitTrack( AliHLTTPCCATrackParam &T, float &Alpha,
                                  AliHLTTPCCATrackParam t0, float Alpha0,
                                  int hits[], int &NTrackHits, bool dir, bool final,
                                  AliHLTTPCCAClusterInfo *infoArray )
{
  // Fit the track

  AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
  AliHLTTPCCATrackParam t = t0;
  AliHLTTPCCATrackLinearisation l( t0 );

  bool first = 1;
  bool doErrors = 1;
  if ( !infoArray ) {
    infoArray = fClusterInfos;
    doErrors = 0;
  }

  t.CalculateFitParameters( fitPar );

  int hitsNew[1000];
  int nHitsNew = 0;

  for ( int ihit = 0; ihit < NTrackHits; ihit++ ) {

    int jhit = dir ? ( NTrackHits - 1 - ihit ) : ihit;
    AliHLTTPCCAClusterInfo &h = infoArray[hits[jhit]];

    int iSlice = h.ISlice();

    float sliceAlpha =  fSliceParam.Alpha( iSlice );

    if ( CAMath::Abs( sliceAlpha - Alpha0 ) > 1.e-4 ) {
      if ( ! t.Rotate(  sliceAlpha - Alpha0, l, .999 ) ) continue;
      Alpha0 = sliceAlpha;
    }

    //float x = fSliceParam.RowX( h.IRow() );
    float x = h.X();

    if ( !t.TransportToXWithMaterial( x, l, fitPar, fSliceParam.GetBz( t ) ) ) continue;

    if ( first ) {
      t.SetCov( 0, 10 );
      t.SetCov( 1,  0 );
      t.SetCov( 2, 10 );
      t.SetCov( 3,  0 );
      t.SetCov( 4,  0 );
      t.SetCov( 5,  1 );
      t.SetCov( 6,  0 );
      t.SetCov( 7,  0 );
      t.SetCov( 8,  0 );
      t.SetCov( 9,  1 );
      t.SetCov( 10,  0 );
      t.SetCov( 11,  0 );
      t.SetCov( 12,  0 );
      t.SetCov( 13,  0 );
      t.SetCov( 14,  10 );
      t.SetChi2( 0 );
      t.SetNDF( -5 );
      t.CalculateFitParameters( fitPar );
    }

    float err2Y = h.Err2Y();
    float err2Z = h.Err2Z();
    if ( doErrors ) fSliceParam.GetClusterErrors2( h.IRow(), h.Z(), l.SinPhi(), l.CosPhi(), l.DzDs(), err2Y, err2Z );
    if( !final ){
      err2Y*= fSliceParam.ClusterError2CorrectionY();
      err2Z*= fSliceParam.ClusterError2CorrectionZ();
    }

    if ( !t.Filter( h.Y(), h.Z(), err2Y, err2Z ) ) continue;

    first = 0;

    hitsNew[nHitsNew++] = hits[jhit];
  }

  if ( CAMath::Abs( t.QPt() ) < 1.e-4 ) t.SetQPt( 1.e-4 );

  bool ok = t.CheckNumericalQuality();

  if ( CAMath::Abs( t.SinPhi() ) > .99 ) ok = 0;
  else if ( l.CosPhi() >= 0 ) t.SetSignCosPhi( 1 );
  else t.SetSignCosPhi( -1 );

  if ( ok ) {
    T = t;
    Alpha = Alpha0;
    NTrackHits = nHitsNew;
    for ( int i = 0; i < NTrackHits; i++ ) {
      hits[dir ?( NTrackHits-1-i ) :i] = hitsNew[i];
    }
  }
  return ok;
}


float AliHLTTPCCAMerger::GetChi2( float x1, float y1, float a00, float a10, float a11,
                                  float x2, float y2, float b00, float b10, float b11  )
{
  //* Calculate Chi2/ndf deviation

  float d[2] = { x1 - x2, y1 - y2 };

  float mSi[3] = { a00 + b00, a10 + b10, a11 + b11 };

  float s = ( mSi[0] * mSi[2] - mSi[1] * mSi[1] );

  if ( s < 1.E-10 ) return 10000.;

  float mS[3] = { mSi[2], -mSi[1], mSi[0] };

  return AliHLTTPCCAMath::Abs( ( ( mS[0]*d[0] + mS[1]*d[1] )*d[0]
                       + ( mS[1]*d[0] + mS[2]*d[1] )*d[1] ) / s / 2 );

}



void AliHLTTPCCAMerger::MakeBorderTracks( int iSlice, int iBorder, AliHLTTPCCABorderTrack B[], int &nB )
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line,
  //* in some cases both inner and outer parameters of the track are transported

  static int statAll = 0, statOK = 0;
  nB = 0;
  float dAlpha = fSliceParam.DAlpha() / 2;
  float x0 = 0;

  if ( iBorder == 0 ) { // transport to the left age of the sector and rotate horisontally
    dAlpha = dAlpha - CAMath::Pi() / 2 ;
  } else if ( iBorder == 1 ) { //  transport to the right age of the sector and rotate horisontally
    dAlpha = -dAlpha - CAMath::Pi() / 2 ;
  } else if ( iBorder == 2 ) { // transport to the left age of the sector and rotate vertically
    dAlpha = dAlpha;
    x0 = fSliceParam.RowX( 63 );
  } else if ( iBorder == 3 ) { // transport to the right age of the sector and rotate vertically
    dAlpha = -dAlpha;
    x0 =  fSliceParam.RowX( 63 );
  } else if ( iBorder == 4 ) { // transport to the middle of the sector, w/o rotation
    dAlpha = 0;
    x0 = fSliceParam.RowX( 63 );
  }

  for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {

    const AliHLTTPCCASliceTrackInfo &track = fTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];

    AliHLTTPCCATrackParam t0 = track.InnerParam();
    AliHLTTPCCATrackParam t1 = track.OuterParam();

    const float maxSin = CAMath::Sin( 60. / 180.*CAMath::Pi() );

    bool ok0 = t0.Rotate( dAlpha, maxSin );
    bool ok1 = t1.Rotate( dAlpha, maxSin );

    bool do0 = ok0;
    bool do1 = ok1 && ( !ok0 || t1.SignCosPhi() * t0.SignCosPhi() < 0 );

    if ( ok0 && !do1 && ok1 && ( t1.X() < t0.X() ) ) {
      do0 = 0;
      do1 = 1;
    }

    if ( do0 ) {
      AliHLTTPCCABorderTrack &b = B[nB];
      b.SetX( t0.GetX() );
      
      if ( t0.TransportToX( x0, fSliceParam.GetBz( t0 ), maxSin ) ) {
        b.SetOK( 1 );
        b.SetTrackID( itr );
        b.SetNClusters( track.NClusters() );
        b.SetIRow( fClusterInfos[ track.FirstClusterRef() + 0 ].IRow() );
        b.SetParam( t0 );
        nB++;
      }
    }
    if ( do1 ) {
      AliHLTTPCCABorderTrack &b = B[nB];
      b.SetX( t1.GetX() );
      
      if ( t1.TransportToX( x0, fSliceParam.GetBz( t1 ), maxSin ) ) {
        b.SetOK( 1 );
        b.SetTrackID( itr );
        b.SetNClusters( track.NClusters() );
        b.SetIRow( fClusterInfos[ track.FirstClusterRef() + track.NClusters()-1 ].IRow() );
        b.SetParam( t1 );
        nB++;
      }
    }
    if ( do0 || do1 ) statOK++;
    statAll++;
  }
}



void AliHLTTPCCAMerger::MergeBorderTracks( int iSlice1, AliHLTTPCCABorderTrack B1[], int N1,
    int iSlice2, AliHLTTPCCABorderTrack B2[], int N2
                                         )
{
  //* merge two sets of tracks

  //std::cout<<" Merge slices "<<iSlice1<<"+"<<iSlice2<<": tracks "<<N1<<"+"<<N2<<std::endl;
  
  float factor2ys = 1.;//1.5;//SG!!!
  float factor2zt = 1.;//1.5;//SG!!!
  float factor2k = 2.0;//2.2;

  factor2k  = 3.5 * 3.5 * factor2k * factor2k;
  factor2ys = 3.5 * 3.5 * factor2ys * factor2ys;
  factor2zt = 3.5 * 3.5 * factor2zt * factor2zt;

  int minNPartHits = 10;//SG!!!
  int minNTotalHits = 20;

  //float maxDX = fSliceParam.RowX(40) -  fSliceParam.RowX(0);

  for ( int i1 = 0; i1 < N1; i1++ ) {
    AliHLTTPCCABorderTrack &b1 = B1[i1];
    if ( !b1.OK() ) continue;
    if ( b1.NClusters() < minNPartHits ) continue;
    const AliHLTTPCCATrackParam &t1 = b1.Param();
    int iBest2 = -1;
    int lBest2 = 0;
    int start2 = ( iSlice1 != iSlice2 ) ? 0 : i1 + 1;
    for ( int i2 = start2; i2 < N2; i2++ ) {
      AliHLTTPCCABorderTrack &b2 = B2[i2];
      if ( !b2.OK() ) continue;
      if ( b2.NClusters() < minNPartHits ) continue;
      if ( b2.NClusters() < lBest2 ) continue;
      if ( b1.NClusters() + b2.NClusters() < minNTotalHits ) continue;

      //if( TMath::Abs(b1.fX - b2.fX)>maxDX ) continue;

      const AliHLTTPCCATrackParam &t2 = b2.Param();

      float c = t2.SignCosPhi() * t1.SignCosPhi() >= 0 ? 1 : -1;
      float dk = t2.QPt() - c * t1.QPt();
      float s2k = t2.Err2QPt() + t1.Err2QPt();
      //std::cout<<" check 1.. "<<dk/sqrt(factor2k)<<std::endl;
      if ( dk*dk > factor2k*s2k ) continue;


      float chi2ys = GetChi2( t1.Y(), c * t1.SinPhi(), t1.Cov()[0], c * t1.Cov()[3], t1.Cov()[5],
                              t2.Y(),  t2.SinPhi(), t2.Cov()[0],  t2.Cov()[3], t2.Cov()[5] );

      //std::cout<<" check 2.. "<<sqrt(chi2ys/factor2ys)<<std::endl;
      if ( chi2ys > factor2ys ) continue;
      

      float chi2zt = GetChi2( t1.Z(), c * t1.DzDs(), t1.Cov()[2], c * t1.Cov()[7], t1.Cov()[9],
                              t2.Z(),  t2.DzDs(), t2.Cov()[2],  t2.Cov()[7], t2.Cov()[9] );

      //std::cout<<" check 3.. "<<sqrt(chi2zt/factor2zt)<<std::endl;
      if ( chi2zt > factor2zt ) continue;

      lBest2 = b2.NClusters();
      iBest2 = b2.TrackID();
    }

    if ( iBest2 < 0 ) continue;

    //std::cout<<"Neighbour found for "<<i1<<": "<<iBest2<<std::endl;
    AliHLTTPCCASliceTrackInfo &newTrack1 = fTrackInfos[fSliceTrackInfoStart[iSlice1] + b1.TrackID() ];
    AliHLTTPCCASliceTrackInfo &newTrack2 = fTrackInfos[fSliceTrackInfoStart[iSlice2] + iBest2 ];

    int old1 = newTrack2.PrevNeighbour();

    if ( old1 >= 0 ) {
      AliHLTTPCCASliceTrackInfo &oldTrack1 = fTrackInfos[fSliceTrackInfoStart[iSlice1] + old1];
      if ( oldTrack1.NClusters()  < newTrack1.NClusters() ) {
        newTrack2.SetPrevNeighbour( -1 );
        oldTrack1.SetNextNeighbour( -1 );
      } else continue;
    }
    int old2 = newTrack1.NextNeighbour();
    if ( old2 >= 0 ) {
      AliHLTTPCCASliceTrackInfo &oldTrack2 = fTrackInfos[fSliceTrackInfoStart[iSlice2] + old2];
      if ( oldTrack2.NClusters() < newTrack2.NClusters() ) {
        oldTrack2.SetPrevNeighbour( -1 );
      } else continue;
    }
    newTrack1.SetNextNeighbour( iBest2 );
    newTrack2.SetPrevNeighbour( b1.TrackID() );
    //std::cout<<"Neighbourhood is set"<<std::endl;
  }

}


void AliHLTTPCCAMerger::Merging()
{
  //* track merging between slices

  fOutput->SetNTracks( 0 );
  fOutput->SetNTrackClusters( 0 );
  fOutput->SetPointers();


  // for each slice set number of the next neighbouring slice

  int nextSlice[100], prevSlice[100];

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    nextSlice[iSlice] = iSlice + 1;
    prevSlice[iSlice] = iSlice - 1;
  }
  int mid = fgkNSlices / 2 - 1 ;
  int last = fgkNSlices - 1 ;
  if ( mid < 0 ) mid = 0; // to avoid compiler warning
  if ( last < 0 ) last = 0; //
  nextSlice[ mid ] = 0;
  prevSlice[ 0 ] = mid;
  nextSlice[ last ] = fgkNSlices / 2;
  prevSlice[ fgkNSlices/2 ] = last;

  int maxNSliceTracks = 0;
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    if ( maxNSliceTracks < fSliceNTrackInfos[iSlice] ) maxNSliceTracks = fSliceNTrackInfos[iSlice];
  }

  if ( 1 ) {// merging track segments withing one slice

    AliHLTResizableArray<AliHLTTPCCABorderTrack> bord( maxNSliceTracks*2 );

    AliHLTTPCCASliceTrackInfo *tmpT = new AliHLTTPCCASliceTrackInfo[maxNSliceTracks];
    AliHLTTPCCAClusterInfo *tmpH = new AliHLTTPCCAClusterInfo[fMaxClusterInfos];

    for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

      int nBord = 0;
      MakeBorderTracks( iSlice, 4, bord.Data(), nBord );
      MergeBorderTracks( iSlice, bord.Data(), nBord, iSlice, bord.Data(), nBord );

      int nTr = 0, nH = 0;
      int sliceFirstClusterRef = 0;
      for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
        AliHLTTPCCASliceTrackInfo &track = fTrackInfos[ fSliceTrackInfoStart[iSlice] + itr];
        if ( itr == 0 ) sliceFirstClusterRef = track.FirstClusterRef();
        track.SetPrevNeighbour( -1 );
        if ( track.NextNeighbour() == -2 ) {
          track.SetNextNeighbour( -1 );
          continue;
        }
        AliHLTTPCCASliceTrackInfo &trackNew = tmpT[nTr];
        trackNew = track;
        trackNew.SetFirstClusterRef( sliceFirstClusterRef + nH );

        for ( int ih = 0; ih < track.NClusters(); ih++ ) tmpH[nH+ih] = fClusterInfos[track.FirstClusterRef()+ih];
        nTr++;
        nH += track.NClusters();

        int jtr =  track.NextNeighbour();

        if ( jtr < 0 ) continue;
        AliHLTTPCCASliceTrackInfo &neighTrack = fTrackInfos[ fSliceTrackInfoStart[iSlice] + jtr];

        track.SetNextNeighbour( -1 );
        neighTrack.SetNextNeighbour( -2 );

        for ( int ih = 0; ih < neighTrack.NClusters(); ih++ )
          tmpH[nH+ih] = fClusterInfos[neighTrack.FirstClusterRef()+ih];

        trackNew.SetNClusters( trackNew.NClusters() + neighTrack.NClusters() );
        trackNew.SetNextNeighbour( -1 );
        nH += neighTrack.NClusters();
        if ( neighTrack.InnerParam().X() < track.InnerParam().X() ) trackNew.SetInnerParam( neighTrack.InnerParam() );
        if ( neighTrack.OuterParam().X() > track.OuterParam().X() ) trackNew.SetOuterParam( neighTrack.OuterParam() );
      }

      fSliceNTrackInfos[iSlice] = nTr;
      for ( int itr = 0; itr < nTr; itr++ ) fTrackInfos[ fSliceTrackInfoStart[iSlice] + itr] = tmpT[itr];
      for ( int ih = 0; ih < nH; ih++ ) fClusterInfos[sliceFirstClusterRef + ih] = tmpH[ih];

    }
    delete[] tmpT;
    delete[] tmpH;
  }


  //* merging tracks between slices


  // arrays for the rotated track parameters

  AliHLTTPCCABorderTrack
  *bCurr0 = new AliHLTTPCCABorderTrack[maxNSliceTracks*2],
  *bNext0 = new AliHLTTPCCABorderTrack[maxNSliceTracks*2],
  *bCurr = new AliHLTTPCCABorderTrack[maxNSliceTracks*2],
  *bNext = new AliHLTTPCCABorderTrack[maxNSliceTracks*2];

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

    int jSlice = nextSlice[iSlice];

    int nCurr0 = 0, nNext0 = 0;
    int nCurr = 0, nNext = 0;

    MakeBorderTracks( iSlice, 0, bCurr, nCurr );
    MakeBorderTracks( jSlice, 1, bNext, nNext );
    MakeBorderTracks( iSlice, 2, bCurr0, nCurr0 );
    MakeBorderTracks( jSlice, 3, bNext0, nNext0 );

    MergeBorderTracks( iSlice, bCurr0, nCurr0, jSlice, bNext0, nNext0 );
    MergeBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext );
  }

  if ( bCurr0 ) delete[] bCurr0;
  if ( bNext0 ) delete[] bNext0;
  if ( bCurr  ) delete[] bCurr;
  if ( bNext  ) delete[] bNext;


  //static int nRejected = 0;

  int nOutTracks = 0;
  int nOutTrackClusters = 0;

  AliHLTTPCCAMergedTrack *outTracks = new AliHLTTPCCAMergedTrack[fMaxTrackInfos];
  unsigned int   *outClusterId = new unsigned int [fMaxClusterInfos];
  UChar_t  *outClusterPackedAmp = new UChar_t [fMaxClusterInfos];

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

    for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {

      AliHLTTPCCASliceTrackInfo &track = fTrackInfos[fSliceTrackInfoStart[iSlice] + itr];

      if ( track.Used() ) continue;
      if ( track.PrevNeighbour() >= 0 ) continue;
      //std::cout<<"Merged track candidate, nhits "<<track.NClusters()<<std::endl;
      AliHLTTPCCATrackParam startPoint = track.InnerParam(), endPoint = track.OuterParam();
      float startAlpha = track.InnerAlpha(), endAlpha = track.OuterAlpha();

      int hits[2000];
      int firstHit = 1000;
      int nHits = 0;
      int jSlice = iSlice;
      int jtr = itr;

      {
        track.SetUsed( 1 );
        for ( int jhit = 0; jhit < track.NClusters(); jhit++ ) {
          int id = track.FirstClusterRef() + jhit;
          hits[firstHit+jhit] = id;
        }
        nHits = track.NClusters();
        jtr = track.NextNeighbour();
        jSlice = nextSlice[iSlice];
      }

      while ( jtr >= 0 ) {
        AliHLTTPCCASliceTrackInfo &segment = fTrackInfos[fSliceTrackInfoStart[jSlice] + jtr];
        if ( segment.Used() ) break;
        segment.SetUsed( 1 );
        bool dir = 0;
        int startHit = firstHit + nHits;
        float d00 = startPoint.GetDistXZ2( segment.InnerParam() );
        float d01 = startPoint.GetDistXZ2( segment.OuterParam() );
        float d10 = endPoint.GetDistXZ2( segment.InnerParam() );
        float d11 = endPoint.GetDistXZ2( segment.OuterParam() );
        if ( d00 <= d01 && d00 <= d10 && d00 <= d11 ) {
          startPoint = segment.OuterParam();
          startAlpha = segment.OuterAlpha();
          dir = 1;
          firstHit -= segment.NClusters();
          startHit = firstHit;
        } else if ( d01 <= d10 && d01 <= d11 ) {
          startPoint = segment.InnerParam();
          startAlpha = segment.InnerAlpha();
          dir = 0;
          firstHit -= segment.NClusters();
          startHit = firstHit;
        } else if ( d10 <= d11 ) {
          endPoint = segment.OuterParam();
          endAlpha = segment.OuterAlpha();
          dir = 0;
        } else {
          endPoint = segment.InnerParam();
          endAlpha = segment.InnerAlpha();
          dir = 1;
        }

        for ( int jhit = 0; jhit < segment.NClusters(); jhit++ ) {
          int id = segment.FirstClusterRef() + jhit;
          hits[startHit+( dir ?( segment.NClusters()-1-jhit ) :jhit )] = id;
        }
        nHits += segment.NClusters();
        jtr = segment.NextNeighbour();
        jSlice = nextSlice[jSlice];
      }

      if ( endPoint.X() < startPoint.X() ) { // swap
        for ( int i = 0; i < nHits; i++ ) hits[i] = hits[firstHit+nHits-1-i];
        firstHit = 0;
      }

      if ( nHits < 30 ) continue;    //SG!!!

      // refit

      // need best t0!!!SG

      endPoint = startPoint;

      if ( !FitTrack( endPoint, endAlpha, startPoint, startAlpha, hits + firstHit, nHits, 0,1 ) ) continue;
      if ( !FitTrack( startPoint, startAlpha, endPoint, endAlpha, hits + firstHit, nHits, 1,1 ) ) continue;
      if ( nHits < 30 ) continue;    //SG!!!

      AliHLTTPCCATrackParam p = startPoint;
      
      if(0){
        double xTPC = 83.65; //SG!!!
        double dAlpha = 0.349066;
	double ymax = 2.* xTPC * CAMath::Tan( dAlpha / 2. );
        
        double dRot = 0;
        if ( p.TransportToXWithMaterial( xTPC, fSliceParam.GetBz( p ) ) ) {
          double y = p.GetY();
          if ( y > ymax ) {         
            if ( p.Rotate( dAlpha ) ){
              dRot = dAlpha;
              p.TransportToXWithMaterial( xTPC, fSliceParam.GetBz( p ) );
            }
          } else if( y< -ymax ){
            if ( p.Rotate( -dAlpha ) ){
              dRot = -dAlpha;
              p.TransportToXWithMaterial( xTPC, fSliceParam.GetBz( p ) );
            }
          }
        }
        
        if ( -ymax <= p.GetY() && p.GetY() <= ymax && p.CheckNumericalQuality() ){
          startPoint = p;
          startAlpha+=dRot;
        }     
      }

      if ( !startPoint.CheckNumericalQuality() ) continue;

      AliHLTTPCCAMergedTrack &mergedTrack = outTracks[nOutTracks];
      mergedTrack.SetNClusters( nHits );
      mergedTrack.SetFirstClusterRef( nOutTrackClusters );
      mergedTrack.SetInnerParam( startPoint );
      mergedTrack.SetInnerAlpha( startAlpha );
      mergedTrack.SetOuterParam( endPoint );
      mergedTrack.SetOuterAlpha( endAlpha );

      for ( int i = 0; i < nHits; i++ ) {
        AliHLTTPCCAClusterInfo &clu = fClusterInfos[hits[firstHit+i]];
        outClusterId[nOutTrackClusters+i] = clu.Id();
        outClusterPackedAmp[nOutTrackClusters+i] = clu.PackedAmp();
      }

      nOutTracks++;
      nOutTrackClusters += nHits;
    }
  }

  fOutput->SetNTracks( nOutTracks );
  #ifdef HLTCA_STANDALONE
  printf("Tracks Output: %d\n", nOutTracks);
  #endif
  fOutput->SetNTrackClusters( nOutTrackClusters );
  fOutput->SetPointers();

  for ( int itr = 0; itr < nOutTracks; itr++ ) fOutput->SetTrack( itr, outTracks[itr] );

  for ( int ic = 0; ic < nOutTrackClusters; ic++ ) {
    fOutput->SetClusterId( ic, outClusterId[ic] );
    fOutput->SetClusterPackedAmp( ic, outClusterPackedAmp[ic] );
  }

  delete[] outTracks;
  delete[] outClusterId;
  delete[] outClusterPackedAmp;
}

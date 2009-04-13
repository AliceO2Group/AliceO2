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


#include "AliHLTTPCCASliceTrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGBTrack.h"
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


class AliHLTTPCCAMerger::AliHLTTPCCAClusterInfo
{
  
public:
  
  UInt_t  ISlice()    const { return fISlice;    }
  UInt_t  IRow()      const { return fIRow;      }
  UInt_t  IClu()      const { return fIClu;      }
  UChar_t PackedAmp() const { return fPackedAmp; }
  Float_t X()         const { return fX;         }
  Float_t Y()         const { return fY;         }
  Float_t Z()         const { return fZ;         }
  Float_t Err2Y()     const { return fErr2Y;     }
  Float_t Err2Z()     const { return fErr2Z;     }
    
  void SetISlice    ( UInt_t v  ) { fISlice    = v; }
  void SetIRow      ( UInt_t v  ) { fIRow      = v; }
  void SetIClu      ( UInt_t v  ) { fIClu      = v; }
  void SetPackedAmp ( UChar_t v ) { fPackedAmp = v; }
  void SetX         ( Float_t v ) { fX         = v; } 
  void SetY         ( Float_t v ) { fY         = v; } 
  void SetZ         ( Float_t v ) { fZ         = v; } 
  void SetErr2Y     ( Float_t v ) { fErr2Y     = v; } 
  void SetErr2Z     ( Float_t v ) { fErr2Z     = v; } 
  
private:
  
  UInt_t fISlice;            // slice number
  UInt_t fIRow;              // row number
  UInt_t fIClu;              // cluster number
  UChar_t fPackedAmp; // packed cluster amplitude
  Float_t fX;                // x position (slice coord.system)
  Float_t fY;                // y position (slice coord.system)
  Float_t fZ;                // z position (slice coord.system)
  Float_t fErr2Y;            // Squared measurement error of y position
  Float_t fErr2Z;            // Squared measurement error of z position
};


class AliHLTTPCCAMerger::AliHLTTPCCASliceTrackInfo
{

public:

  const AliHLTTPCCATrackParam &InnerParam() const { return fInnerParam;      }
  const AliHLTTPCCATrackParam &OuterParam() const { return fOuterParam;      }  
  Float_t InnerAlpha()                      const { return fInnerAlpha;      }
  Float_t OuterAlpha()                      const { return fOuterAlpha;      }
  Int_t   NClusters()                       const { return fNClusters;       }
  Int_t   FirstClusterRef()                 const { return fFirstClusterRef; }
  Int_t   PrevNeighbour()                   const { return fPrevNeighbour;   }
  Int_t   NextNeighbour()                   const { return fNextNeighbour;   }
  Bool_t  Used()                            const { return fUsed;            }
  
  void SetInnerParam( const AliHLTTPCCATrackParam &v ) { fInnerParam = v;      }
  void SetOuterParam( const AliHLTTPCCATrackParam &v ) { fOuterParam = v;      }
  void SetInnerAlpha( Float_t v )                      { fInnerAlpha = v;      }
  void SetOuterAlpha( Float_t v )                      { fOuterAlpha = v;      }
  void SetNClusters ( Int_t v )                        { fNClusters = v;       }
  void SetFirstClusterRef( Int_t v )                   { fFirstClusterRef = v; }
  void SetPrevNeighbour( Int_t v )                     { fPrevNeighbour = v;   }
  void SetNextNeighbour( Int_t v )                     { fNextNeighbour = v;   }
  void SetUsed( Bool_t v )                             { fUsed = v;            }

private:

  AliHLTTPCCATrackParam fInnerParam; // inner parameters
  AliHLTTPCCATrackParam fOuterParam; // outer parameters
  Float_t fInnerAlpha;               // alpha angle for inner parameters
  Float_t fOuterAlpha;               // alpha angle for outer parameters
  Int_t fNClusters;                  // N clusters
  Int_t fFirstClusterRef;  // index of the first track cluster in the global cluster array
  Int_t fPrevNeighbour;    // neighbour in the previous slise
  Int_t fNextNeighbour;    // neighbour in the next slise
  Bool_t fUsed;            // is the slice track already merged

};


class AliHLTTPCCAMerger::AliHLTTPCCABorderTrack
{

public:
 
  const AliHLTTPCCATrackParam &Param() const { return fParam;     }
  Int_t   TrackID()                    const { return fTrackID;   }
  Int_t   NClusters()                  const { return fNClusters; }
  Int_t   IRow()                       const { return fIRow;      }
  Float_t X()                          const { return fX;         }
  Bool_t  OK()                         const { return fOK;        }
  
  void SetParam     ( const AliHLTTPCCATrackParam &v ) { fParam     = v; }
  void SetTrackID   ( Int_t v )                        { fTrackID   = v; }
  void SetNClusters ( Int_t v )                        { fNClusters = v; }
  void SetIRow      ( Int_t v )                        { fIRow      = v; }
  void SetX         ( Float_t v )                      { fX         = v; }
  void SetOK        ( Bool_t v )                       { fOK        = v; }

private:

  AliHLTTPCCATrackParam fParam;  // track parameters at the border
  Int_t   fTrackID;              // track index
  Int_t   fNClusters;            // n clusters
  Int_t   fIRow;                 // row number of the closest cluster
  Float_t fX;                    // X coordinate of the closest cluster
  Bool_t  fOK;                   // is the track rotated and extrapolated correctly

};



AliHLTTPCCAMerger::AliHLTTPCCAMerger()
  :
  fSliceParam(),
  fOutput(0),  
  fTrackInfos(0),
  fMaxTrackInfos(0),
  fClusterInfos(0),
  fMaxClusterInfos(0)
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
  if( fTrackInfos ) delete[] fTrackInfos;
  if( fClusterInfos ) delete[] fClusterInfos;
  if( fOutput ) delete[] ((char*)(fOutput));
}

void AliHLTTPCCAMerger::Clear()
{
  for ( int i = 0; i < fgkNSlices; ++i ) {
    fkSlices[i] = 0;
  }
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
  
  Int_t nTracksTotal=0;
  Int_t nTrackClustersTotal=0;
  for( Int_t iSlice=0; iSlice<fgkNSlices; iSlice++ ){
    if( !fkSlices[iSlice] ) continue;
    nTracksTotal+=fkSlices[iSlice]->NTracks();
    nTrackClustersTotal+=fkSlices[iSlice]->NTrackClusters();    
  }

  // book/clean memory if necessary
  {
    if( nTracksTotal>fMaxTrackInfos || ( fMaxTrackInfos>100 && nTracksTotal< 0.5*fMaxTrackInfos ) ){
      if( fTrackInfos ) delete[] fTrackInfos;
      fMaxTrackInfos = (Int_t ) (nTracksTotal*1.2);
      fTrackInfos = new AliHLTTPCCASliceTrackInfo[fMaxTrackInfos];
    }

    if( nTrackClustersTotal>fMaxClusterInfos || ( fMaxClusterInfos>1000 && nTrackClustersTotal< 0.5*fMaxClusterInfos ) ){
      if( fClusterInfos ) delete[] fClusterInfos;
      fMaxClusterInfos = (Int_t ) (nTrackClustersTotal*1.2);
      fClusterInfos = new AliHLTTPCCAClusterInfo [fMaxClusterInfos];
    }

    if( fOutput ) delete[] ( (char*)(fOutput));
    Int_t size = fOutput->EstimateSize(nTracksTotal, nTrackClustersTotal);
    fOutput = (AliHLTTPCCAMergerOutput*)(new float2[size/sizeof(float2)+1]);
  }

  // unpack track and cluster information
  
  Int_t nTracksCurrent = 0; 
  Int_t nClustersCurrent = 0;

  for( Int_t iSlice=0; iSlice<fgkNSlices; iSlice++ ){

    fSliceTrackInfoStart[ iSlice ] = nTracksCurrent;
    fSliceNTrackInfos[ iSlice ] = 0;
 
    if( !fkSlices[iSlice] ) continue;

    const AliHLTTPCCASliceOutput &slice = *(fkSlices[iSlice]);

    for( Int_t itr=0; itr<slice.NTracks(); itr++ ){

      const AliHLTTPCCASliceTrack &sTrack = slice.Track( itr );
      AliHLTTPCCATrackParam t0 = sTrack.Param();
      Int_t nCluNew = 0;
      
      for( Int_t iTrClu=0; iTrClu<sTrack.NClusters(); iTrClu++ ){

	// unpack cluster information

	AliHLTTPCCAClusterInfo &clu = fClusterInfos[nClustersCurrent + nCluNew];
	Int_t ic = sTrack.FirstClusterRef() + iTrClu;

	clu.SetISlice( iSlice );
	clu.SetIRow( AliHLTTPCCADataCompressor::IDrc2IRow( slice.ClusterIDrc( ic ) ) );
	clu.SetIClu( AliHLTTPCCADataCompressor::IDrc2IClu( slice.ClusterIDrc( ic ) ) );	
	clu.SetPackedAmp( slice.ClusterPackedAmp( ic ) );
	float2 yz = slice.ClusterUnpackedYZ( ic );
	clu.SetX( slice.ClusterUnpackedX( ic ) );
	clu.SetY( yz.x );
	clu.SetZ( yz.y );

	if( !t0.TransportToX( fSliceParam.RowX( clu.IRow() ), fSliceParam.GetBz(t0), .999 ) ) continue;

	Float_t err2Y, err2Z;
	fSliceParam.GetClusterErrors2( clu.IRow(), clu.Z(), t0.SinPhi(), t0.GetCosPhi(), t0.DzDs(), err2Y, err2Z );
	
	clu.SetErr2Y( err2Y );
	clu.SetErr2Z( err2Z );
	nCluNew++ ;
      }

      if( nCluNew<.8*sTrack.NClusters() ) continue;
      
      // refit the track 
      
      Int_t hits[1000];
      Int_t nHits = nCluNew; 
      for( Int_t i=0; i<nHits; i++ ) hits[i] = nClustersCurrent + i;

      AliHLTTPCCATrackParam startPoint = sTrack.Param();
      AliHLTTPCCATrackParam endPoint = startPoint;
      Float_t startAlpha = fSliceParam.Alpha( iSlice );
      Float_t endAlpha = startAlpha;
 
      if( !FitTrack( endPoint, endAlpha, startPoint, startAlpha, hits, nHits, 0 ) ) continue;

      startPoint = endPoint;
      startAlpha = endAlpha;
      if( !FitTrack( startPoint, startAlpha, endPoint, endAlpha, hits, nHits, 1 ) ) continue;

      if( nHits<.8*sTrack.NClusters() ) continue;

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
      
      for( Int_t i=0; i<nHits; i++ ) 
	fClusterInfos[nClustersCurrent + i] = fClusterInfos[hits[i]];
      nTracksCurrent++;
      fSliceNTrackInfos[ iSlice ]++;
      nClustersCurrent+=nHits;      
    }
    //std::cout<<"Unpack slice "<<iSlice<<": ntracks "<<slice.NTracks()<<"/"<<fSliceNTrackInfos[iSlice]<<std::endl;
  }
}



Bool_t AliHLTTPCCAMerger::FitTrack( AliHLTTPCCATrackParam &T, Float_t &Alpha, 
				    AliHLTTPCCATrackParam t0, Float_t Alpha0, 
				    Int_t hits[], Int_t &NTrackHits, Bool_t dir )
{
  // Fit the track

  AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
  AliHLTTPCCATrackParam t = t0;
  AliHLTTPCCATrackLinearisation l(t0);

  Bool_t first = 1;
 
  t.CalculateFitParameters( fitPar );

  Int_t hitsNew[1000];
  Int_t nHitsNew = 0;

  for( Int_t ihit=0; ihit<NTrackHits; ihit++){
    
    Int_t jhit = dir ?(NTrackHits-1-ihit) :ihit;
    AliHLTTPCCAClusterInfo &h = fClusterInfos[hits[jhit]];

    Int_t iSlice = h.ISlice();

    Float_t sliceAlpha =  fSliceParam.Alpha( iSlice );

    if( CAMath::Abs( sliceAlpha - Alpha0)>1.e-4 ){
      if( ! t.Rotate(  sliceAlpha - Alpha0, l, .999 ) ) continue;
      Alpha0 = sliceAlpha;
    }

    //Float_t x = fSliceParam.RowX( h.IRow() );
    Float_t x = h.X();
    
    if( !t.TransportToXWithMaterial( x, l, fitPar, fSliceParam.GetBz(t) ) ) continue;
    
    if( first ){
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
      t.SetCov(10,  0 );
      t.SetCov(11,  0 );
      t.SetCov(12,  0 );
      t.SetCov(13,  0 );
      t.SetCov(14,  10 );
      t.SetChi2( 0 );
      t.SetNDF( -5 );
      t.CalculateFitParameters( fitPar );
    }
  
    if( !t.Filter( h.Y(), h.Z(), h.Err2Y(), h.Err2Z() ) ) continue;	  	    

    first = 0;

    hitsNew[nHitsNew++] = hits[jhit];
  }
  
  if( CAMath::Abs(t.QPt())<1.e-8 ) t.SetQPt( 1.e-8 );
  
  Bool_t ok=1;
  
  const Float_t *c = t.Cov();
  for( Int_t i=0; i<15; i++ ) ok = ok && finite(c[i]);
  for( Int_t i=0; i<5; i++ ) ok = ok && finite(t.Par()[i]);
  ok = ok && (t.GetX()>50);
  
  if( c[0]<=0 || c[2]<=0 || c[5]<=0 || c[9]<=0 || c[14]<=0 ) ok = 0;
  if( c[0]>5. || c[2]>5. || c[5]>2. || c[9]>2 || c[14]>2. ) ok = 0;
  
  if( CAMath::Abs(t.SinPhi())>.99 ) ok = 0;
  else if( l.CosPhi()>=0 ) t.SetSignCosPhi( 1 );
  else t.SetSignCosPhi( -1 );

  if( ok ){
    T = t;
    Alpha = Alpha0;
    NTrackHits = nHitsNew;
    for( Int_t i=0; i<NTrackHits; i++ ){
      hits[dir ?(NTrackHits-1-i) :i] = hitsNew[i];
    }
  }
  return ok;
}


Float_t AliHLTTPCCAMerger::GetChi2( Float_t x1, Float_t y1, Float_t a00, Float_t a10, Float_t a11, 
				    Float_t x2, Float_t y2, Float_t b00, Float_t b10, Float_t b11  )
{
  //* Calculate Chi2/ndf deviation   

  Float_t d[2]={ x1-x2, y1-y2 };

  Float_t mSi[3] = { a00 + b00, a10 + b10, a11 + b11 };

  Float_t s = ( mSi[0]*mSi[2] - mSi[1]*mSi[1] );

  if( s < 1.E-10 ) return 10000.;
    
  Float_t mS[3] = { mSi[2], -mSi[1], mSi[0] };      

  return TMath::Abs( ( ( mS[0]*d[0] + mS[1]*d[1] )*d[0]
		       +(mS[1]*d[0] + mS[2]*d[1] )*d[1] )/s/2);

}

  

void AliHLTTPCCAMerger::MakeBorderTracks( Int_t iSlice, Int_t iBorder, AliHLTTPCCABorderTrack B[], Int_t &nB )
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line, 
  //* in some cases both inner and outer parameters of the track are transported 

  static int statAll=0, statOK=0;  
  nB = 0;  
  Float_t dAlpha = fSliceParam.DAlpha() /2;
  Float_t x0 = 0;

  if( iBorder==0 ){ // transport to the left age of the sector and rotate horisontally
    dAlpha = dAlpha - CAMath::Pi()/2 ;
  } else if( iBorder==1 ){ //  transport to the right age of the sector and rotate horisontally
    dAlpha = -dAlpha - CAMath::Pi()/2 ;  
  } else if( iBorder==2 ){ // transport to the left age of the sector and rotate vertically
    dAlpha = dAlpha;
    x0 = fSliceParam.RowX( 63 );
  }else if( iBorder==3 ){ // transport to the right age of the sector and rotate vertically
    dAlpha = -dAlpha;
    x0 =  fSliceParam.RowX( 63 );
  } else if( iBorder==4 ){ // transport to the middle of the sector, w/o rotation
    dAlpha = 0;
    x0 = fSliceParam.RowX(63);
  }

  for (Int_t itr=0; itr<fSliceNTrackInfos[iSlice]; itr++) {

    const AliHLTTPCCASliceTrackInfo &track = fTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];

    AliHLTTPCCATrackParam t0 = track.InnerParam();
    AliHLTTPCCATrackParam t1 = track.OuterParam();

    const Float_t maxSin = CAMath::Sin(60./180.*CAMath::Pi());

    Bool_t ok0 = t0.Rotate( dAlpha, maxSin );
    Bool_t ok1 = t1.Rotate( dAlpha, maxSin );
    
    Bool_t do0 = ok0;
    Bool_t do1 = ok1 && ( !ok0 || t1.SignCosPhi()*t0.SignCosPhi()<0 );
    
    if( ok0 && !do1 && ok1 && (t1.X() < t0.X()) ){
      do0 = 0;
      do1 = 1;
    }

    if( do0 ){
      AliHLTTPCCABorderTrack &b = B[nB];
      b.SetX( t0.GetX() );
      if( t0.TransportToX( x0, fSliceParam.GetBz(t0), maxSin ) ){
	b.SetOK( 1 );
	b.SetTrackID( itr );
	b.SetNClusters( track.NClusters() );
	b.SetIRow( fClusterInfos[ track.FirstClusterRef() + 0 ].IRow() );
	b.SetParam( t0 );
 	nB++;
      }
    }
    if( do1 ){
      AliHLTTPCCABorderTrack &b = B[nB];
      b.SetX( t1.GetX() );
      if( t1.TransportToX( x0, fSliceParam.GetBz(t1), maxSin ) ){
	b.SetOK( 1 );
	b.SetTrackID( itr );
	b.SetNClusters( track.NClusters() );
	b.SetIRow( fClusterInfos[ track.FirstClusterRef() + track.NClusters()-1 ].IRow() );
	b.SetParam( t1 );  
	nB++;
      }
    }
    if( do0 || do1 ) statOK++;
    statAll++;
  }
}



void AliHLTTPCCAMerger::SplitBorderTracks( Int_t iSlice1, AliHLTTPCCABorderTrack B1[], Int_t N1,
					   Int_t iSlice2, AliHLTTPCCABorderTrack B2[], Int_t N2 
					   )
{
  //* split two sets of tracks

  Float_t factor2ys = 1.;//1.5;//SG!!!
  Float_t factor2zt = 1.;//1.5;//SG!!!
  Float_t factor2k = 2.0;//2.2;

  factor2k  = 3.5*3.5*factor2k*factor2k;
  factor2ys = 3.5*3.5*factor2ys*factor2ys;
  factor2zt = 3.5*3.5*factor2zt*factor2zt;

  Int_t minNPartHits = 10;//SG!!!
  Int_t minNTotalHits = 20;

  //Float_t maxDX = fSliceParam.RowX(40) -  fSliceParam.RowX(0);

  for (Int_t i1=0; i1<N1; i1++) {      
    AliHLTTPCCABorderTrack &b1 = B1[i1];
    if( !b1.OK() ) continue;
    if( b1.NClusters() < minNPartHits ) continue;
    const AliHLTTPCCATrackParam &t1 = b1.Param();
    Int_t iBest2 = -1;
    Int_t lBest2 = 0;
    Int_t start2 = (iSlice1!=iSlice2) ?0 :i1+1;
    for (Int_t i2=start2; i2<N2; i2++) {
      AliHLTTPCCABorderTrack &b2 = B2[i2];
      if( !b2.OK() ) continue;
      if( b2.NClusters() < minNPartHits ) continue;
      if( b2.NClusters() < lBest2 ) continue;
      if( b1.NClusters() + b2.NClusters() < minNTotalHits ) continue;

      //if( TMath::Abs(b1.fX - b2.fX)>maxDX ) continue;

      const AliHLTTPCCATrackParam &t2 = b2.Param();

      Float_t c= t2.SignCosPhi()*t1.SignCosPhi()>=0 ?1 :-1;
      Float_t dk = t2.QPt() - c*t1.QPt(); 
      Float_t s2k = t2.Err2QPt() + t1.Err2QPt();
      
      if( dk*dk>factor2k*s2k ) continue;     


      Float_t chi2ys = GetChi2( t1.Y(),c*t1.SinPhi(),t1.Cov()[0],c*t1.Cov()[3],t1.Cov()[5], 
				t2.Y(),  t2.SinPhi(),t2.Cov()[0],  t2.Cov()[3],t2.Cov()[5] );

      if( chi2ys>factor2ys ) continue;

      Float_t chi2zt = GetChi2( t1.Z(),c*t1.DzDs(),t1.Cov()[2],c*t1.Cov()[7],t1.Cov()[9], 
				t2.Z(),  t2.DzDs(),t2.Cov()[2],  t2.Cov()[7],t2.Cov()[9] );
            
      if( chi2zt>factor2zt ) continue;
      
      lBest2 = b2.NClusters();
      iBest2 = b2.TrackID();
    }
      
    if( iBest2 <0 ) continue;

    AliHLTTPCCASliceTrackInfo &newTrack1 = fTrackInfos[fSliceTrackInfoStart[iSlice1]+b1.TrackID() ];    
    AliHLTTPCCASliceTrackInfo &newTrack2 = fTrackInfos[fSliceTrackInfoStart[iSlice2]+iBest2 ];

    Int_t old1 = newTrack2.PrevNeighbour();
  
    if( old1 >= 0 ){
      AliHLTTPCCASliceTrackInfo &oldTrack1 = fTrackInfos[fSliceTrackInfoStart[iSlice1]+old1];
      if( oldTrack1.NClusters()  < newTrack1.NClusters() ){
	newTrack2.SetPrevNeighbour(-1);
	oldTrack1.SetNextNeighbour(-1);	
      } else continue;
    }
    Int_t old2 = newTrack1.NextNeighbour();
    if( old2 >= 0 ){
      AliHLTTPCCASliceTrackInfo &oldTrack2 = fTrackInfos[fSliceTrackInfoStart[iSlice2]+old2];
      if( oldTrack2.NClusters() < newTrack2.NClusters() ){
	oldTrack2.SetPrevNeighbour(-1);
      } else continue;
    }
    newTrack1.SetNextNeighbour(iBest2); 
    newTrack2.SetPrevNeighbour(b1.TrackID());	
  }  
  
}


void AliHLTTPCCAMerger::Merging()
{
  //* track merging between slices

  fOutput->SetNTracks( 0 );
  fOutput->SetNTrackClusters( 0 );
  fOutput->SetPointers();


  // for each slice set number of the next neighbouring slice
     
  Int_t nextSlice[100], prevSlice[100];

  for( Int_t iSlice=0; iSlice<fgkNSlices; iSlice++ ){
    nextSlice[iSlice] = iSlice + 1;
    prevSlice[iSlice] = iSlice - 1;
  }
  Int_t mid = fgkNSlices/2 - 1 ;
  Int_t last = fgkNSlices - 1 ;
  if( mid<0 ) mid = 0; // to avoid compiler warning
  if( last<0 ) last = 0; // 
  nextSlice[ mid ] = 0;
  prevSlice[ 0 ] = mid;
  nextSlice[ last ] = fgkNSlices/2;
  prevSlice[ fgkNSlices/2 ] = last;
  
  Int_t maxNSliceTracks = 0;
  for( Int_t iSlice=0; iSlice<fgkNSlices; iSlice++ ){
    if( maxNSliceTracks < fSliceNTrackInfos[iSlice] ) maxNSliceTracks = fSliceNTrackInfos[iSlice];
  }
  
  if(1){// merging track segments withing one slice 
    
    AliHLTTPCCABorderTrack bord[maxNSliceTracks*10];
    
    AliHLTTPCCASliceTrackInfo *tmpT = new AliHLTTPCCASliceTrackInfo[maxNSliceTracks];
    AliHLTTPCCAClusterInfo *tmpH = new AliHLTTPCCAClusterInfo[fMaxClusterInfos];

    for( Int_t iSlice=0; iSlice<fgkNSlices; iSlice++ ){

      Int_t nBord=0;
      MakeBorderTracks( iSlice, 4, bord, nBord );   
      SplitBorderTracks( iSlice, bord, nBord, iSlice, bord, nBord );    

      Int_t nTr=0, nH=0;
      Int_t sliceFirstClusterRef = 0;
      for( Int_t itr=0; itr<fSliceNTrackInfos[iSlice]; itr++ ){
	AliHLTTPCCASliceTrackInfo &track = fTrackInfos[ fSliceTrackInfoStart[iSlice]+itr];
	if( itr==0 ) sliceFirstClusterRef = track.FirstClusterRef();
	track.SetPrevNeighbour( -1 );
	if( track.NextNeighbour() == -2 ){
	  track.SetNextNeighbour( -1 );
	  continue;
	}
	AliHLTTPCCASliceTrackInfo &trackNew = tmpT[nTr];
	trackNew = track;
	trackNew.SetFirstClusterRef( sliceFirstClusterRef + nH);

	for( Int_t ih=0; ih<track.NClusters(); ih++ ) tmpH[nH+ih] = fClusterInfos[track.FirstClusterRef()+ih];
	nTr++;
	nH+=track.NClusters();

	int jtr =  track.NextNeighbour();

	if( jtr<0 ) continue;
	AliHLTTPCCASliceTrackInfo &neighTrack = fTrackInfos[ fSliceTrackInfoStart[iSlice]+jtr];
	
	track.SetNextNeighbour(-1);
	neighTrack.SetNextNeighbour(-2);

	for( Int_t ih=0; ih<neighTrack.NClusters(); ih++ ) 
	  tmpH[nH+ih] = fClusterInfos[neighTrack.FirstClusterRef()+ih];

	trackNew.SetNClusters( trackNew.NClusters() + neighTrack.NClusters() );
	trackNew.SetNextNeighbour( -1 );
	nH+=neighTrack.NClusters();
	if( neighTrack.InnerParam().X() < track.InnerParam().X() ) trackNew.SetInnerParam( neighTrack.InnerParam());
	if( neighTrack.OuterParam().X() > track.OuterParam().X() ) trackNew.SetOuterParam( neighTrack.OuterParam());
      }
      
      fSliceNTrackInfos[iSlice] = nTr;
      for( Int_t itr=0; itr<nTr; itr++ ) fTrackInfos[ fSliceTrackInfoStart[iSlice]+itr] = tmpT[itr];
      for( Int_t ih=0; ih<nH; ih++ ) fClusterInfos[sliceFirstClusterRef + ih] = tmpH[ih];

    }
    delete[] tmpT;
    delete[] tmpH;
  }


  //* merging tracks between slices      

  
  // arrays for the rotated track parameters

  AliHLTTPCCABorderTrack 
    *bCurr0 = new AliHLTTPCCABorderTrack[maxNSliceTracks*10], 
    *bNext0 = new AliHLTTPCCABorderTrack[maxNSliceTracks*10],
    *bCurr = new AliHLTTPCCABorderTrack[maxNSliceTracks*10], 
    *bNext = new AliHLTTPCCABorderTrack[maxNSliceTracks*10];

  for( Int_t iSlice=0; iSlice<fgkNSlices; iSlice++ ){
    
    Int_t jSlice = nextSlice[iSlice];
    
    Int_t nCurr0 = 0, nNext0 = 0;
    Int_t nCurr = 0, nNext = 0;

    MakeBorderTracks( iSlice, 0, bCurr, nCurr );
    MakeBorderTracks( jSlice, 1, bNext, nNext );
    MakeBorderTracks( iSlice, 2, bCurr0, nCurr0 );
    MakeBorderTracks( jSlice, 3, bNext0, nNext0 );
    
    SplitBorderTracks( iSlice, bCurr0, nCurr0, jSlice, bNext0, nNext0 );   
    SplitBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext );    
  }

  if( bCurr0 ) delete[] bCurr0;
  if( bNext0 ) delete[] bNext0;
  if( bCurr  ) delete[] bCurr;
  if( bNext  ) delete[] bNext;  


  //static Int_t nRejected = 0;
    
  Int_t nOutTracks = 0;
  Int_t nOutTrackClusters = 0;

  AliHLTTPCCAMergedTrack *outTracks = new AliHLTTPCCAMergedTrack[fMaxTrackInfos];
  UInt_t   *outClusterIDsrc = new UInt_t [fMaxClusterInfos];
  UChar_t  *outClusterPackedAmp = new UChar_t [fMaxClusterInfos];
  
  for( Int_t iSlice = 0; iSlice<fgkNSlices; iSlice++ ){

    for( Int_t itr=0; itr<fSliceNTrackInfos[iSlice]; itr++ ){

      AliHLTTPCCASliceTrackInfo &track = fTrackInfos[fSliceTrackInfoStart[iSlice]+itr];

      if( track.Used() ) continue;
      if( track.PrevNeighbour()>=0 ) continue;

      AliHLTTPCCATrackParam startPoint = track.InnerParam(), endPoint = track.OuterParam();
      Float_t startAlpha = track.InnerAlpha(), endAlpha = track.OuterAlpha();      

      Int_t hits[2000];
      Int_t firstHit = 1000;
      Int_t nHits = 0;
      Int_t jSlice = iSlice;
      Int_t jtr = itr;

      {
	track.SetUsed( 1 );
	for( Int_t jhit=0; jhit<track.NClusters(); jhit++){
	  Int_t id = track.FirstClusterRef() + jhit;
	  hits[firstHit+jhit] = id;
	}
	nHits=track.NClusters();
	jtr = track.NextNeighbour();
	jSlice = nextSlice[iSlice];			
      }

      while( jtr >=0 ){
	AliHLTTPCCASliceTrackInfo &segment = fTrackInfos[fSliceTrackInfoStart[jSlice]+jtr];
	if( segment.Used() ) break;
	segment.SetUsed( 1 );
	Bool_t dir = 0;
	Int_t startHit = firstHit+ nHits;
	Float_t d00 = startPoint.GetDistXZ2(segment.InnerParam() );
	Float_t d01 = startPoint.GetDistXZ2(segment.OuterParam() );
	Float_t d10 = endPoint.GetDistXZ2(segment.InnerParam() );
	Float_t d11 = endPoint.GetDistXZ2(segment.OuterParam() );
	if( d00<=d01 && d00<=d10 && d00<=d11 ){
	  startPoint = segment.OuterParam();
	  startAlpha = segment.OuterAlpha();
	  dir = 1;
	  firstHit -= segment.NClusters();
	  startHit = firstHit;
	}else if( d01<=d10 && d01<=d11 ){
	  startPoint = segment.InnerParam();
	  startAlpha = segment.InnerAlpha();
	  dir = 0;
	  firstHit -= segment.NClusters();
	  startHit = firstHit;
	}else if( d10<=d11 ){
	  endPoint = segment.OuterParam();
	  endAlpha = segment.OuterAlpha();
	  dir = 0;
	}else{
	  endPoint = segment.InnerParam();
	  endAlpha = segment.InnerAlpha();
	  dir = 1;
	}
	
	for( Int_t jhit=0; jhit<segment.NClusters(); jhit++){
	  Int_t id = segment.FirstClusterRef() + jhit;
	  hits[startHit+(dir ?(segment.NClusters()-1-jhit) :jhit)] = id;
	}
	nHits+=segment.NClusters();
	jtr = segment.NextNeighbour();
	jSlice = nextSlice[jSlice];			
      }

      if( endPoint.X() < startPoint.X() ){ // swap
	for( Int_t i=0; i<nHits; i++ ) hits[i] = hits[firstHit+nHits-1-i];
	firstHit = 0;
      }
      
      if( nHits < 30 ) continue;     //SG!!!

      // refit 

      // need best t0!!!SG

      endPoint = startPoint;
      if( !FitTrack( endPoint, endAlpha, startPoint, startAlpha, hits+firstHit, nHits, 0 ) ) continue;
      if( !FitTrack( startPoint, startAlpha, endPoint, endAlpha, hits+firstHit, nHits, 1 ) ) continue;

      if( nHits < 30 ) continue;     //SG!!!    
      
      AliHLTTPCCATrackParam &p = startPoint;
      
      {
	Double_t xTPC=83.65; //SG!!!
	Double_t dAlpha = 0.00609235;
	AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
	p.CalculateFitParameters( fitPar );
	
	if( p.TransportToXWithMaterial( xTPC, fitPar, fSliceParam.GetBz( p ) ) ){
	  Double_t y=p.GetY();
	  Double_t ymax=xTPC*CAMath::Tan(dAlpha/2.); 
	  if (y > ymax) {
	    if( p.Rotate( dAlpha ) ){ startAlpha+=dAlpha;  p.TransportToXWithMaterial( xTPC, fitPar, fSliceParam.GetBz(p) ); }
	  } else if (y <-ymax) {
	    if( p.Rotate( -dAlpha ) ){  startAlpha-=dAlpha; p.TransportToXWithMaterial( xTPC, fitPar, fSliceParam.GetBz(p) );}
	  }
	}
      }
      
      {
	Bool_t ok=1;
	
	const Float_t *c = p.Cov();
	for( Int_t i=0; i<15; i++ ) ok = ok && finite(c[i]);
	for( Int_t i=0; i<5; i++ ) ok = ok && finite(p.Par()[i]);
	ok = ok && (p.GetX()>50);
	
	if( c[0]<=0 || c[2]<=0 || c[5]<=0 || c[9]<=0 || c[14]<=0 ) ok = 0;
	if( c[0]>5. || c[2]>5. || c[5]>2. || c[9]>2 || c[14]>2 ) ok = 0;
	if(!ok) continue;	
      }

      AliHLTTPCCAMergedTrack &mergedTrack = outTracks[nOutTracks];
      mergedTrack.SetNClusters(nHits);
      mergedTrack.SetFirstClusterRef(nOutTrackClusters);
      mergedTrack.SetInnerParam(startPoint);
      mergedTrack.SetInnerAlpha(startAlpha);
      mergedTrack.SetOuterParam(endPoint);
      mergedTrack.SetOuterAlpha(endAlpha);
 
      for( Int_t i = 0; i<nHits; i++ ){
	AliHLTTPCCAClusterInfo &clu = fClusterInfos[hits[firstHit+i]];
	outClusterIDsrc[nOutTrackClusters+i] = 
	  AliHLTTPCCADataCompressor::ISliceIRowIClu2IDsrc(clu.ISlice(), clu.IRow(), clu.IClu());
	outClusterPackedAmp[nOutTrackClusters+i]=clu.PackedAmp();
      }
      
      nOutTracks++;
      nOutTrackClusters+= nHits;
    }
  }

  fOutput->SetNTracks( nOutTracks );
  fOutput->SetNTrackClusters( nOutTrackClusters );
  fOutput->SetPointers();

  for( Int_t itr=0; itr<nOutTracks; itr++ ) fOutput->SetTrack( itr, outTracks[itr] );

  for( Int_t ic=0; ic<nOutTrackClusters; ic++ ){
    fOutput->SetClusterIDsrc( ic, outClusterIDsrc[ic] );
    fOutput->SetClusterPackedAmp( ic, outClusterPackedAmp[ic] );
  }
 
  delete[] outTracks;
  delete[] outClusterIDsrc;
  delete[] outClusterPackedAmp;
}

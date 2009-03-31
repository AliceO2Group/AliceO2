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


AliHLTTPCCAMerger::AliHLTTPCCAMerger()
  :
  fSliceParam(),
  fkSlices(0),
  fOutput(0),  
  fTrackInfos(0),
  fMaxTrackInfos(0),
  fClusterInfos(0),
  fMaxClusterInfos(0)
{
  //* constructor
}


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
  //* dummy
}

const AliHLTTPCCAMerger &AliHLTTPCCAMerger::operator=(const AliHLTTPCCAMerger&) const
{
  //* dummy
  return *this;
}

AliHLTTPCCAMerger::~AliHLTTPCCAMerger()
{
  //* destructor
  if( fTrackInfos ) delete[] fTrackInfos;
  if( fClusterInfos ) delete[] fClusterInfos;
  if( fOutput ) delete[] ((char*)(fOutput));
}


void AliHLTTPCCAMerger::Reconstruct( const AliHLTTPCCASliceOutput **SliceOutput )
{
  //* main merging routine

  fkSlices = SliceOutput;
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

    const AliHLTTPCCASliceOutput &slice = *(fkSlices[iSlice]);
    fSliceTrackInfoStart[ iSlice ] = nTracksCurrent;
    fSliceNTrackInfos[ iSlice ] = 0;

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
	clu.SetY( yz.x );
	clu.SetZ( yz.y );

	if( !t0.TransportToX( fSliceParam.RowX( clu.IRow() ), .999 ) ) continue;
	Float_t err2Y, err2Z;
	fSliceParam.GetClusterErrors2( clu.IRow(), clu.Z(), t0.SinPhi(), t0.CosPhi(), t0.DzDs(), err2Y, err2Z );
	
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
      track.fInnerParam = startPoint;
      track.fInnerAlpha = startAlpha;
      track.fOuterParam = endPoint;
      track.fOuterAlpha = endAlpha;
      track.fFirstClusterRef = nClustersCurrent;
      track.fNClusters = nHits;
      track.fPrevNeighbour = -1;
      track.fNextNeighbour = -1;
      track.fUsed = 0;
      
      for( Int_t i=0; i<nHits; i++ ) 
	fClusterInfos[nClustersCurrent + i] = fClusterInfos[hits[i]];
      nTracksCurrent++;
      fSliceNTrackInfos[ iSlice ]++;
      nClustersCurrent+=nHits;      
    }
  }
}



Bool_t AliHLTTPCCAMerger::FitTrack( AliHLTTPCCATrackParam &T, Float_t &Alpha, 
				    AliHLTTPCCATrackParam t0, Float_t Alpha0, 
				    Int_t hits[], Int_t &NTrackHits, Bool_t dir )
{
  // Fit the track

  AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
  AliHLTTPCCATrackParam t = t0;
  
  Bool_t first = 1;
 
  t0.CalculateFitParameters( fitPar, fSliceParam.Bz() );

  Int_t hitsNew[1000];
  Int_t nHitsNew = 0;

  for( Int_t ihit=0; ihit<NTrackHits; ihit++){
    Int_t jhit = dir ?(NTrackHits-1-ihit) :ihit;
    AliHLTTPCCAClusterInfo &h = fClusterInfos[hits[jhit]];
    Int_t iSlice = h.ISlice();

    Float_t sliceAlpha =  fSliceParam.Alpha( iSlice );

    if( CAMath::Abs( sliceAlpha - Alpha0)>1.e-4 ){
      if( ! t.RotateNoCos(  sliceAlpha - Alpha0, t0, .999 ) ) continue;
      Alpha0 = sliceAlpha;
    }

    Float_t x = fSliceParam.RowX( h.IRow() );
    
    if( !t.TransportToXWithMaterial( x, t0, fitPar ) ) continue;

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
      t.SetCov(14,  1 );
      t.SetChi2( 0 );
      t.SetNDF( -5 );
      t0.CalculateFitParameters( fitPar, fSliceParam.Bz() );
    }
  
    if( !t.Filter2NoCos( h.Y(), h.Z(), h.Err2Y(), h.Err2Z() ) ) continue;	  	    
    first = 0;

    hitsNew[nHitsNew++] = hits[jhit];
  }
  

  if( CAMath::Abs(t.Kappa())<1.e-8 ) t.SetKappa( 1.e-8 );
  
  Bool_t ok=1;
  
  const Float_t *c = t.Cov();
  for( Int_t i=0; i<15; i++ ) ok = ok && finite(c[i]);
  for( Int_t i=0; i<5; i++ ) ok = ok && finite(t.Par()[i]);
  ok = ok && (t.GetX()>50);
  
  if( c[0]<=0 || c[2]<=0 || c[5]<=0 || c[9]<=0 || c[14]<=0 ) ok = 0;
  if( c[0]>5. || c[2]>5. || c[5]>2. || c[9]>2 || c[14]>2 ) ok = 0;
  
  if( CAMath::Abs(t.SinPhi())>.99 ) ok = 0;
  else if( t0.CosPhi()>=0 ) t.SetCosPhi( CAMath::Sqrt(1.-t.SinPhi()*t.SinPhi()) );
  else t.SetCosPhi( -CAMath::Sqrt(1.-t.SinPhi()*t.SinPhi()) );

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

    AliHLTTPCCASliceTrackInfo &track = fTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];

    AliHLTTPCCATrackParam t0 = track.fInnerParam;
    AliHLTTPCCATrackParam t1 = track.fOuterParam;

    const Float_t maxSin = CAMath::Sin(60./180.*CAMath::Pi());

    Bool_t ok0 = t0.Rotate( dAlpha, maxSin );
    Bool_t ok1 = t1.Rotate( dAlpha, maxSin );
    
    Bool_t do0 = ok0;
    Bool_t do1 = ok1 && ( !ok0 || t1.CosPhi()*t0.CosPhi()<0 );
    
    if( ok0 && !do1 && ok1 && (t1.X() < t0.X()) ){
      do0 = 0;
      do1 = 1;
    }

    if( do0 ){
      AliHLTTPCCABorderTrack &b = B[nB];
      b.fOK = 1;
      b.fITrack = itr;
      b.fNHits = track.fNClusters;
      b.fIRow = fClusterInfos[ track.fFirstClusterRef + 0 ].IRow();
      b.fParam = t0;
      b.fX = t0.GetX();
      if( b.fParam.TransportToX( x0, maxSin ) ) nB++;
    }
    if( do1 ){
      AliHLTTPCCABorderTrack &b = B[nB];
      b.fOK = 1;
      b.fITrack = itr;
      b.fNHits = track.fNClusters;
      b.fIRow = fClusterInfos[ track.fFirstClusterRef + track.fNClusters-1 ].IRow();
      b.fParam = t1;    
      b.fX = t0.GetX();
      if( b.fParam.TransportToX( x0, maxSin ) ) nB++;
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
    if( !b1.fOK ) continue;
    if( b1.fNHits < minNPartHits ) continue;
    AliHLTTPCCATrackParam &t1 = b1.fParam;
    Int_t iBest2 = -1;
    Int_t lBest2 = 0;
    Int_t start2 = (iSlice1!=iSlice2) ?0 :i1+1;
    for (Int_t i2=start2; i2<N2; i2++) {
      AliHLTTPCCABorderTrack &b2 = B2[i2];
      if( !b2.fOK ) continue;
      if( b2.fNHits < minNPartHits ) continue;
      if( b2.fNHits < lBest2 ) continue;
      if( b1.fNHits + b2.fNHits < minNTotalHits ) continue;

      //if( TMath::Abs(b1.fX - b2.fX)>maxDX ) continue;

      AliHLTTPCCATrackParam &t2 = b2.fParam;

      Float_t c= t2.CosPhi()*t1.CosPhi()>=0 ?1 :-1;
      Float_t dk = t2.Kappa() - c*t1.Kappa(); 
      Float_t s2k = t2.Err2Kappa() + t1.Err2Kappa();
      
      if( dk*dk>factor2k*s2k ) continue;     


      Float_t chi2ys = GetChi2( t1.Y(),c*t1.SinPhi(),t1.Cov()[0],c*t1.Cov()[3],t1.Cov()[5], 
				t2.Y(),  t2.SinPhi(),t2.Cov()[0],  t2.Cov()[3],t2.Cov()[5] );

      if( chi2ys>factor2ys ) continue;

      Float_t chi2zt = GetChi2( t1.Z(),c*t1.DzDs(),t1.Cov()[2],c*t1.Cov()[7],t1.Cov()[9], 
				t2.Z(),  t2.DzDs(),t2.Cov()[2],  t2.Cov()[7],t2.Cov()[9] );
            
      if( chi2zt>factor2zt ) continue;
      
      lBest2 = b2.fNHits;
      iBest2 = b2.fITrack;
    }
      
    if( iBest2 <0 ) continue;

    AliHLTTPCCASliceTrackInfo &newTrack1 = fTrackInfos[fSliceTrackInfoStart[iSlice1]+b1.fITrack ];
    
    AliHLTTPCCASliceTrackInfo &newTrack2 = fTrackInfos[fSliceTrackInfoStart[iSlice2]+iBest2 ];

    Int_t old1 = newTrack2.fPrevNeighbour;
  
    if( old1 >= 0 ){
      AliHLTTPCCASliceTrackInfo &oldTrack1 = fTrackInfos[fSliceTrackInfoStart[iSlice1]+old1];
      if( oldTrack1.fNClusters  < newTrack1.fNClusters ){
	newTrack2.fPrevNeighbour = -1;
	oldTrack1.fNextNeighbour = -1;	
      } else continue;
    }
    Int_t old2 = newTrack1.fNextNeighbour;
    if( old2 >= 0 ){
      AliHLTTPCCASliceTrackInfo &oldTrack2 = fTrackInfos[fSliceTrackInfoStart[iSlice2]+old2];
      if( oldTrack2.fNClusters < newTrack2.fNClusters ){
	oldTrack2.fPrevNeighbour = -1;
      } else continue;
    }
    newTrack1.fNextNeighbour = iBest2; 
    newTrack2.fPrevNeighbour = b1.fITrack;	
  }  
  
}


void AliHLTTPCCAMerger::Merging()
{
  //* track merging between slices


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
	if( itr==0 ) sliceFirstClusterRef = track.fFirstClusterRef;
	track.fPrevNeighbour = -1;
	if( track.fNextNeighbour == -2 ){
	  track.fNextNeighbour = -1;
	  continue;
	}
	AliHLTTPCCASliceTrackInfo &trackNew = tmpT[nTr];
	trackNew = track;
	trackNew.fFirstClusterRef = sliceFirstClusterRef + nH;

	for( Int_t ih=0; ih<track.fNClusters; ih++ ) tmpH[nH+ih] = fClusterInfos[track.fFirstClusterRef+ih];
	nTr++;
	nH+=track.fNClusters;

	int jtr =  track.fNextNeighbour;

	if( jtr<0 ) continue;
	AliHLTTPCCASliceTrackInfo &neighTrack = fTrackInfos[ fSliceTrackInfoStart[iSlice]+jtr];
	
	track.fNextNeighbour = -1;
	neighTrack.fNextNeighbour = -2;

	for( Int_t ih=0; ih<neighTrack.fNClusters; ih++ ) 
	  tmpH[nH+ih] = fClusterInfos[neighTrack.fFirstClusterRef+ih];

	trackNew.fNClusters += neighTrack.fNClusters;
	trackNew.fNextNeighbour = -1;
	nH+=neighTrack.fNClusters;
	if( neighTrack.fInnerParam.X() < track.fInnerParam.X() ) trackNew.fInnerParam = neighTrack.fInnerParam;
	if( neighTrack.fOuterParam.X() > track.fOuterParam.X() ) trackNew.fOuterParam = neighTrack.fOuterParam;
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

      if( track.fUsed ) continue;
      if( track.fPrevNeighbour>=0 ) continue;

      AliHLTTPCCATrackParam startPoint = track.fInnerParam, endPoint = track.fOuterParam;
      Float_t startAlpha = track.fInnerAlpha, endAlpha = track.fOuterAlpha;      

      Int_t hits[2000];
      Int_t firstHit = 1000;
      Int_t nHits = 0;
      Int_t jSlice = iSlice;
      Int_t jtr = itr;

      {
	track.fUsed = 1;
	for( Int_t jhit=0; jhit<track.fNClusters; jhit++){
	  Int_t id = track.fFirstClusterRef + jhit;
	  hits[firstHit+jhit] = id;
	}
	nHits=track.fNClusters;
	jtr = track.fNextNeighbour;
	jSlice = nextSlice[iSlice];			
      }

      while( jtr >=0 ){
	AliHLTTPCCASliceTrackInfo &segment = fTrackInfos[fSliceTrackInfoStart[jSlice]+jtr];
	if( segment.fUsed ) break;
	segment.fUsed = 1;
	Bool_t dir = 0;
	Int_t startHit = firstHit+ nHits;
	Float_t d00 = startPoint.GetDistXZ2(segment.fInnerParam );
	Float_t d01 = startPoint.GetDistXZ2(segment.fOuterParam );
	Float_t d10 = endPoint.GetDistXZ2(segment.fInnerParam);
	Float_t d11 = endPoint.GetDistXZ2(segment.fOuterParam );
	if( d00<=d01 && d00<=d10 && d00<=d11 ){
	  startPoint = segment.fOuterParam;
	  startAlpha = segment.fOuterAlpha;
	  dir = 1;
	  firstHit -= segment.fNClusters;
	  startHit = firstHit;
	}else if( d01<=d10 && d01<=d11 ){
	  startPoint = segment.fInnerParam;
	  startAlpha = segment.fInnerAlpha;
	  dir = 0;
	  firstHit -= segment.fNClusters;
	  startHit = firstHit;
	}else if( d10<=d11 ){
	  endPoint = segment.fOuterParam;
	  endAlpha = segment.fOuterAlpha;
	  dir = 0;
	}else{
	  endPoint = segment.fInnerParam;
	  endAlpha = segment.fInnerAlpha;
	  dir = 1;
	}
	
	for( Int_t jhit=0; jhit<segment.fNClusters; jhit++){
	  Int_t id = segment.fFirstClusterRef + jhit;
	  hits[startHit+(dir ?(segment.fNClusters-1-jhit) :jhit)] = id;
	}
	nHits+=segment.fNClusters;
	jtr = segment.fNextNeighbour;
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
      AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
      p.CalculateFitParameters( fitPar, fSliceParam.Bz() );
      
      {
	Double_t xTPC=83.65; //SG!!!
	Double_t dAlpha = 0.00609235;
	
	if( p.TransportToXWithMaterial( xTPC, fitPar ) ){
	  Double_t y=p.GetY();
	  Double_t ymax=xTPC*CAMath::Tan(dAlpha/2.); 
	  if (y > ymax) {
	    if( p.Rotate( dAlpha ) ){ startAlpha+=dAlpha;  p.TransportToXWithMaterial( xTPC, fitPar ); }
	  } else if (y <-ymax) {
	    if( p.Rotate( -dAlpha ) ){  startAlpha-=dAlpha; p.TransportToXWithMaterial( xTPC, fitPar );}
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

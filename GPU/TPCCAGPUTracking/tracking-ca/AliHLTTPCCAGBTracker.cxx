// $Id$
//***************************************************************************
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
//***************************************************************************

#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCAGBHit.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGBTrack.h"
#include "AliHLTTPCCATrackParam.h"

#include "AliHLTTPCCAMath.h"
#include "TStopwatch.h"

//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#include "TApplication.h"
#endif //DRAW


AliHLTTPCCAGBTracker::AliHLTTPCCAGBTracker()
  :
    fSlices(0), 
    fNSlices(0), 
    fHits(0),
    fNHits(0),
    fTrackHits(0), 
    fTracks(0), 
    fNTracks(0),
    fSliceTrackInfos(0),
    fTime(0),
    fStatNEvents(0)
{
  //* constructor
  for( Int_t i=0; i<20; i++ ) fStatTime[i] = 0;
}

AliHLTTPCCAGBTracker::AliHLTTPCCAGBTracker(const AliHLTTPCCAGBTracker&)
  : 
    fSlices(0), 
    fNSlices(0), 
    fHits(0),
    fNHits(0),
    fTrackHits(0), 
    fTracks(0), 
    fNTracks(0),
    fSliceTrackInfos(0),
    fTime(0),
    fStatNEvents(0)
{
  //* dummy
}

AliHLTTPCCAGBTracker &AliHLTTPCCAGBTracker::operator=(const AliHLTTPCCAGBTracker&)
{
  //* dummy
  return *this;
}

AliHLTTPCCAGBTracker::~AliHLTTPCCAGBTracker()
{
  //* destructor
  StartEvent();
  if( fSliceTrackInfos ) delete[] fSliceTrackInfos;
  fSliceTrackInfos = 0;
  if( fSlices ) delete[] fSlices;
  fSlices=0;
}

void AliHLTTPCCAGBTracker::SetNSlices( Int_t N )
{
  //* set N of slices
  StartEvent();
  fNSlices = N;
  if( fSliceTrackInfos ) delete[] fSliceTrackInfos;
  fSliceTrackInfos = 0;
  if( fSlices ) delete[] fSlices;
  fSlices=0;
  fSlices = new AliHLTTPCCATracker[N];
  fSliceTrackInfos = new AliHLTTPCCAGBSliceTrackInfo *[N];
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    fSliceTrackInfos[iSlice] = 0;
  }
}

void AliHLTTPCCAGBTracker::StartEvent()
{
  //* clean up track and hit arrays

  if( fSliceTrackInfos ){
    for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
      if( fSliceTrackInfos[iSlice] ) delete[] fSliceTrackInfos[iSlice];
      fSliceTrackInfos[iSlice] = 0;
    }
  }
  if( fTrackHits ) delete[] fTrackHits;
  fTrackHits = 0;
  if( fTracks ) delete[] fTracks;
  fTracks = 0;
  if( fHits ) delete[] fHits;
  fHits=0;
  fNHits = 0;
  fNTracks = 0;
  for( int i=0; i<fNSlices; i++) fSlices[i].StartEvent();
}


void AliHLTTPCCAGBTracker::SetNHits( Int_t nHits )
{
  //* set the number of hits
  if( fHits ) delete[] fHits;
  fHits = 0;
  fHits = new AliHLTTPCCAGBHit[ nHits ];
  fNHits = 0;
}  

void AliHLTTPCCAGBTracker::ReadHit( Float_t x, Float_t y, Float_t z, 
				    Float_t errY, Float_t errZ, Float_t amp,
				    Int_t ID, Int_t iSlice, Int_t iRow )
{
  //* read the hit to local array
  AliHLTTPCCAGBHit &hit = fHits[fNHits];
  hit.X() = x;
  hit.Y() = y;
  hit.Z() = z;  
  hit.ErrX() = 1.e-4;//fSlices[iSlice].Param().ErrX();
  hit.ErrY() = errY;
  hit.ErrZ() = errZ;
  hit.Amp() = amp;
  hit.ID() = ID;
  hit.ISlice()=iSlice;
  hit.IRow() = iRow;
  hit.IsUsed() = 0;
  fNHits++;
}

void AliHLTTPCCAGBTracker::FindTracks()
{
  //* main tracking routine
  fTime = 0;
  fStatNEvents++;  
#ifdef DRAW
  if( fStatNEvents<=1 ){
    if( !gApplication ){
      TApplication *myapp = new TApplication("myapp",0,0);
    }    
    AliHLTTPCCADisplay::Instance().Init();
  }
  AliHLTTPCCADisplay::Instance().SetSliceView();
  //AliHLTTPCCADisplay::Instance().SetTPCView();
  //AliHLTTPCCADisplay::Instance().DrawTPC();
#endif //DRAW  

  if( fNHits<=0 ) return;
  
  std::sort(fHits,fHits+fNHits, AliHLTTPCCAGBHit::Compare );  

  // Read hits, row by row

  int nHitsTotal = fNHits;
  Float_t *hitY = new Float_t [nHitsTotal];
  Float_t *hitZ = new Float_t [nHitsTotal];

  Int_t sliceNHits[fNSlices];  
  Int_t rowNHits[fNSlices][200];
  
  for( Int_t is=0; is<fNSlices; is++ ){
    sliceNHits[is] = 0;
    for( Int_t ir=0; ir<200; ir++ ) rowNHits[is][ir] = 0;    
  }

  for( int ih=0; ih<nHitsTotal; ih++){
    AliHLTTPCCAGBHit &h = fHits[ih];    
    sliceNHits[h.ISlice()]++;
    rowNHits[h.ISlice()][h.IRow()]++;
  }
  
  Int_t firstSliceHit = 0;
  for( Int_t is=0; is<fNSlices; is++ ){
    fFirstSliceHit[is] = firstSliceHit;
    Int_t rowFirstHits[200];
    Int_t firstRowHit = 0;
    for( Int_t ir=0; ir<200; ir++ ){
      rowFirstHits[ir] = firstRowHit;
      for( Int_t ih=0; ih<rowNHits[is][ir]; ih++){
	AliHLTTPCCAGBHit &h = fHits[firstSliceHit + firstRowHit + ih];    
	hitY[firstRowHit + ih] = h.Y();
	hitZ[firstRowHit + ih] = h.Z();	
      }
      firstRowHit+=rowNHits[is][ir];
    }
    //if( is==24 ){//SG!!!
    fSlices[is].ReadEvent( rowFirstHits, rowNHits[is], hitY, hitZ, sliceNHits[is] );
    //}
    firstSliceHit+=sliceNHits[is];
  }


  delete[] hitY;
  delete[] hitZ;

  TStopwatch timer1;
  TStopwatch timer2;
  //cout<<"Start CA reconstruction"<<endl;
  for( int iSlice=0; iSlice<fNSlices; iSlice++ ){
    TStopwatch timer;
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    slice.Reconstruct();
    timer.Stop();
    //fTime+= timer.CpuTime();
    //blaTime+= timer.CpuTime();
    fStatTime[0] += timer.CpuTime();
    fStatTime[1]+=slice.Timers()[0];
    fStatTime[2]+=slice.Timers()[1];
    fStatTime[3]+=slice.Timers()[2];
    fStatTime[4]+=slice.Timers()[3];
    fStatTime[5]+=slice.Timers()[4];
    fStatTime[6]+=slice.Timers()[5];
    fStatTime[7]+=slice.Timers()[6];
    fStatTime[8]+=slice.Timers()[7];
  }

  timer2.Stop();
  //cout<<"blaTime = "<<timer2.CpuTime()*1.e3<<endl;

  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    AliHLTTPCCATracker &iS = fSlices[iSlice];
    if( fSliceTrackInfos[iSlice] ) delete[] fSliceTrackInfos[iSlice];
    fSliceTrackInfos[iSlice]=0;
    int iNTracks = iS.NOutTracks();
    fSliceTrackInfos[iSlice] = new AliHLTTPCCAGBSliceTrackInfo[iNTracks];
    for( Int_t itr=0; itr<iNTracks; itr++ ){
      fSliceTrackInfos[iSlice][itr].fPrevNeighbour = -1;
      fSliceTrackInfos[iSlice][itr].fNextNeighbour = -1; 
      fSliceTrackInfos[iSlice][itr].fUsed = 0;
    }
  }
  
  //cout<<"Start CA merging"<<endl;
  TStopwatch timerMerge;
  Merging();
  timerMerge.Stop();
  fStatTime[9]+=timerMerge.CpuTime();  
  //fTime+=timerMerge.CpuTime();
  //cout<<"End CA merging"<<endl;
  timer1.Stop();
  fTime+= timer1.CpuTime();

#ifdef DRAW
  AliHLTTPCCADisplay::Instance().Ask();
#endif //DRAW
}

void AliHLTTPCCAGBTracker::Merging()
{
  //* track merging between slices

  Float_t dalpha = fSlices[1].Param().Alpha() - fSlices[0].Param().Alpha();
  Int_t nextSlice[100], prevSlice[100];
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    nextSlice[iSlice] = iSlice + 1;
    prevSlice[iSlice] = iSlice - 1;
  }
  nextSlice[ fNSlices/2 - 1 ] = 0;
  prevSlice[ 0 ] = fNSlices/2 - 1;
  nextSlice[ fNSlices - 1 ] = fNSlices/2;
  prevSlice[ fNSlices/2 ] = fNSlices - 1;
  
  TStopwatch timerMerge1;

  Int_t maxNSliceTracks = 0;
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    AliHLTTPCCATracker &iS = fSlices[iSlice];
    if( maxNSliceTracks < iS.NOutTracks() ) maxNSliceTracks = iS.NOutTracks();
  }

  //* arrays for rotated track parameters

  AliHLTTPCCATrackParam *iTrParams[2], *jTrParams[2];
  Bool_t *iOK[2], *jOK[2];
  for( int i=0; i<2; i++ ){
    iTrParams[i] = new AliHLTTPCCATrackParam[maxNSliceTracks];
    jTrParams[i] = new AliHLTTPCCATrackParam[maxNSliceTracks];
    iOK[i] = new Bool_t [maxNSliceTracks];
    jOK[i] = new Bool_t [maxNSliceTracks];
  }
  
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    //cout<<"\nMerge slice "<<iSlice<<endl<<endl;
    AliHLTTPCCATracker &iS = fSlices[iSlice];
    Int_t jSlice = nextSlice[iSlice];
    AliHLTTPCCATracker &jS = fSlices[jSlice];    
    int iNTracks = iS.NOutTracks();
    int jNTracks = jS.NOutTracks();
    if( iNTracks<=0 || jNTracks<=0 ) continue;
    
    //* prepare slice tracks for merging
    
    for (Int_t itr=0; itr<iNTracks; itr++) {      
      iOK[0][itr] = 0;
      iOK[1][itr] = 0;
      if( iS.OutTracks()[itr].NHits()<10 ) continue;
      AliHLTTPCCATrackParam &iT1 = iTrParams[0][itr];
      AliHLTTPCCATrackParam &iT2 = iTrParams[1][itr];
      iT1 = iS.OutTracks()[itr].StartPoint();
      iT2 = iS.OutTracks()[itr].EndPoint();
      iOK[0][itr] = iT1.Rotate( dalpha/2 - CAMath::Pi()/2 );
      iOK[1][itr] = iT2.Rotate( dalpha/2 - CAMath::Pi()/2 );

      if( iOK[0][itr] ){
	iOK[0][itr] = iT1.TransportToX( 0, .99 );
	if( iS.Param().RMin() > iT1.Y() || iS.Param().RMax() < iT1.Y() ) iOK[0][itr]=0;
      }
      if( iOK[1][itr] ){
	iOK[1][itr] = iT2.TransportToX( 0, .99 );
	if( iS.Param().RMin() > iT2.Y() || iS.Param().RMax() < iT2.Y() ) iOK[1][itr]=0;
      }
    }

    for (Int_t jtr=0; jtr<jNTracks; jtr++) {      
      jOK[0][jtr] = 0;
      jOK[1][jtr] = 0;
      if( jS.OutTracks()[jtr].NHits()<10 ) continue;
      AliHLTTPCCATrackParam &jT1 = jTrParams[0][jtr];
      AliHLTTPCCATrackParam &jT2 = jTrParams[1][jtr];
      jT1 = jS.OutTracks()[jtr].StartPoint();
      jT2 = jS.OutTracks()[jtr].EndPoint();
      jOK[0][jtr] = jT1.Rotate( -dalpha/2 - CAMath::Pi()/2 );
      jOK[1][jtr] = jT2.Rotate( -dalpha/2 - CAMath::Pi()/2 );
      if( jOK[0][jtr] ){
	jOK[0][jtr] = jT1.TransportToX( 0, .99 );
	if( jS.Param().RMin() > jT1.Y() || jS.Param().RMax() < jT1.Y() ) jOK[0][jtr]=0;
      }
      if( jOK[1][jtr] ){
	jOK[1][jtr] = jT2.TransportToX( 0, .99 );
	if( jS.Param().RMin() > jT2.Y() || jS.Param().RMax() < jT2.Y() ) jOK[1][jtr]=0;
      }
    }

    //* start merging
    //cout<<"Start slice merging.."<<endl;
    for (Int_t itr=0; itr<iNTracks; itr++) {      
      if( !iOK[0][itr] && !iOK[1][itr] ) continue;
      Int_t jBest = -1;
      Int_t lBest = 0;
      for (Int_t jtr=0; jtr<jNTracks; jtr++) {
	if( jS.OutTracks()[jtr].NHits() < lBest ) continue;	
	if( !jOK[0][jtr] && !jOK[1][jtr] ) continue;
	for( Int_t ip=0; ip<2 && (jBest!=jtr) ; ip++ ){
	  if( !iOK[ip][itr] ) continue;
	  for( Int_t jp=0; jp<2 && (jBest!=jtr) ; jp++ ){
	    if( !jOK[jp][jtr] ) continue;	  
	    AliHLTTPCCATrackParam &iT = iTrParams[ip][itr];
	    AliHLTTPCCATrackParam &jT = jTrParams[jp][jtr];
	    // check for neighbouring	
	    {
	      Float_t factor2 = 3.5*3.5;
	      float d = jT.GetY() - iT.GetY();
	      float s2 = jT.GetErr2Y() + iT.GetErr2Y();
	      if( d*d>factor2*s2 ){
		continue;
	      }
	      d = jT.GetZ() - iT.GetZ();
	      s2 = jT.GetErr2Z() + iT.GetErr2Z();
	      if( d*d>factor2*s2 ){	    
		continue;
	      }
	      Bool_t ok = 1;
	      { // phi, kappa, DsDz signs are the same 
		d = jT.GetSinPhi() - iT.GetSinPhi();
		s2 = jT.GetErr2SinPhi() + iT.GetErr2SinPhi();
		if( d*d>factor2*s2 ) ok = 0;
		d = jT.GetKappa() - iT.GetKappa(); 
		s2 = jT.GetErr2Kappa() + iT.GetErr2Kappa();
		if( d*d>factor2*s2 ) ok = 0;
		d = jT.GetDzDs() - iT.GetDzDs();
		s2 = jT.GetErr2DzDs() + iT.GetErr2DzDs();
		if( d*d>factor2*s2 ) ok = 0;
	      }
	      if( !ok ){ // phi, kappa, DsDz signs are the different
		d = jT.GetSinPhi() + iT.GetSinPhi();
		s2 = jT.GetErr2SinPhi() + iT.GetErr2SinPhi();
		if( d*d>factor2*s2 ) continue;
		d = jT.GetKappa() + iT.GetKappa(); 
		s2 = jT.GetErr2Kappa() + iT.GetErr2Kappa();
		if( d*d>factor2*s2 ) continue;
		d = jT.GetDzDs() + iT.GetDzDs();
		s2 = jT.GetErr2DzDs() + iT.GetErr2DzDs();
		if( d*d>factor2*s2 ) continue;
	      }
	      // tracks can be matched

	      lBest = jS.OutTracks()[jtr].NHits();
	      jBest = jtr;
	    }
	  }
	}
      }
      if( jBest>=0 ){
	Int_t oldi = fSliceTrackInfos[jSlice][jBest].fPrevNeighbour;
	if( oldi >= 0 ){
	  if( iS.OutTracks()[ oldi ].NHits() < iS.OutTracks()[ itr ].NHits() ){
	    fSliceTrackInfos[jSlice][jBest].fPrevNeighbour = -1;
	    fSliceTrackInfos[iSlice][oldi].fNextNeighbour = -1;
	  } else continue;
	}
	//SG!!!
	fSliceTrackInfos[iSlice][itr].fNextNeighbour = jBest;
	fSliceTrackInfos[jSlice][jBest].fPrevNeighbour = itr;	
      }    
    }
  }

  for( Int_t i=0; i<2; i++){
    if( iTrParams[i] ) delete[] iTrParams[i];
    if( jTrParams[i] ) delete[] jTrParams[i];
    if( iOK[i] ) delete[] iOK[i];
    if( jOK[i] ) delete[] jOK[i];
  }

  timerMerge1.Stop();
  fStatTime[10]+=timerMerge1.CpuTime();

  TStopwatch timerMerge2;

  Int_t nTracksTot = 0;
  for( Int_t iSlice = 0; iSlice<fNSlices; iSlice++ ){    
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    nTracksTot+= slice.NOutTracks();
  }
  
  if( fTrackHits ) delete[] fTrackHits;
  fTrackHits = 0;
  if(fTracks ) delete[] fTracks;
  fTracks = 0;
  fTrackHits = new Int_t [fNHits*10];
  fTracks = new AliHLTTPCCAGBTrack[nTracksTot];
  fNTracks = 0;

  Int_t nTrackHits = 0;

  //cout<<"\nStart global track creation...\n"<<endl;  

	  static int nRejected = 0;

  Int_t maxNRows = fSlices[0].Param().NRows();
  
  for( Int_t iSlice = 0; iSlice<fNSlices; iSlice++ ){
    
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    for( Int_t itr=0; itr<slice.NOutTracks(); itr++ ){
      if( fSliceTrackInfos[iSlice][itr].fUsed ) continue;
      //cout<<"\n slice "<<iSlice<<", track "<<itr<<"\n"<<endl;
      //AliHLTTPCCAOutTrack &tCA = slice.OutTracks()[itr];
      AliHLTTPCCAGBTrack &t = fTracks[fNTracks];
      //t.Param() = tCA.StartPoint();
       //t.Alpha() = slice.Param().Alpha();
      t.NHits() = 0;
      t.FirstHitRef() = nTrackHits;

      struct FitPoint{    
	Int_t fISlice;
	Int_t fHitID;
	Float_t fX, fY, fZ, fErr2Y, fErr2Z, fAmp;
      } fitPoints[300];
      for( Int_t i=0; i<maxNRows; i++ ) fitPoints[i].fISlice = -1;
     
      Int_t nHits = 0;
      Int_t jSlice = iSlice;
      Int_t jtr = itr;
      do{ 
	if( fSliceTrackInfos[jSlice][jtr].fUsed ) break;
	fSliceTrackInfos[jSlice][jtr].fUsed = 1;
	AliHLTTPCCATracker &jslice = fSlices[jSlice];
	AliHLTTPCCAOutTrack &jTr = jslice.OutTracks()[jtr];
	for( int jhit=0; jhit<jTr.NHits(); jhit++){
	  int id = fFirstSliceHit[jSlice] + jslice.OutTrackHits()[jTr.FirstHitRef()+jhit];	  
	  AliHLTTPCCAGBHit &h = fHits[id];
	  FitPoint &p =  fitPoints[h.IRow()];
	  if( p.fISlice >=0 ) continue;
	  p.fISlice = h.ISlice();
	  p.fHitID = id;
	  p.fX = jslice.Rows()[h.IRow()].X();
	  p.fY = h.Y();
	  p.fZ = h.Z();
	  //p.fErr2Y = h.ErrY()*h.ErrY();
	  //p.fErr2Z = h.ErrZ()*h.ErrZ();
	  p.fAmp = h.Amp();
	  nHits++;	    
	}
	jtr = fSliceTrackInfos[jSlice][jtr].fNextNeighbour;
	jSlice = nextSlice[jSlice];	
      } while( jtr >=0 ); 

      if( nHits < 10 ) continue;     //SG!!!

      Int_t firstRow = 0, lastRow = maxNRows-1;
      for( firstRow=0; firstRow<maxNRows; firstRow++ ){
	if( fitPoints[firstRow].fISlice>=0 ) break;
      }
      for( lastRow=maxNRows-1; lastRow>=0; lastRow-- ){
	if( fitPoints[lastRow].fISlice>=0 ) break;
      }
      Int_t mmidRow = (firstRow + lastRow )/2;
      Int_t midRow = firstRow;
      for( Int_t i=firstRow+1; i<=lastRow; i++ ){
	if( fitPoints[i].fISlice<0 ) continue;	
	if( CAMath::Abs(i-mmidRow)>=CAMath::Abs(midRow-mmidRow) ) continue;
	midRow = i;
      }
      if( midRow==firstRow || midRow==lastRow ) continue;
  
      Int_t searchRows[300];
      Int_t nSearchRows = 0;

      for( Int_t i=firstRow; i<=lastRow; i++ ) searchRows[nSearchRows++] = i;
      for( Int_t i=lastRow+1; i<maxNRows; i++ ) searchRows[nSearchRows++] = i;
      for( Int_t i=firstRow-1; i>=0; i-- ) searchRows[nSearchRows++] = i;
      
      // refit 

      AliHLTTPCCATrackParam t0;

      {	

	{
	  FitPoint &p0 =  fitPoints[firstRow];
	  FitPoint &p1 =  fitPoints[midRow];
	  FitPoint &p2 =  fitPoints[lastRow];
	  Float_t x0=p0.fX, y0=p0.fY, z0=p0.fZ;
	  Float_t x1=p1.fX, y1=p1.fY, z1=p1.fZ;
	  Float_t x2=p2.fX, y2=p2.fY, z2=p2.fZ;
	  if( p1.fISlice!=p0.fISlice ){
	    Float_t dAlpha = fSlices[p0.fISlice].Param().Alpha() - fSlices[p1.fISlice].Param().Alpha();
	    Float_t c = CAMath::Cos(dAlpha);
	    Float_t s = CAMath::Sin(dAlpha);
	    x1 = p1.fX*c + p1.fY*s;
	    y1 = p1.fY*c - p1.fX*s;
	  }
	  if( p2.fISlice!=p0.fISlice ){
	    Float_t dAlpha = fSlices[p0.fISlice].Param().Alpha() - fSlices[p2.fISlice].Param().Alpha();
	    Float_t c = CAMath::Cos(dAlpha);
	    Float_t s = CAMath::Sin(dAlpha);
	    x2 = p2.fX*c + p2.fY*s;
	    y2 = p2.fY*c - p2.fX*s;
	  }

	  Float_t sp0[5] = {x0, y0, z0, .5, .5 };	
	  Float_t sp1[5] = {x1, y1, z1, .5, .5 };
	  Float_t sp2[5] = {x2, y2, z2, .5, .5 };
	  t0.ConstructXYZ3(sp0,sp1,sp2,1., 0);
	}
	
	Int_t currslice = fitPoints[firstRow].fISlice;

	for( Int_t rowID=0; rowID<nSearchRows; rowID++ ){
	  Int_t iRow = searchRows[rowID];	  
	  FitPoint &p =  fitPoints[iRow];

	  if( p.fISlice>=0 ){ 

	    //* Existing hit

	    //* Rotate to the new slice

	    if( p.fISlice!=currslice ){ 
	      if( !t0.Rotate( fSlices[p.fISlice].Param().Alpha() - fSlices[currslice].Param().Alpha() ) ) continue;	
	      currslice = p.fISlice;
	    }
	    //* Transport to the new row
	    
	    if( !t0.TransportToX( p.fX, .99 ) ) continue;

	    //* Calculate hit errors
	    
	    GetErrors2( p.fISlice, iRow, t0, p.fErr2Y, p.fErr2Z );

	  } else { 
	    //continue; //SG!!
	    //* Search for the missed hit

	    Float_t factor2 = 3.5*3.5;

	    AliHLTTPCCATracker *cslice = &(fSlices[currslice]);
	    AliHLTTPCCARow *row = &(cslice->Rows()[iRow]);
	    if( !t0.TransportToX( row->X(), .99 ) ) continue;

	    if( t0.GetY() > row->MaxY() ){ //next slice

	      Int_t j = nextSlice[currslice];

	      //* Rotate to the new slice
	      
	      if( !t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[j].Param().Alpha() ) ) continue;	      
	      currslice = j;
	      cslice = &(fSlices[currslice]);
	      row = &(cslice->Rows()[iRow]);
	      if( !t0.TransportToX( row->X(), .99 ) ) continue;
	      if( CAMath::Abs(t0.GetY()) > row->MaxY() ) continue;

	    }else if( t0.GetY() < -row->MaxY() ){ //prev slice
	      Int_t j = prevSlice[currslice];
	      //* Rotate to the new slice
	      if( !t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[j].Param().Alpha() ) ) break;
	      currslice = j;
	      cslice = &(fSlices[currslice]);
	      row = &(cslice->Rows()[iRow]);
	      if( !t0.TransportToX( row->X(), .99 ) ) continue;		
	      if( CAMath::Abs(t0.GetY()) > row->MaxY() ) continue;
	    }
	    
	    Int_t bestsh = -1;
	    Float_t ds = 1.e10;
	    for( Int_t ish=0; ish<row->NHits(); ish++ ){
	      AliHLTTPCCAHit &sh = cslice->Hits()[row->FirstHit()+ish];
	      Float_t dy = sh.Y() - t0.GetY();
	      Float_t dz = sh.Z() - t0.GetZ();
	      Float_t dds = dy*dy+dz*dz;
	      if( dds<ds ){
		ds = dds;
		bestsh = ish;
	      }
	    }
	    if( bestsh<0 ) continue;

	    //* Calculate hit errors
	    
	    GetErrors2( currslice, iRow, t0, p.fErr2Y, p.fErr2Z );

	    AliHLTTPCCAHit &sh = cslice->Hits()[row->FirstHit()+bestsh];
	    Float_t dy = sh.Y() - t0.GetY();
	    Float_t dz = sh.Z() - t0.GetZ();
	    Float_t s2z = /*t0.GetErr2Z() + */ p.fErr2Z;
	    if( dz*dz>factor2*s2z ) continue;		
	    Float_t s2y = /*t0.GetErr2Y() + */ p.fErr2Y;
	    if( dy*dy>factor2*s2y ) continue;

	    p.fISlice = currslice;
	    p.fHitID = fFirstSliceHit[p.fISlice] + cslice->HitsID()[row->FirstHit() + bestsh];
	    p.fX = row->X();
	    p.fY = sh.Y();
	    p.fZ = sh.Z();
	    p.fAmp = fHits[p.fHitID].Amp();
	  }

	  //* Update the track
	  
	  t0.Filter2( p.fY, p.fZ, p.fErr2Y, p.fErr2Z, .99 );	  
	}

	//* final refit, dE/dx calculation
	//cout<<"\n\nstart refit..\n"<<endl;

	AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
	{
	  Double_t sumDeDx = 0;
	  Int_t nDeDx = 0;
	  t.NHits() = 0;
	
	  t0.CalculateFitParameters( fitPar, fSlices[0].Param().Bz() );

	  t0.Cov()[ 0] = .1;
	  t0.Cov()[ 1] = 0;
	  t0.Cov()[ 2] = .1;
	  t0.Cov()[ 3] = 0;
	  t0.Cov()[ 4] = 0;
	  t0.Cov()[ 5] = .1;
	  t0.Cov()[ 6] = 0;
	  t0.Cov()[ 7] = 0;
	  t0.Cov()[ 8] = 0;
	  t0.Cov()[ 9] = .1;
	  t0.Cov()[10] = 0;
	  t0.Cov()[11] = 0;
	  t0.Cov()[12] = 0;
	  t0.Cov()[13] = 0;
	  t0.Cov()[14] = .1;
	  t0.Chi2() = 0;
	  t0.NDF() = -5;	
	  bool first = 1;
	  for( Int_t iRow = maxNRows-1; iRow>=0; iRow-- ){
	    FitPoint &p =  fitPoints[iRow];
	    if( p.fISlice<0 ) continue;
	    fTrackHits[nTrackHits+t.NHits()] = p.fHitID;
	    t.NHits()++;
	    
	    //* Rotate to the new slice

	    if( p.fISlice!=currslice ){ 
	      //cout<<"rotate..."<<endl;
	      //cout<<" before rotation:"<<endl;
	      //t0.Print();
	      if( !t0.Rotate( fSlices[p.fISlice].Param().Alpha() - fSlices[currslice].Param().Alpha() ) ) continue;	
	      //cout<<" after rotation:"<<endl;
	      //t0.Print();
	      currslice = p.fISlice;
	    }
	    //* Transport to the new row
	    
	    //cout<<" before transport:"<<endl;
	    //t0.Print();
	    
	    //if( !t0.TransportToX( p.fX, .99 ) ) continue;	    
	    if( !t0.TransportToXWithMaterial( p.fX, fitPar ) ) continue;	    
	    //if( !t0.TransportToX( p.fX, .99 ) ) continue;	    
	    //cout<<" after transport:"<<endl;
	    //t0.Print();

	    //* Update the track
	    
	    if( first ){
	      t0.Cov()[ 0] = .5*.5;
	      t0.Cov()[ 1] = 0;
	      t0.Cov()[ 2] = .5*.5;
	      t0.Cov()[ 3] = 0;
	      t0.Cov()[ 4] = 0;
	      t0.Cov()[ 5] = .2*.2;
	      t0.Cov()[ 6] = 0;
	      t0.Cov()[ 7] = 0;
	      t0.Cov()[ 8] = 0;
	      t0.Cov()[ 9] = .2*.2;
	      t0.Cov()[10] = 0;
	      t0.Cov()[11] = 0;
	      t0.Cov()[12] = 0;
	      t0.Cov()[13] = 0;
	      t0.Cov()[14] = .5*.5;
	      t0.Chi2() = 0;
	      t0.NDF() = -5;
	    }

	    //cout<<" before filtration:"<<endl;
	    //t0.Print();

	    if( !t0.Filter2( p.fY, p.fZ, p.fErr2Y, p.fErr2Z, .99 ) ) continue;	  
	    //cout<<" after filtration:"<<endl;
	    //t0.Print();
	      first = 0;
	    
	    if( CAMath::Abs( t0.CosPhi() )>1.e-4 ){
	      Float_t dLdX = CAMath::Sqrt(1.+t0.DzDs()*t0.DzDs())/CAMath::Abs(t0.CosPhi());
	      sumDeDx+=p.fAmp/dLdX;
	      nDeDx++;
	    } 
	  }
	  t.DeDx() = 0;
	  if( nDeDx >0 ) t.DeDx() = sumDeDx/nDeDx;
	  if( t0.GetErr2Y()<=0 ){
	    cout<<"nhits = "<<t.NHits()<<", t0.GetErr2Y() = "<<t0.GetErr2Y()<<endl;
	    t0.Print();
	    //exit(1);
	  }
	}

	if( t.NHits()<30 ) continue;//SG!!
	Double_t dAlpha = 0;
	{
	  Double_t xTPC=83.65;
	  Double_t ddAlpha = 0.00609235;
	  
	  if( t0.TransportToXWithMaterial( xTPC, fitPar ) ){
	    Double_t y=t0.GetY();
	    Double_t ymax=xTPC*CAMath::Tan(dAlpha/2.); 
	    if (y > ymax) {
	      if( t0.Rotate( ddAlpha ) ){ dAlpha=ddAlpha;  t0.TransportToXWithMaterial( xTPC, fitPar ); }
	    } else if (y <-ymax) {
	      if( t0.Rotate( -ddAlpha ) ){  dAlpha=-ddAlpha; t0.TransportToXWithMaterial( xTPC, fitPar );}
	    }	    
	  }
	}

	{
	  Bool_t ok=1;
	  
	  Float_t *c = t0.Cov();
	  for( int i=0; i<15; i++ ) ok = ok && finite(c[i]);
	  for( int i=0; i<5; i++ ) ok = ok && finite(t0.Par()[i]);
	  ok = ok && (t0.GetX()>50);
	  
	  if( c[0]<=0 || c[2]<=0 || c[5]<=0 || c[9]<=0 || c[14]<=0 ) ok = 0;
	  //if( c[0]>5. || c[2]>5. || c[5]>2. || c[9]>2 || c[14]>2 ) ok = 0;
	  if(!ok){
	    nRejected++;
	    //cout<<"\n\nRejected: "<<nRejected<<"\n"<<endl;
	    continue;
	  }
	}

	if( CAMath::Abs(t0.Kappa())<1.e-8 ) t0.Kappa() = 1.e-8;	
	t.Param() = t0;
	t.Alpha() = fSlices[currslice].Param().Alpha() + dAlpha;
	nTrackHits+= t.NHits();
	fNTracks++;   
      }
    }
  }
  //cout<<"\n\nRejected: "<<nRejected<<"\n"<<endl;
  timerMerge2.Stop();
  fStatTime[11]+=timerMerge2.CpuTime();

  TStopwatch timerMerge3;

  //* selection  
  //cout<<"Selection..."<<endl;
  {
    AliHLTTPCCAGBTrack *vtracks = new AliHLTTPCCAGBTrack [fNTracks];
    Int_t *vhits = new Int_t [fNHits];
    AliHLTTPCCAGBTrack **vptracks = new AliHLTTPCCAGBTrack* [fNTracks];

    for( Int_t itr=0; itr<fNTracks; itr++ ){
      vptracks[itr] = &(fTracks[itr]);
    }
    Int_t nTracks = 0;
    Int_t nHits = 0;
    std::sort(vptracks, vptracks+fNTracks, AliHLTTPCCAGBTrack::ComparePNClusters );
    for( Int_t itr=0; itr<fNTracks; itr++ ){
      AliHLTTPCCAGBTrack &t = *(vptracks[itr]);
      AliHLTTPCCAGBTrack &to = vtracks[nTracks];
      to=*(vptracks[itr]);
      to.FirstHitRef() = nHits;
      to.NHits() = 0;
      for( Int_t ih=0; ih<t.NHits(); ih++ ){
	Int_t jh = fTrackHits[t.FirstHitRef()+ih];
	AliHLTTPCCAGBHit &h = fHits[jh];
	if( h.IsUsed() ) continue;
	vhits[to.FirstHitRef() + to.NHits()] = jh;
	to.NHits()++;
	h.IsUsed() = 1;
      }
      if( to.NHits()<10 ) continue;//SG!!!
      nHits+=to.NHits();
      nTracks++;
      //cout<<to.Param().GetErr2Y()<<" "<<to.Param().GetErr2Z()<<endl;
    }
    fNTracks = nTracks;
    if( fTrackHits ) delete[] fTrackHits;
    if( fTracks ) delete[] fTracks;
    fTrackHits = vhits;
    fTracks = vtracks;
    delete[] vptracks;
  }
  timerMerge3.Stop();
  fStatTime[12]+=timerMerge3.CpuTime();
}

void AliHLTTPCCAGBTracker::GetErrors2( Int_t iSlice, Int_t iRow, AliHLTTPCCATrackParam &t, Float_t &Err2Y, Float_t &Err2Z )
{
  //
  // Use calibrated cluster error from OCDB
  //
    
  Float_t z = CAMath::Abs((250.-0.275)-CAMath::Abs(t.GetZ()));
  Int_t    type = (iRow<63) ? 0: (iRow>126) ? 1:2;
  Float_t cosPhiInv = CAMath::Abs(t.GetCosPhi())>1.e-2 ?1./t.GetCosPhi() :0;
  Float_t angleY = t.GetSinPhi()*cosPhiInv ;
  Float_t angleZ = t.GetDzDs()*cosPhiInv ;

  AliHLTTPCCATracker &slice = fSlices[iSlice];

  Err2Y = slice.Param().GetClusterError2(0,type, z,angleY);  
  Err2Z = slice.Param().GetClusterError2(1,type, z,angleZ);
}


void AliHLTTPCCAGBTracker::GetErrors2( AliHLTTPCCAGBHit &h, AliHLTTPCCATrackParam &t, Float_t &Err2Y, Float_t &Err2Z )
{
  //
  // Use calibrated cluster error from OCDB
  //

  GetErrors2( h.ISlice(), h.IRow(), t, Err2Y, Err2Z );
}

void AliHLTTPCCAGBTracker::WriteSettings( ostream &out ) const
{
  //* write settings to the file
  out<< NSlices()<<endl;  
  for( int iSlice=0; iSlice<NSlices(); iSlice++ ){    
    fSlices[iSlice].Param().WriteSettings( out );
  }
}

void AliHLTTPCCAGBTracker::ReadSettings( istream &in )
{
  //* Read settings from the file
  Int_t nSlices=0;
  in >> nSlices;
  SetNSlices( nSlices );
  for( int iSlice=0; iSlice<NSlices(); iSlice++ ){    
    AliHLTTPCCAParam param;
    param.ReadSettings ( in );
    fSlices[iSlice].Initialize( param ); 
  }
}

void AliHLTTPCCAGBTracker::WriteEvent( ostream &out ) const
{
  // write event to the file

  out<<NHits()<<endl;
  for (Int_t ih=0; ih<NHits(); ih++) {
    AliHLTTPCCAGBHit &h = fHits[ih];
    out<<h.X()<<" ";
    out<<h.Y()<<" ";
    out<<h.Z()<<" ";
    out<<h.ErrY()<<" ";
    out<<h.ErrZ()<<" ";
    out<<h.Amp()<<" ";
    out<<h.ID()<<" ";
    out<<h.ISlice()<<" ";
    out<<h.IRow()<<endl;
  }
}

void AliHLTTPCCAGBTracker::ReadEvent( istream &in ) 
{
  //* Read event from file 

  StartEvent();
  Int_t nHits;
  in >> nHits;
  SetNHits(nHits);
  for (Int_t i=0; i<nHits; i++) {
    Float_t x, y, z, errY, errZ;
    Float_t amp;
    Int_t id, iSlice, iRow;
    in>>x>>y>>z>>errY>>errZ>>amp>>id>>iSlice>>iRow;
    ReadHit( x, y, z, errY, errZ, amp, id, iSlice, iRow );
  }
}

void AliHLTTPCCAGBTracker::WriteTracks( ostream &out ) const 
{
  //* Write tracks to file 

  out<<fTime<<endl;
  Int_t nTrackHits = 0;
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    nTrackHits+=fTracks[itr].NHits();
  }
  out<<nTrackHits<<endl;
  for( Int_t ih=0; ih<nTrackHits; ih++ ){
    out<< fTrackHits[ih]<<" ";
  }
  out<<endl;
  
  out<<NTracks()<<endl;
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    AliHLTTPCCAGBTrack &t = fTracks[itr];    
    AliHLTTPCCATrackParam &p = t.Param();	
    out<< t.NHits()<<" ";
    out<< t.FirstHitRef()<<" ";
    out<< t.Alpha()<<" ";
    out<< t.DeDx()<<endl;
    out<< p.X()<<" ";
    out<< p.CosPhi()<<" ";
    out<< p.Chi2()<<" ";
    out<< p.NDF()<<endl;
    for( Int_t i=0; i<5; i++ ) out<<p.Par()[i]<<" ";
    out<<endl;
    for( Int_t i=0; i<15; i++ ) out<<p.Cov()[i]<<" ";
    out<<endl;
  }
}

void AliHLTTPCCAGBTracker::ReadTracks( istream &in )
{
  //* Read tracks  from file 

  in>>fTime;
  fStatTime[0]+=fTime;
  fStatNEvents++;
  if( fTrackHits ) delete[] fTrackHits;
  fTrackHits = 0;  
  Int_t nTrackHits = 0;
  in >> nTrackHits;
  fTrackHits = new Int_t [nTrackHits];
  for( Int_t ih=0; ih<nTrackHits; ih++ ){
    in >> TrackHits()[ih];
  }
  if( fTracks ) delete[] fTracks;
  fTracks = 0;  
  in >> fNTracks;
  fTracks = new AliHLTTPCCAGBTrack[fNTracks];
  for( Int_t itr=0; itr<NTracks(); itr++ ){
    AliHLTTPCCAGBTrack &t = Tracks()[itr];    
    AliHLTTPCCATrackParam &p = t.Param();	
    in>> t.NHits();
    in>> t.FirstHitRef();
    in>> t.Alpha();
    in>> t.DeDx();
    in>> p.X();
    in>> p.CosPhi();
    in>> p.Chi2();
    in>> p.NDF();
    for( Int_t i=0; i<5; i++ ) in>>p.Par()[i];
    for( Int_t i=0; i<15; i++ ) in>>p.Cov()[i];
  }
}

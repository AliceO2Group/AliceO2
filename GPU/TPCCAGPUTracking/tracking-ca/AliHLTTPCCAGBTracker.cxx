// $Id$
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


#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCAGBHit.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGBTrack.h"
#include "AliHLTTPCCATrackParam.h"
//#include "AliHLTTPCCAEventHeader.h"

#include "AliHLTTPCCAMath.h"
#include "TStopwatch.h"

//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#endif //DRAW


AliHLTTPCCAGBTracker::AliHLTTPCCAGBTracker()
  :
    fSlices(0), 
    fNSlices(0), 
    fHits(0),
    fExt2IntHitID(0),
    fNHits(0),
    fTrackHits(0), 
    fTracks(0), 
    fNTracks(0),
    fSliceTrackInfos(0),
    fTime(0),
    fStatNEvents(0),
    fSliceTrackerTime(0)
{
  //* constructor
  for( Int_t i=0; i<20; i++ ) fStatTime[i] = 0;
}

AliHLTTPCCAGBTracker::AliHLTTPCCAGBTracker(const AliHLTTPCCAGBTracker&)
  : 
    fSlices(0), 
    fNSlices(0), 
    fHits(0),
    fExt2IntHitID(0),
    fNHits(0),
    fTrackHits(0), 
    fTracks(0), 
    fNTracks(0),
    fSliceTrackInfos(0),
    fTime(0),
    fStatNEvents(0),
    fSliceTrackerTime(0)
{
  //* dummy
}

const AliHLTTPCCAGBTracker &AliHLTTPCCAGBTracker::operator=(const AliHLTTPCCAGBTracker&) const
{
  //* dummy
  return *this;
}

AliHLTTPCCAGBTracker::~AliHLTTPCCAGBTracker()
{
  //* destructor
  StartEvent();
  if( fSliceTrackInfos ){
    for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
      if( fSliceTrackInfos[iSlice] ) delete[] fSliceTrackInfos[iSlice];
      fSliceTrackInfos[iSlice] = 0;
    }
    delete[] fSliceTrackInfos;
  }
  fSliceTrackInfos = 0;
  if( fSlices ) delete[] fSlices;
  fSlices=0;
}

void AliHLTTPCCAGBTracker::SetNSlices( Int_t N )
{
  //* set N of slices
  StartEvent();
  if( fSliceTrackInfos ){
    for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
      if( fSliceTrackInfos[iSlice] ) delete[] fSliceTrackInfos[iSlice];
      fSliceTrackInfos[iSlice] = 0;
    }
    delete[] fSliceTrackInfos;
  }
  fSliceTrackInfos = 0;
  fNSlices = N;
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
  if( fExt2IntHitID ) delete[] fExt2IntHitID;
  fExt2IntHitID = 0;
  fNHits = 0;
  fNTracks = 0;
  for( Int_t i=0; i<fNSlices; i++) fSlices[i].StartEvent();
}


void AliHLTTPCCAGBTracker::SetNHits( Int_t nHits )
{
  //* set the number of hits
  if( fHits ) delete[] fHits;
  fHits = 0;
  if( fExt2IntHitID ) delete[] fExt2IntHitID;
  fExt2IntHitID = 0;
  fHits = new AliHLTTPCCAGBHit[ nHits ];
  fExt2IntHitID = new Int_t[ nHits ];
  fNHits = 0;
}  

void AliHLTTPCCAGBTracker::ReadHit( Float_t x, Float_t y, Float_t z, 
				    Float_t errY, Float_t errZ, Float_t amp,
				    Int_t ID, Int_t iSlice, Int_t iRow )
{
  //* read the hit to local array
  AliHLTTPCCAGBHit &hit = fHits[fNHits];
  hit.SetX( x );
  hit.SetY( y );
  hit.SetZ( z );  
  hit.SetErrX( 1.e-4 );//fSlices[iSlice].Param().ErrX();
  hit.SetErrY( errY );
  hit.SetErrZ( errZ );
  hit.SetAmp( amp );
  hit.SetID( ID );
  hit.SetISlice( iSlice );
  hit.SetIRow( iRow );
  hit.SetIsUsed( 0 );
  fNHits++;  
}

void AliHLTTPCCAGBTracker::FindTracks()
{
  //* main tracking routine
  fTime = 0;
  fStatNEvents++;  

#ifdef DRAW
  AliHLTTPCCADisplay::Instance().SetGB(this);
  AliHLTTPCCADisplay::Instance().SetTPCView();
  AliHLTTPCCADisplay::Instance().DrawTPC();
  AliHLTTPCCADisplay::Instance().Ask();
#endif //DRAW  

  
  std::sort(fHits,fHits+fNHits, AliHLTTPCCAGBHit::Compare );  

  for( Int_t i=0; i<fNHits; i++ )  fExt2IntHitID[fHits[i].ID()] = i;

  // Read hits, row by row

  Int_t nHitsTotal = fNHits;
  Float_t *hitY = new Float_t [nHitsTotal];
  Float_t *hitZ = new Float_t [nHitsTotal];

  Int_t sliceNHits[fNSlices];  
  Int_t rowNHits[fNSlices][200];
  
  for( Int_t is=0; is<fNSlices; is++ ){
    sliceNHits[is] = 0;
    for( Int_t ir=0; ir<200; ir++ ) rowNHits[is][ir] = 0;    
  }

  for( Int_t ih=0; ih<nHitsTotal; ih++){
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
    fSlices[is].ReadEvent( rowFirstHits, rowNHits[is], hitY, hitZ, sliceNHits[is] );
    
    //Int_t data[ rowNHits[is]]
    //AliHLTTPCCAEventHeader event;

    firstSliceHit+=sliceNHits[is];
  }


  if( hitY ) delete[] hitY;
  hitY=0;
  if( hitZ ) delete[] hitZ;
  hitZ=0;
 
  if( fNHits<=0 ) return;

  TStopwatch timer1;
  TStopwatch timer2;
  //std::cout<<"Start CA reconstruction"<<std::endl;
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    //std::cout<<"Reconstruct slice "<<iSlice<<std::endl;
    TStopwatch timer;
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    slice.Reconstruct();
    timer.Stop();
    //fTime+= timer.CpuTime();
    //blaTime+= timer.CpuTime();
    fStatTime[0] += timer.CpuTime();
    fStatTime[1]+=slice.Timer(0);
    fStatTime[2]+=slice.Timer(1);
    fStatTime[3]+=slice.Timer(2);
    fStatTime[4]+=slice.Timer(3);
    fStatTime[5]+=slice.Timer(4);
    fStatTime[6]+=slice.Timer(5);
    fStatTime[7]+=slice.Timer(6);
    fStatTime[8]+=slice.Timer(7);
  }

  timer2.Stop();
  //std::cout<<"blaTime = "<<timer2.CpuTime()*1.e3<<std::endl;
  fSliceTrackerTime = timer2.CpuTime();
  
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    AliHLTTPCCATracker &iS = fSlices[iSlice];
    if( fSliceTrackInfos[iSlice] ) delete[] fSliceTrackInfos[iSlice];
    fSliceTrackInfos[iSlice]=0;
    Int_t iNTracks = *iS.NOutTracks();
    fSliceTrackInfos[iSlice] = new AliHLTTPCCAGBSliceTrackInfo[iNTracks];

    for( Int_t itr=0; itr<iNTracks; itr++ ){
      fSliceTrackInfos[iSlice][itr].fPrevNeighbour = -1;
      fSliceTrackInfos[iSlice][itr].fNextNeighbour = -1; 
      fSliceTrackInfos[iSlice][itr].fUsed = 0;
    }
  }
  
  //std::cout<<"Start CA merging"<<std::endl;
  TStopwatch timerM;
  Refit(); 
  timerM.Stop();
  fStatTime[9]+=timerM.CpuTime();  
  //fTime+=timerMerge.CpuTime();
  //std::cout<<"Refit time = "<<timerM.CpuTime()*1.e3<<"ms"<<std::endl;

  TStopwatch timerMerge;
  Merging();
  timerMerge.Stop();
  fStatTime[9]+=timerMerge.CpuTime();  
  //fTime+=timerMerge.CpuTime();
  //std::cout<<"Merge time = "<<timerMerge.CpuTime()*1.e3<<"ms"<<std::endl;
  //std::cout<<"End CA merging"<<std::endl;
  timer1.Stop();
  fTime+= timer1.CpuTime();

#ifdef DRAW
  AliHLTTPCCADisplay::Instance().Ask();
#endif //DRAW
}


void AliHLTTPCCAGBTracker::FindTracks0()
{
  //* main tracking routine
  fTime = 0;
  fStatNEvents++;  
#ifdef DRAW
  AliHLTTPCCADisplay::Instance().SetTPCView();
  AliHLTTPCCADisplay::Instance().DrawTPC();
#endif //DRAW  

  if( fNHits<=0 ) return;
  
  std::sort(fHits,fHits+fNHits, AliHLTTPCCAGBHit::Compare );  

  // Read hits, row by row

  Int_t nHitsTotal = fNHits;
  Float_t *hitY = new Float_t [nHitsTotal];
  Float_t *hitZ = new Float_t [nHitsTotal];

  Int_t sliceNHits[fNSlices];  
  Int_t rowNHits[fNSlices][200];
  
  for( Int_t is=0; is<fNSlices; is++ ){
    sliceNHits[is] = 0;
    for( Int_t ir=0; ir<200; ir++ ) rowNHits[is][ir] = 0;    
  }

  for( Int_t ih=0; ih<nHitsTotal; ih++){
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
    
    //Int_t data[ rowNHits[is]]
    //AliHLTTPCCAEventHeader event;

    firstSliceHit+=sliceNHits[is];
  }

  if( hitY ) delete[] hitY;
  if( hitZ ) delete[] hitZ;
}


void AliHLTTPCCAGBTracker::FindTracks1()
{
  //* main tracking routine

  TStopwatch timer2;
  //std::cout<<"Start CA reconstruction"<<std::endl;
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    TStopwatch timer;
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    slice.Reconstruct();
    timer.Stop();
    //fTime+= timer.CpuTime();
    //blaTime+= timer.CpuTime();
    fStatTime[0] += timer.CpuTime();
    fStatTime[1]+=slice.Timer(0);
    fStatTime[2]+=slice.Timer(1);
    fStatTime[3]+=slice.Timer(2);
    fStatTime[4]+=slice.Timer(3);
    fStatTime[5]+=slice.Timer(4);
    fStatTime[6]+=slice.Timer(5);
    fStatTime[7]+=slice.Timer(6);
    fStatTime[8]+=slice.Timer(7);
  }

  timer2.Stop();
  //std::cout<<"blaTime = "<<timer2.CpuTime()*1.e3<<std::endl;
  fSliceTrackerTime = timer2.CpuTime();
}


void AliHLTTPCCAGBTracker::FindTracks2()
{
  //* main tracking routine

  TStopwatch timer1;

  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    AliHLTTPCCATracker &iS = fSlices[iSlice];
    if( fSliceTrackInfos[iSlice] ) delete[] fSliceTrackInfos[iSlice];
    fSliceTrackInfos[iSlice]=0;
    Int_t iNTracks = *iS.NOutTracks();    
    fSliceTrackInfos[iSlice] = new AliHLTTPCCAGBSliceTrackInfo[iNTracks];
    for( Int_t itr=0; itr<iNTracks; itr++ ){
      fSliceTrackInfos[iSlice][itr].fPrevNeighbour = -1;
      fSliceTrackInfos[iSlice][itr].fNextNeighbour = -1; 
      fSliceTrackInfos[iSlice][itr].fUsed = 0;
    }
  }
  
  //std::cout<<"Start CA merging"<<std::endl;
  TStopwatch timerMerge;
  Merging();
  timerMerge.Stop();
  fStatTime[9]+=timerMerge.CpuTime();  
  //fTime+=timerMerge.CpuTime();
  //std::cout<<"End CA merging"<<std::endl;
  timer1.Stop();
  fTime+= fSliceTrackerTime + timer1.CpuTime();

#ifdef DRAW
  AliHLTTPCCADisplay::Instance().Ask();
#endif //DRAW
}


void AliHLTTPCCAGBTracker::Refit()
{
  //* Refit the slice tracks

  Int_t maxNRows = fSlices[0].Param().NRows();
  
  for( Int_t iSlice = 0; iSlice<fNSlices; iSlice++ ){
    
    AliHLTTPCCATracker &slice = fSlices[iSlice];

    for( Int_t itr=0; itr<*slice.NOutTracks(); itr++ ){

      AliHLTTPCCAOutTrack &iTr = slice.OutTracks()[itr];

      struct FitPoint{    
	Int_t fISlice;
	Int_t fHitID;
	Float_t fX, fY, fZ, fErr2Y, fErr2Z, fAmp;
      } fitPoints[300];
      for( Int_t i=0; i<maxNRows; i++ ) fitPoints[i].fISlice = -1;
      
      Int_t nHits = 0;

      for( Int_t ihit=0; ihit<iTr.NHits(); ihit++){
	Int_t id = fFirstSliceHit[iSlice] + slice.OutTrackHits()[iTr.FirstHitRef()+ihit];
	AliHLTTPCCAGBHit &h = fHits[id];
	FitPoint &p =  fitPoints[h.IRow()];
	if( p.fISlice >=0 ) continue;
	p.fISlice = h.ISlice();
	p.fHitID = id;
	p.fX = slice.Row(h.IRow()).X();
	p.fY = h.Y();
	p.fZ = h.Z();
	p.fAmp = h.Amp();
	nHits++;	    
      }
     
      //if( nHits < 10 ) continue; SG!!!

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
      
      //std::cout<<"\nRefit slice "<<iSlice<<" track "<<itr<<": "<<std::endl;

     // refit down 
      {
	AliHLTTPCCATrackParam t0 = iTr.StartPoint();
	/*
	  {	
	  
	  {
	  FitPoint &p0 =  fitPoints[firstRow];
	  FitPoint &p1 =  fitPoints[midRow];
	  FitPoint &p2 =  fitPoints[lastRow];
	  Float_t x0=p0.fX, y0=p0.fY, z0=p0.fZ;
	  Float_t x1=p1.fX, y1=p1.fY, z1=p1.fZ;
	  Float_t x2=p2.fX, y2=p2.fY, z2=p2.fZ;
	  Float_t sp0[5] = {x0, y0, z0, .5, .5 };	
	  Float_t sp1[5] = {x1, y1, z1, .5, .5 };
	  Float_t sp2[5] = {x2, y2, z2, .5, .5 };
	  t0.ConstructXYZ3(sp0,sp1,sp2,1., 0);
	}
		
	for( Int_t rowID=0; rowID<nSearchRows; rowID++ ){
	  Int_t iRow = searchRows[rowID];	  
	  FitPoint &p =  fitPoints[iRow];	  
	  if( p.fISlice<0 ) continue;
	  if( !t0.TransportToX( p.fX, .99 ) ) continue;
	  GetErrors2( p.fISlice, iRow, t0, p.fErr2Y, p.fErr2Z );
	  t0.Filter2( p.fY, p.fZ, p.fErr2Y, p.fErr2Z, .99 );	  
	}
	*/
	//* final refit, dE/dx calculation
	//std::cout<<"\n\nstart refit..\n"<<std::endl;

	AliHLTTPCCATrackParam t00 = t0;
	AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
	if(1){
	  Double_t sumDeDx = 0;
	  Int_t nDeDx = 0;
	  
	  t0.CalculateFitParameters( fitPar, fSlices[0].Param().Bz() );
	  /*
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
	  */
	  t0.SetChi2( 0 );
	  t0.SetNDF( -5 );	
	  Bool_t first = 1;
	  for( Int_t iRow = maxNRows-1; iRow>=0; iRow-- ){
	    FitPoint &p =  fitPoints[iRow];
	    if( p.fISlice<0 ) continue;
	    //std::cout<<" row "<<iRow<<std::endl;	   
	    //t00.Print();
	    //t0.Print();
	    //if( !t0.TransportToX( p.fX, t00, .99 ) ){
	    if( !t0.TransportToXWithMaterial( p.fX, t00, fitPar ) ){
	      //std::cout<<"row "<<iRow<<": can not transport!!!"<<std::endl;
	      //t00.Print();
	      //t0.Print();
	      continue;	    
	    }
	    
	    //* Update the track
	    
	    if( first ){
	      t0.SetCov( 0, 10 );
	      t0.SetCov( 1,  0 );
	      t0.SetCov( 2, 10 );
	      t0.SetCov( 3,  0 );
	      t0.SetCov( 4,  0 );
	      t0.SetCov( 5,  1 );
	      t0.SetCov( 6,  0 );
	      t0.SetCov( 7,  0 );
	      t0.SetCov( 8,  0 );
	      t0.SetCov( 9,  1 );
	      t0.SetCov(10,  0 );
	      t0.SetCov(11,  0 );
	      t0.SetCov(12,  0 );
	      t0.SetCov(13,  0 );
	      t0.SetCov(14,  1 );
	      t0.SetChi2( 0 );
	      t0.SetNDF( -5 );
	    }
	    
	    slice.GetErrors2( iRow, t00, p.fErr2Y, p.fErr2Z );

	    if( !t0.Filter2NoCos( p.fY, p.fZ, p.fErr2Y, p.fErr2Z ) ){
	      //std::cout<<"row "<<iRow<<": can not filter!!!"<<std::endl;
	      //t00.Print();
	      //t0.Print();
	      continue;	  
	    }
	    first = 0;
	    
	    if( CAMath::Abs( t00.CosPhi() )>1.e-4 ){
	      Float_t dLdX = CAMath::Sqrt(1.+t00.DzDs()*t00.DzDs())/CAMath::Abs(t00.CosPhi());
	      sumDeDx+=p.fAmp/dLdX;
	      nDeDx++;
	    }
	    
	  }
	  //t.DeDx() = 0;
	  //if( nDeDx >0 ) t.DeDx() = sumDeDx/nDeDx;
	  if( t0.GetErr2Y()<=0 ){
	    //std::cout<<"nhits = "<<t.NHits()<<", t0.GetErr2Y() = "<<t0.GetErr2Y()<<std::endl;
	    //t0.Print();
	    //exit(1);
	  }
	}
	
	{
	  Bool_t ok=1;
	  
	  const Float_t *c = t0.Cov();
	  for( Int_t i=0; i<15; i++ ) ok = ok && finite(c[i]);
	  for( Int_t i=0; i<5; i++ ) ok = ok && finite(t0.Par()[i]);
	  ok = ok && (t0.GetX()>50);
	  
	  if( c[0]<=0 || c[2]<=0 || c[5]<=0 || c[9]<=0 || c[14]<=0 ) ok = 0;
	  //if( c[0]>5. || c[2]>5. || c[5]>2. || c[9]>2 || c[14]>2 ) ok = 0;

	  if( CAMath::Abs(t0.SinPhi())>.99 ) ok = 0;
	  else t0.SetCosPhi( CAMath::Sqrt(1.-t0.SinPhi()*t0.SinPhi()) );

	  if(!ok){
	    //std::cout<<" nan check: track rejected"<<std::endl;
	    //nRejected++;
	    //std::cout<<"\n\nRejected: "<<nRejected<<"\n"<<std::endl;
	    continue;
	  }
	}
	
	if( CAMath::Abs(t0.Kappa())<1.e-8 ) t0.SetKappa( 1.e-8 );	
	t0.TransportToX( slice.Row(firstRow).X(), .99 );
	iTr.SetStartPoint( t0 );
      }

      // refit up 
      {
	AliHLTTPCCATrackParam t0 = iTr.StartPoint();

	AliHLTTPCCATrackParam t00 = t0;
	AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
	if(1){
	  
	  t0.CalculateFitParameters( fitPar, fSlices[0].Param().Bz() );
	  t0.SetChi2( 0 );
	  t0.SetNDF( -5 );	
	  Bool_t first = 1;
	  for( Int_t iRow = 0; iRow<maxNRows; iRow++ ){
	    FitPoint &p =  fitPoints[iRow];
	    if( p.fISlice<0 ) continue;
	    if( !t0.TransportToXWithMaterial( p.fX, t00, fitPar ) ){
	      //std::cout<<"row "<<iRow<<": can not transport!!!"<<std::endl;
	      //t00.Print();
	      //t0.Print();
	      continue;	    
	    }
	    
	    //* Update the track
	    
	    if( first ){
	      t0.SetCov( 0, 10 );
	      t0.SetCov( 1,  0 );
	      t0.SetCov( 2, 10 );
	      t0.SetCov( 3,  0 );
	      t0.SetCov( 4,  0 );
	      t0.SetCov( 5,  1 );
	      t0.SetCov( 6,  0 );
	      t0.SetCov( 7,  0 );
	      t0.SetCov( 8,  0 );
	      t0.SetCov( 9,  1 );
	      t0.SetCov(10,  0 );
	      t0.SetCov(11,  0 );
	      t0.SetCov(12,  0 );
	      t0.SetCov(13,  0 );
	      t0.SetCov(14,  1 );
	      t0.SetChi2( 0 );
	      t0.SetNDF( -5 );
	    }
	    
	    slice.GetErrors2( iRow, t00, p.fErr2Y, p.fErr2Z );

	    if( !t0.Filter2NoCos( p.fY, p.fZ, p.fErr2Y, p.fErr2Z ) ){
	      //std::cout<<"row "<<iRow<<": can not filter!!!"<<std::endl;
	      //t00.Print();
	      //t0.Print();
	      continue;	  
	    }
	    first = 0;
	  }
	}
	
	{
	  Bool_t ok=1;
	  
	  const Float_t *c = t0.Cov();
	  for( Int_t i=0; i<15; i++ ) ok = ok && finite(c[i]);
	  for( Int_t i=0; i<5; i++ ) ok = ok && finite(t0.Par()[i]);
	  ok = ok && (t0.GetX()>50);
	  
	  if( c[0]<=0 || c[2]<=0 || c[5]<=0 || c[9]<=0 || c[14]<=0 ) ok = 0;
	  //if( c[0]>5. || c[2]>5. || c[5]>2. || c[9]>2 || c[14]>2 ) ok = 0;

	  if( CAMath::Abs(t0.SinPhi())>.99 ) ok = 0;
	  else t0.SetCosPhi( CAMath::Sqrt(1.-t0.SinPhi()*t0.SinPhi()) );

	  if(!ok){
	    //std::cout<<" refit: nan check: track rejected"<<std::endl;
	    //nRejected++;
	    //std::cout<<"\n\nRejected: "<<nRejected<<"\n"<<std::endl;
	    continue;
	  }
	}
	
	if( CAMath::Abs(t0.Kappa())<1.e-8 ) t0.SetKappa( 1.e-8 );	
	t0.TransportToX( slice.Row(lastRow).X(), .99 );
	iTr.SetEndPoint( t0 );		

      }
    }
  }
}


Bool_t AliHLTTPCCAGBTracker::FitTrack( AliHLTTPCCATrackParam &T, AliHLTTPCCATrackParam t0, 
				       Float_t &Alpha, Int_t hits[], Int_t &NTrackHits, Float_t &DeDx,
				       Bool_t dir )
{
  // Fit the track

  AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
  Double_t sumDeDx = 0;
  Int_t nDeDx = 0;
  AliHLTTPCCATrackParam t = t0;
  Bool_t first = 1;
 
  t0.CalculateFitParameters( fitPar, fSlices[0].Param().Bz() );

  Int_t hitsNew[1000];
  Int_t nHitsNew = 0;

  for( Int_t ihit=0; ihit<NTrackHits; ihit++){
    Int_t jhit = dir ?(NTrackHits-1-ihit) :ihit;
    AliHLTTPCCAGBHit &h = fHits[hits[jhit]];
    Int_t iSlice = h.ISlice();
    AliHLTTPCCATracker &slice = fSlices[iSlice];

    if( CAMath::Abs( slice.Param().Alpha()-Alpha)>1.e-4 ){
      if( ! t.RotateNoCos(  slice.Param().Alpha() - Alpha, t0, .999 ) ) continue;
      Alpha = slice.Param().Alpha();
    }

    Float_t x = slice.Row(h.IRow()).X();
    
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
      t0.CalculateFitParameters( fitPar, fSlices[0].Param().Bz() );
    }
  
    Float_t err2Y, err2Z;
    slice.GetErrors2( h.IRow(), t0, err2Y, err2Z );
    if( !t.Filter2NoCos( h.Y(), h.Z(), err2Y, err2Z ) ) continue;	  	    
    first = 0;

    if( CAMath::Abs( t0.CosPhi() )>1.e-4 ){
      Float_t dLdX = CAMath::Sqrt(1.+t0.DzDs()*t0.DzDs())/CAMath::Abs(t0.CosPhi());
      sumDeDx+=h.Amp()/dLdX;
      nDeDx++;
    }
    hitsNew[nHitsNew++] = hits[jhit];
  }
  

  DeDx = 0;
  if( nDeDx >0 ) DeDx = sumDeDx/nDeDx;
	
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
    NTrackHits = nHitsNew;
    for( Int_t i=0; i<NTrackHits; i++ ){
      hits[dir ?(NTrackHits-1-i) :i] = hitsNew[i];
    }
  }
  return ok;
}


Float_t AliHLTTPCCAGBTracker::GetChi2( Float_t x1, Float_t y1, Float_t a00, Float_t a10, Float_t a11, 
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

  
void AliHLTTPCCAGBTracker::MakeBorderTracks( Int_t iSlice, Int_t iBorder, AliHLTTPCCABorderTrack B[], Int_t &nB )
{
  //* prepare slice tracks for merging
  static int statAll=0, statOK=0;  
  nB = 0;
  AliHLTTPCCATracker &slice = fSlices[iSlice];
  Float_t dAlpha = ( fSlices[1].Param().Alpha() - fSlices[0].Param().Alpha() )/2;
  Float_t x0 = 0;

  if( iBorder==0 ){
    dAlpha = dAlpha - CAMath::Pi()/2 ;
  } else if( iBorder==1 ){
    dAlpha = -dAlpha - CAMath::Pi()/2 ;  
  } else if( iBorder==2 ){
    dAlpha = dAlpha;
    x0 = slice.Row(63).X();
  }else if( iBorder==3 ){
    dAlpha = -dAlpha;
    x0 = slice.Row(63).X();
  } else if( iBorder==4 ){
    dAlpha = 0;
    x0 = slice.Row(63).X();
  }

  for (Int_t itr=0; itr<*slice.NOutTracks(); itr++) {
    AliHLTTPCCAOutTrack &t = slice.OutTracks()[itr];

    AliHLTTPCCATrackParam t0 = t.StartPoint();
    AliHLTTPCCATrackParam t1 = t.EndPoint();
    //const Float_t maxSin = .9;

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
      b.fNHits = t.NHits();
      b.fIRow = fHits[ fFirstSliceHit[iSlice] + slice.OutTrackHits()[t.FirstHitRef()+0]].IRow();
      b.fParam = t0;
      b.fX = t0.GetX();
      if( b.fParam.TransportToX( x0, maxSin ) ) nB++;
      //else std::cout<<"0: can not transport to x="<<x0<<std::endl;

    }
    if( do1 ){
      AliHLTTPCCABorderTrack &b = B[nB];
      b.fOK = 1;
      b.fITrack = itr;
      b.fNHits = t.NHits();
      b.fIRow = fHits[ fFirstSliceHit[iSlice] + slice.OutTrackHits()[t.FirstHitRef()+t.NHits()-1]].IRow();
      b.fParam = t1;    
      b.fX = t0.GetX();
      if( b.fParam.TransportToX( x0, maxSin ) ) nB++;
      //else std::cout<<"1: can not transport to x="<<x0<<std::endl;
    }
    if( do0 || do1 ) statOK++;
    statAll++;
  }
  //std::cout<<"\n\n Stat all, stat ok = "<<statAll<<" "<<statOK<<std::endl;
}


void AliHLTTPCCAGBTracker::SplitBorderTracks( Int_t iSlice1, AliHLTTPCCABorderTrack B1[], Int_t N1,
					      Int_t iSlice2, AliHLTTPCCABorderTrack B2[], Int_t N2, 
					      Float_t Alpha 
					      )
{
  //* split two sets of tracks

  Float_t factor2y = 10;//2.6;
  Float_t factor2z = 10;//4.0;
  Float_t factor2s = 10;//2.6;
  Float_t factor2t = 10;//2.0;
  Float_t factor2k = 2.0;//2.2;

  Float_t factor2ys = 1.;//1.5;//SG!!!
  Float_t factor2zt = 1.;//1.5;//SG!!!

  AliHLTTPCCATracker &slice1 = fSlices[iSlice1];
  AliHLTTPCCATracker &slice2 = fSlices[iSlice2];
 
  factor2y = 3.5*3.5*factor2y*factor2y;
  factor2z = 3.5*3.5*factor2z*factor2z;
  factor2s = 3.5*3.5*factor2s*factor2s;
  factor2t = 3.5*3.5*factor2t*factor2t;
  factor2k = 3.5*3.5*factor2k*factor2k;
  factor2ys = 3.5*3.5*factor2ys*factor2ys;
  factor2zt = 3.5*3.5*factor2zt*factor2zt;

  Int_t minNPartHits = 10;//SG!!!
  Int_t minNTotalHits = 20;
  //Float_t maxDX = slice1.Row(40).X() -  slice1.Row(0).X();

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

      if( Alpha!=-1 ){
#ifdef DRAW
	std::cout<<"Try to merge tracks "<<i1<<" and "<<i2<<":"<<std::endl;
	t1.Print();
	t2.Print();

	Float_t c= t2.CosPhi()*t1.CosPhi()>0 ?1 :-1;
	Float_t dy = t2.GetY() - t1.GetY();
	Float_t dz = t2.GetZ() - t1.GetZ();
	Float_t ds = t2.GetSinPhi() - c*t1.GetSinPhi();
	Float_t dt = t2.GetDzDs() - c*t1.GetDzDs();
	Float_t dk = t2.GetKappa() - c*t1.GetKappa(); 
	
	Float_t chi2ys = GetChi2( t1.GetY(),t1.GetSinPhi(),t1.GetCov()[0],t1.GetCov()[3],t1.GetCov()[5], 
				  t2.GetY(),t2.GetSinPhi(),t2.GetCov()[0],t2.GetCov()[3],t2.GetCov()[5] );
	Float_t chi2zt = GetChi2( t1.GetZ(),t1.GetDzDs(),t1.GetCov()[2],t1.GetCov()[7],t1.GetCov()[9], 
				  t2.GetZ(),t2.GetDzDs(),t2.GetCov()[2],t2.GetCov()[7],t2.GetCov()[9] );



	Float_t sy2 = t2.GetErr2Y() + t1.GetErr2Y();
	Float_t sz2 = t2.GetErr2Z() + t1.GetErr2Z();
	Float_t ss2 = t2.GetErr2SinPhi() + t1.GetErr2SinPhi();
	Float_t st2 = t2.GetErr2DzDs() + t1.GetErr2DzDs();
	Float_t sk2 = t2.GetErr2Kappa() + t1.GetErr2Kappa();
	
	std::cout<<"dy, sy= "<<dy<<" "<<CAMath::Sqrt(factor2y*sy2)<<std::endl;
	std::cout<<"dz, sz= "<<dz<<" "<<CAMath::Sqrt(factor2z*sz2)<<std::endl;
	std::cout<<"ds, ss= "<<ds<<" "<<CAMath::Sqrt(factor2s*ss2)<<std::endl;
	std::cout<<"dt, st= "<<dt<<" "<<CAMath::Sqrt(factor2t*st2)<<std::endl;
	std::cout<<"dk, sk= "<<dk<<" "<<CAMath::Sqrt(factor2k*sk2)<<std::endl;

	std::cout<<"dys, sy= "<<CAMath::Sqrt(chi2ys)<<" "<<CAMath::Sqrt(factor2y)<<std::endl;
	std::cout<<"dzt, st= "<<CAMath::Sqrt(chi2zt)<<" "<<CAMath::Sqrt(factor2z)<<std::endl;

	if( dy*dy<factor2y*sy2
	    && dz*dz<factor2z*sz2 
	    && ds*ds<factor2s*ss2 
	    && dt*dt<factor2t*st2 
	    && dk*dk<factor2k*sk2 
	    ){	    
	  std::cout<<"tracks are merged"<<std::endl;
	} else std::cout<<"tracks are not merged"<<std::endl;

	AliHLTTPCCADisplay::Instance().ClearView();
	AliHLTTPCCADisplay::Instance().SetTPCView();
	AliHLTTPCCADisplay::Instance().DrawTPC();
	AliHLTTPCCADisplay::Instance().DrawGBHits( *this, kGreen, 1. );
	AliHLTTPCCADisplay::Instance().SetCurrentSlice(&slice1);
	AliHLTTPCCADisplay::Instance().DrawSliceOutTrack( t1, Alpha, b1.fITrack, kRed, 2. );    
	AliHLTTPCCADisplay::Instance().SetCurrentSlice(&slice2);
	AliHLTTPCCADisplay::Instance().DrawSliceOutTrack( t2, Alpha, b2.fITrack, kBlue, 2. );          
	AliHLTTPCCADisplay::Instance().Ask();
#endif
      }

      Float_t chi2ys = GetChi2( t1.GetY(),t1.GetSinPhi(),t1.GetCov()[0],t1.GetCov()[3],t1.GetCov()[5], 
				t2.GetY(),t2.GetSinPhi(),t2.GetCov()[0],t2.GetCov()[3],t2.GetCov()[5] );
      Float_t chi2zt = GetChi2( t1.GetZ(),t1.GetDzDs(),t1.GetCov()[2],t1.GetCov()[7],t1.GetCov()[9], 
				t2.GetZ(),t2.GetDzDs(),t2.GetCov()[2],t2.GetCov()[7],t2.GetCov()[9] );
      if( chi2ys>factor2ys ) continue;
      if( chi2zt>factor2zt ) continue;

      Float_t dy = t2.GetY() - t1.GetY();
      Float_t sy2 = t2.GetErr2Y() + t1.GetErr2Y();
      if( dy*dy>factor2y*sy2 ) continue;
	
      Float_t dz = t2.GetZ() - t1.GetZ();
      Float_t sz2 = t2.GetErr2Z() + t1.GetErr2Z();
      if( dz*dz>factor2z*sz2 ) continue;	
	
      Float_t d, s2;
      Float_t c= t2.CosPhi()*t1.CosPhi()>0 ?1 :-1;

      d = t2.GetSinPhi() - c*t1.GetSinPhi();
      s2 = t2.GetErr2SinPhi() + t1.GetErr2SinPhi();
      if( d*d>factor2s*s2 ) continue;
      d = t2.GetDzDs() - c*t1.GetDzDs();
      s2 = t2.GetErr2DzDs() + t1.GetErr2DzDs();
      if( d*d>factor2t*s2 ) continue;
      d = t2.GetKappa() - c*t1.GetKappa(); 
      s2 = t2.GetErr2Kappa() + t1.GetErr2Kappa();
      if( d*d>factor2k*s2 ) continue;
	
      lBest2 = b2.fNHits;
      iBest2 = b2.fITrack;
    }
      
    if( iBest2>=0 ){	
      Int_t old1 = fSliceTrackInfos[iSlice2][iBest2].fPrevNeighbour;
      if( old1 >= 0 ){
	if( slice1.OutTracks()[ old1 ].NHits() < slice1.OutTracks()[ b1.fITrack ].NHits() ){
	  fSliceTrackInfos[iSlice2][iBest2].fPrevNeighbour = -1;
	  fSliceTrackInfos[iSlice1][old1].fNextNeighbour = -1;	
	} else continue;
      }
      Int_t old2 = fSliceTrackInfos[iSlice1][b1.fITrack].fNextNeighbour;
      if( old2 >= 0 ){
	if( slice2.OutTracks()[ old2 ].NHits() < slice2.OutTracks()[ iBest2 ].NHits() ){
	  fSliceTrackInfos[iSlice2][old2].fPrevNeighbour = -1;
	} else continue;
      }
      fSliceTrackInfos[iSlice1][b1.fITrack].fNextNeighbour = iBest2;
      fSliceTrackInfos[iSlice2][iBest2].fPrevNeighbour = b1.fITrack;	
    }  
  }
}


void AliHLTTPCCAGBTracker::Merging()
{
  //* track merging between slices

#ifdef DRAW
  AliHLTTPCCADisplay &disp = AliHLTTPCCADisplay::Instance();
  if(0){
  disp.SetSliceView();
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    Int_t nh=fFirstSliceHit[iSlice+1]-fFirstSliceHit[iSlice];
    if( nh<=0 ) continue;
    disp.SetCurrentSlice(&slice);
    disp.DrawSlice( &slice );
    //disp.DrawGBHits( *this, -1, .5 );
    disp.DrawSliceHits(-1,.5);
    disp.Ask();
    std::cout<<"N out tracks = "<<*slice.NOutTracks()<<std::endl;
    for( Int_t itr=0; itr<*slice.NOutTracks(); itr++ ){
      std::cout<<"track N "<<itr<<", nhits="<<slice.OutTracks()[itr].NHits()<<std::endl;
      disp.DrawSliceOutTrack( itr, kRed );
      disp.Ask();
      int id = slice.OutTracks()[itr].OrigTrackID();
      disp.DrawSliceTrack( id, kBlue );
      disp.Ask();
    }
    disp.Ask();
  }
  }
  AliHLTTPCCADisplay::Instance().SetTPCView();
  AliHLTTPCCADisplay::Instance().DrawTPC();
  AliHLTTPCCADisplay::Instance().DrawGBHits( *this );
  disp.Ask(); 
  std::cout<<"Slice tracks:"<<std::endl;
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    disp.SetCurrentSlice(&slice);    
    for( Int_t itr=0; itr<*slice.NOutTracks(); itr++ ){
      disp.DrawSliceOutTrack( itr, kBlue, 2. );
    }
  }
  //AliHLTTPCCADisplay::Instance().DrawGBHits( *this );
  disp.Ask(); 
 
#endif //DRAW  

  Int_t nextSlice[100], prevSlice[100];

  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    nextSlice[iSlice] = iSlice + 1;
    prevSlice[iSlice] = iSlice - 1;
  }
  Int_t mid = NSlices()/2 - 1 ;
  Int_t last = NSlices() - 1 ;
  if( mid<0 ) mid = 0; // to avoid compiler warning
  if( last<0 ) last = 0; // 
  nextSlice[ mid ] = 0;
  prevSlice[ 0 ] = mid;
  nextSlice[ last ] = fNSlices/2;
  prevSlice[ fNSlices/2 ] = last;
  
  Int_t maxNSliceTracks = 0;
  for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    AliHLTTPCCATracker &iS = fSlices[iSlice];
    if( maxNSliceTracks < *iS.NOutTracks() ) maxNSliceTracks = *iS.NOutTracks();
  }
  
  if(1){// merging segments withing one slice //SG!!!

    AliHLTTPCCABorderTrack bord[maxNSliceTracks*10];

    for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){         
      //std::cout<<" merging tracks withing slice "<<iSlice<<":"<<std::endl;

#ifdef DRAW
  if(0){
    AliHLTTPCCADisplay &disp = AliHLTTPCCADisplay::Instance();
    std::cout<<" merging tracks withing slice "<<iSlice<<":"<<std::endl;
    disp.SetSliceView();
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    Int_t nh=fFirstSliceHit[iSlice+1]-fFirstSliceHit[iSlice];
    if( nh>0 ){
      disp.SetCurrentSlice(&slice);
      disp.DrawSlice( &slice );
      disp.DrawSliceHits(-1,.5);
      std::cout<<"N out tracks = "<<*slice.NOutTracks()<<std::endl;
      for( Int_t itr=0; itr<*slice.NOutTracks(); itr++ ){
	std::cout<<"track N "<<itr<<", nhits="<<slice.OutTracks()[itr].NHits()<<std::endl;
	disp.DrawSliceOutTrack( itr, kRed );
	//disp.Ask();
	//int id = slice.OutTracks()[itr].OrigTrackID();
	//disp.DrawSliceTrack( id, kBlue );
	//disp.Ask();
      }
      disp.Ask();
    }  
  }  
#endif //DRAW  


      AliHLTTPCCATracker &iS = fSlices[iSlice];
      Int_t nBord=0;
      MakeBorderTracks( iSlice, 4, bord, nBord );
#ifdef DRAW
      std::cout<<"\nMerge tracks withing slice "<<iSlice<<":\n"<<std::endl;
#endif
      Float_t alph = -1;//iS.Param().Alpha();
      SplitBorderTracks( iSlice, bord, nBord, iSlice, bord, nBord, alph );    

      AliHLTTPCCAOutTrack tmpT[*iS.NOutTracks()];
      Int_t tmpH[*iS.NOutTrackHits()];
      Int_t nTr=0, nH=0;
      for( Int_t itr=0; itr<*iS.NOutTracks(); itr++ ){
	fSliceTrackInfos[iSlice][itr].fPrevNeighbour = -1;
	if( fSliceTrackInfos[iSlice][itr].fNextNeighbour == -2 ){
	  fSliceTrackInfos[iSlice][itr].fNextNeighbour = -1;
	  continue;
	}
	AliHLTTPCCAOutTrack &it = iS.OutTracks()[itr];
	AliHLTTPCCAOutTrack &t = tmpT[nTr];
	t = it;
	t.SetFirstHitRef( nH );
	for( Int_t ih=0; ih<it.NHits(); ih++ ) tmpH[nH+ih] = iS.OutTrackHits()[it.FirstHitRef()+ih];
	nTr++;
	nH+=it.NHits();

	int jtr =  fSliceTrackInfos[iSlice][itr].fNextNeighbour;

	if( jtr<0 ) continue;
	fSliceTrackInfos[iSlice][itr].fNextNeighbour = -1;
	fSliceTrackInfos[iSlice][jtr].fNextNeighbour = -2;

	AliHLTTPCCAOutTrack &jt = iS.OutTracks()[jtr];
	for( Int_t ih=0; ih<jt.NHits(); ih++ ) tmpH[nH+ih] = iS.OutTrackHits()[jt.FirstHitRef()+ih];
	t.SetNHits( t.NHits() + jt.NHits() );
	nH+=jt.NHits();	
	if( jt.StartPoint().X() < it.StartPoint().X() ) t.SetStartPoint( jt.StartPoint() );
	if( jt.EndPoint().X() > it.EndPoint().X() ) t.SetEndPoint( jt.EndPoint() );
      }
      
      *iS.NOutTracks() = nTr;
      *iS.NOutTrackHits() = nH;
      for( Int_t itr=0; itr<nTr; itr++ ) iS.OutTracks()[itr] = tmpT[itr];
      for( Int_t ih=0; ih<nH; ih++ ) iS.OutTrackHits()[ih] = tmpH[ih];
      //std::cout<<"\nMerge tracks withing slice "<<iSlice<<" ok\n"<<std::endl;

#ifdef DRAW
  if(0){
    AliHLTTPCCADisplay &disp = AliHLTTPCCADisplay::Instance();
    std::cout<<" merginged tracks withing slice "<<iSlice<<":"<<std::endl;
    disp.SetSliceView();
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    Int_t nh=fFirstSliceHit[iSlice+1]-fFirstSliceHit[iSlice];
    if( nh>0 ){
      disp.SetCurrentSlice(&slice);
      disp.DrawSlice( &slice );
      disp.DrawSliceHits(-1,.5);
      std::cout<<"N out tracks = "<<*slice.NOutTracks()<<std::endl;
      for( Int_t itr=0; itr<*slice.NOutTracks(); itr++ ){
	std::cout<<"track N "<<itr<<", nhits="<<slice.OutTracks()[itr].NHits()<<std::endl;
	disp.DrawSliceOutTrack( itr, kRed );
      }
      disp.Ask();
    }  
  }  
#endif //DRAW  

	  }
    }

#ifdef DRAW
    if(0){
    AliHLTTPCCADisplay &disp = AliHLTTPCCADisplay::Instance();
    AliHLTTPCCADisplay::Instance().SetTPCView();
    AliHLTTPCCADisplay::Instance().DrawTPC();
    AliHLTTPCCADisplay::Instance().DrawGBHits( *this );    
    std::cout<<"Slice tracks:"<<std::endl;
    for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
      AliHLTTPCCATracker &slice = fSlices[iSlice];
      disp.SetCurrentSlice(&slice);    
      for( Int_t itr=0; itr<*slice.NOutTracks(); itr++ ){
	disp.DrawSliceOutTrack( itr, -1, 2. );
      }
    }
    //AliHLTTPCCADisplay::Instance().DrawGBHits( *this );
    disp.Ask(); 
  }
#endif //DRAW  

      
  //* arrays for the rotated track parameters


    AliHLTTPCCABorderTrack 
      *bCurr0 = new AliHLTTPCCABorderTrack[maxNSliceTracks*10], 
      *bNext0 = new AliHLTTPCCABorderTrack[maxNSliceTracks*10],
      *bCurr = new AliHLTTPCCABorderTrack[maxNSliceTracks*10], 
      *bNext = new AliHLTTPCCABorderTrack[maxNSliceTracks*10];

    for( Int_t iSlice=0; iSlice<fNSlices; iSlice++ ){
    
    Int_t jSlice = nextSlice[iSlice];

#ifdef DRAW
    std::cout<<" Merging slices "<<iSlice<<" and "<<jSlice<<std::endl;
#endif
    //AliHLTTPCCATracker &iS = fSlices[iSlice];
    //AliHLTTPCCATracker &jS = fSlices[jSlice];    

    Int_t nCurr0 = 0, nNext0 = 0;
    Int_t nCurr = 0, nNext = 0;

    MakeBorderTracks( iSlice, 0, bCurr, nCurr );
    MakeBorderTracks( jSlice, 1, bNext, nNext );
    MakeBorderTracks( iSlice, 2, bCurr0, nCurr0 );
    MakeBorderTracks( jSlice, 3, bNext0, nNext0 );

#ifdef DRAW
    std::cout<<"\nMerge0 tracks :\n"<<std::endl;
#endif
    Float_t alph = -1;//iS.Param().Alpha() + ( fSlices[1].Param().Alpha() - fSlices[0].Param().Alpha() )/2;
    Float_t alph1 = -1;//alph - CAMath::Pi()/2;
    SplitBorderTracks( iSlice, bCurr0, nCurr0, jSlice, bNext0, nNext0, alph );   //SG!!!
#ifdef DRAW
    std::cout<<"\nMerge1 tracks :\n"<<std::endl;
#endif
    SplitBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext, alph1 );    //SG!!!
    }

    if( bCurr0 ) delete[] bCurr0;
    if( bNext0 ) delete[] bNext0;
    if( bCurr  ) delete[] bCurr;
    if( bNext  ) delete[] bNext;

  TStopwatch timerMerge2;

  Int_t nTracksTot = 0;
  for( Int_t iSlice = 0; iSlice<fNSlices; iSlice++ ){    
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    nTracksTot+= *slice.NOutTracks();
  }
  
  if( fTrackHits ) delete[] fTrackHits;
  fTrackHits = 0;
  if(fTracks ) delete[] fTracks;
  fTracks = 0;
  fTrackHits = new Int_t [fNHits*100];//SG!!!
  fTracks = new AliHLTTPCCAGBTrack[nTracksTot];
  fNTracks = 0;

  Int_t nTrackHits = 0;

  //std::cout<<"\nStart global track creation...\n"<<std::endl;  
  
  //static Int_t nRejected = 0;

  //Int_t maxNRows = fSlices[0].Param().NRows();
  
  for( Int_t iSlice = 0; iSlice<fNSlices; iSlice++ ){

    AliHLTTPCCATracker &slice = fSlices[iSlice];
    for( Int_t itr=0; itr<*slice.NOutTracks(); itr++ ){
      if( fSliceTrackInfos[iSlice][itr].fUsed ) continue;
      if( fSliceTrackInfos[iSlice][itr].fPrevNeighbour>=0 ) continue;
      //std::cout<<"\n slice "<<iSlice<<", track "<<itr<<"\n"<<std::endl;
      AliHLTTPCCAOutTrack &tCA = slice.OutTracks()[itr];
      AliHLTTPCCAGBTrack &t = fTracks[fNTracks];

      AliHLTTPCCATrackParam startPoint = tCA.StartPoint(), endPoint = tCA.EndPoint();
      Float_t startAlpha = slice.Param().Alpha(), endAlpha = slice.Param().Alpha();
      t.SetNHits( 0 );

      t.SetFirstHitRef( nTrackHits );

      Int_t hits[2000];
      Int_t firstHit = 1000;
      Int_t nHits = 0;
      Int_t jSlice = iSlice;
      Int_t jtr = itr;
      {
	fSliceTrackInfos[jSlice][jtr].fUsed = 1;
	for( Int_t jhit=0; jhit<tCA.NHits(); jhit++){
	  Int_t id = fFirstSliceHit[iSlice] + slice.OutTrackHits()[tCA.FirstHitRef()+jhit];	  
	  hits[firstHit+jhit] = id;
	}
	nHits=tCA.NHits();
	jtr = fSliceTrackInfos[iSlice][itr].fNextNeighbour;
	jSlice = nextSlice[iSlice];			
      } 
      while( jtr >=0 ){
	if( fSliceTrackInfos[jSlice][jtr].fUsed ) break;
	fSliceTrackInfos[jSlice][jtr].fUsed = 1;
	AliHLTTPCCATracker &jslice = fSlices[jSlice];
	AliHLTTPCCAOutTrack &jTr = jslice.OutTracks()[jtr];
	Bool_t dir = 0;
	Int_t startHit = firstHit+ nHits;
	Float_t d00 = startPoint.GetDistXZ2(jTr.StartPoint() );
	Float_t d01 = startPoint.GetDistXZ2(jTr.EndPoint() );
	Float_t d10 = endPoint.GetDistXZ2(jTr.StartPoint() );
	Float_t d11 = endPoint.GetDistXZ2(jTr.EndPoint() );
	if( d00<=d01 && d00<=d10 && d00<=d11 ){
	  startPoint = jTr.EndPoint();
	  startAlpha = jslice.Param().Alpha();
	  dir = 1;
	  firstHit -= jTr.NHits();
	  startHit = firstHit;
	}else if( d01<=d10 && d01<=d11 ){
	  startPoint = jTr.StartPoint();
	  startAlpha = jslice.Param().Alpha();
	  dir = 0;
	  firstHit -= jTr.NHits();
	  startHit = firstHit;
	}else if( d10<=d11 ){
	  endPoint = jTr.EndPoint();
	  endAlpha = jslice.Param().Alpha();
	  dir = 0;
	}else{
	  endPoint = jTr.StartPoint();
	  endAlpha = jslice.Param().Alpha();
	  dir = 1;
	}
	
	for( Int_t jhit=0; jhit<jTr.NHits(); jhit++){
	  Int_t id = fFirstSliceHit[jSlice] + jslice.OutTrackHits()[jTr.FirstHitRef()+jhit];	  
	  hits[startHit+(dir ?(jTr.NHits()-1-jhit) :jhit)] = id;
	}
	nHits+=jTr.NHits();
	jtr = fSliceTrackInfos[jSlice][jtr].fNextNeighbour;
	jSlice = nextSlice[jSlice];			
      }

      if( endPoint.X() < startPoint.X() ){ // swap
	for( Int_t i=0; i<nHits; i++ ) hits[i] = hits[firstHit+nHits-1-i];
	firstHit = 0;
      }

      if( nHits < 30 ) continue;     //SG!!!

      // refit 
      Float_t dEdX=0;
      if( !FitTrack( endPoint, startPoint, startAlpha, hits+firstHit, nHits, dEdX, 0 ) ) continue;
      endAlpha = startAlpha;
      if( !FitTrack( startPoint, endPoint, startAlpha, hits+firstHit, nHits, dEdX, 1 ) ) continue;

      if( nHits < 30 ) continue;     //SG!!!
    
      
      t.SetNHits( nHits );
      t.SetParam( startPoint );
      t.SetAlpha( startAlpha );
      t.SetDeDx( dEdX );

      for( Int_t i = 0; i<nHits; i++ ){
	fTrackHits[nTrackHits+i] = hits[firstHit+i];
      }	        

      AliHLTTPCCATrackParam p = t.Param();
      AliHLTTPCCATrackParam::AliHLTTPCCATrackFitParam fitPar;
      p.CalculateFitParameters( fitPar, fSlices[0].Param().Bz() );

      Double_t dAlpha = 0;
      {
	Double_t xTPC=83.65;
	Double_t ddAlpha = 0.00609235;
	
	if( p.TransportToXWithMaterial( xTPC, fitPar ) ){
	  Double_t y=p.GetY();
	  Double_t ymax=xTPC*CAMath::Tan(dAlpha/2.); 
	  if (y > ymax) {
	    if( p.Rotate( ddAlpha ) ){ dAlpha=ddAlpha;  p.TransportToXWithMaterial( xTPC, fitPar ); }
	  } else if (y <-ymax) {
	    if( p.Rotate( -ddAlpha ) ){  dAlpha=-ddAlpha; p.TransportToXWithMaterial( xTPC, fitPar );}
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
      t.SetParam( p );
      t.SetAlpha( t.Alpha() + dAlpha );
      nTrackHits+= t.NHits();
      fNTracks++;   
    }
  }
  
  //std::cout<<"\n\nRejected: "<<nRejected<<"\n"<<std::endl;
  timerMerge2.Stop();
  fStatTime[11]+=timerMerge2.CpuTime();

  TStopwatch timerMerge3;

  //* selection  
  //std::cout<<"Selection..."<<std::endl;
  if(0){
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
      to.SetFirstHitRef( nHits );
      to.SetNHits( 0 );
      for( Int_t ih=0; ih<t.NHits(); ih++ ){
	Int_t jh = fTrackHits[t.FirstHitRef()+ih];
	AliHLTTPCCAGBHit &h = fHits[jh];
	if( h.IsUsed() ) continue;
	vhits[to.FirstHitRef() + to.NHits()] = jh;
	to.SetNHits( to.NHits()+1);
	h.SetIsUsed( 1 );
      }
      //if( to.NHits()<10 ) continue;//SG!!!
      nHits+=to.NHits();
      nTracks++;
      //std::cout<<to.Param().GetErr2Y()<<" "<<to.Param().GetErr2Z()<<std::endl;
    }
    fNTracks = nTracks;
    if( fTrackHits ) delete[] fTrackHits;
    if( fTracks ) delete[] fTracks;
    fTrackHits = vhits;
    fTracks = vtracks;
    if( vptracks ) delete[] vptracks;
  }
  timerMerge3.Stop();
  fStatTime[12]+=timerMerge3.CpuTime();

#ifdef DRAW
  std::cout<<"Global tracks: "<<std::endl;
  AliHLTTPCCADisplay::Instance().ClearView();
  AliHLTTPCCADisplay::Instance().SetTPCView();
  AliHLTTPCCADisplay::Instance().DrawTPC();
  AliHLTTPCCADisplay::Instance().DrawGBHits( *this );
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    std::cout<<itr<<" nhits= "<<fTracks[itr].NHits()<<std::endl;
    AliHLTTPCCADisplay::Instance().DrawGBTrack( itr, kBlue, 2. );    
    //AliHLTTPCCADisplay::Instance().Ask();
  }
  AliHLTTPCCADisplay::Instance().Ask();
#endif
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

void AliHLTTPCCAGBTracker::WriteSettings( std::ostream &out ) const
{
  //* write settings to the file
  out<< NSlices()<<std::endl;  
  for( Int_t iSlice=0; iSlice<NSlices(); iSlice++ ){    
    fSlices[iSlice].Param().WriteSettings( out );
  }
}

void AliHLTTPCCAGBTracker::ReadSettings( std::istream &in )
{
  //* Read settings from the file
  Int_t nSlices=0;
  in >> nSlices;
  SetNSlices( nSlices );
  for( Int_t iSlice=0; iSlice<NSlices(); iSlice++ ){    
    AliHLTTPCCAParam param;
    param.ReadSettings ( in );
    fSlices[iSlice].Initialize( param ); 
  }
}

void AliHLTTPCCAGBTracker::WriteEvent( std::ostream &out ) const
{
  // write event to the file

  out<<NHits()<<std::endl;
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
    out<<h.IRow()<<std::endl;
  }
}

void AliHLTTPCCAGBTracker::ReadEvent( std::istream &in ) 
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

void AliHLTTPCCAGBTracker::WriteTracks( std::ostream &out ) const 
{
  //* Write tracks to file 

  out<<fSliceTrackerTime<<std::endl;
  Int_t nTrackHits = 0;
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    nTrackHits+=fTracks[itr].NHits();
  }
  out<<nTrackHits<<std::endl;
  for( Int_t ih=0; ih<nTrackHits; ih++ ){
    out<< fTrackHits[ih]<<" ";
  }
  out<<std::endl;
  
  out<<NTracks()<<std::endl;
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    AliHLTTPCCAGBTrack &t = fTracks[itr];    
    const AliHLTTPCCATrackParam &p = t.Param();	
    out<< t.NHits()<<" ";
    out<< t.FirstHitRef()<<" ";
    out<< t.Alpha()<<" ";
    out<< t.DeDx()<<std::endl;
    out<< p.GetX()<<" ";
    out<< p.GetCosPhi()<<" ";
    out<< p.GetChi2()<<" ";
    out<< p.GetNDF()<<std::endl;
    for( Int_t i=0; i<5; i++ ) out<<p.GetPar()[i]<<" ";
    out<<std::endl;
    for( Int_t i=0; i<15; i++ ) out<<p.GetCov()[i]<<" ";
    out<<std::endl;
  }
}

void AliHLTTPCCAGBTracker::ReadTracks( std::istream &in )
{
  //* Read tracks  from file 

  in>>fTime;
  fSliceTrackerTime = fTime;
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
    AliHLTTPCCATrackParam p;
    Int_t i;
    Float_t f;
    in>>i;
    t.SetNHits(i);
    in>>i;
    t.SetFirstHitRef( i );
    in>>f;
    t.SetAlpha(f);
    in>>f;
    t.SetDeDx( f );
    in>>f;
    p.SetX( f );
    in>>f;
    p.SetCosPhi( f );
    in>>f;
    p.SetChi2( f );
    in>>i;
    p.SetNDF( i );
    for( Int_t j=0; j<5; j++ ){ in>>f; p.SetPar( j, f); }
    for( Int_t j=0; j<15; j++ ){ in>>f; p.SetCov(j,f); }
    t.SetParam(p);	
  }
}

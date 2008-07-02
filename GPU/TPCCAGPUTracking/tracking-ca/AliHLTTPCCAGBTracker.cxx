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

#include "TMath.h"
#include "TStopwatch.h"
#include "Riostream.h"
#include "TROOT.h"

//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#include "TApplication.h"
#endif //DRAW

ClassImp(AliHLTTPCCAGBTracker)

AliHLTTPCCAGBTracker::AliHLTTPCCAGBTracker()
  : TObject(),
    fSlices(0), 
    fNSlices(0), 
    fHits(0),
    fNHits(0),
    fTrackHits(0), 
    fTracks(0), 
    fNTracks(0),
    fSliceTrackInfos(0),
    fStatNEvents(0)
{
  //* constructor
  fStatTime[0] = 0;
  fStatTime[1] = 0;
  fStatTime[2] = 0;
  fStatTime[3] = 0;
  fStatTime[4] = 0;
  fStatTime[5] = 0;
  fStatTime[6] = 0;
  fStatTime[7] = 0;
  fStatTime[8] = 0;
  fStatTime[9] = 0;
}

AliHLTTPCCAGBTracker::AliHLTTPCCAGBTracker(const AliHLTTPCCAGBTracker&)
  : TObject(),
    fSlices(0), 
    fNSlices(0), 
    fHits(0),
    fNHits(0),
    fTrackHits(0), 
    fTracks(0), 
    fNTracks(0),
    fSliceTrackInfos(0),
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

void AliHLTTPCCAGBTracker::ReadHit( Double_t x, Double_t y, Double_t z, 
				    Double_t errY, Double_t errZ,
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
  hit.ID() = ID;
  hit.ISlice()=iSlice;
  hit.IRow() = iRow;
  hit.IsUsed() = 0;

  hit.SliceHit().Y() = y;
  hit.SliceHit().Z() = z;
  hit.SliceHit().ErrY() = errY;
  hit.SliceHit().ErrZ() = errZ;
  hit.SliceHit().ID() = 0;
  fNHits++;
}

void AliHLTTPCCAGBTracker::FindTracks()
{
  //* main tracking routine

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
  
  sort(fHits,fHits+fNHits, AliHLTTPCCAGBHit::Compare );

  // Read hits, row by row

  Int_t oldRow = -1;
  Int_t oldSlice = -1;
  Int_t nRowHits = 0;
  Int_t firstRowHit = 0;
  int nHitsTotal = fNHits;
  AliHLTTPCCAHit *vHits = new AliHLTTPCCAHit[nHitsTotal]; // CA hit array
  
  for( int ih=0; ih<nHitsTotal; ih++){
    AliHLTTPCCAGBHit &h = fHits[ih];
    h.SliceHit().ID() = ih;
    vHits[ih] = h.SliceHit();
    if( h.IRow() != oldRow || h.ISlice() != oldSlice ){
      if( oldRow>=0 && oldSlice>=0 ){	
	fSlices[oldSlice].ReadHitRow( oldRow, vHits+firstRowHit, nRowHits );
      }
      oldRow = h.IRow();
      oldSlice = h.ISlice();
      firstRowHit = ih;
      nRowHits = 0;
    }
    nRowHits++;    
  }	
  if( oldRow>=0 && oldSlice>=0 ){
    fSlices[oldSlice].ReadHitRow( oldRow, vHits+firstRowHit, nRowHits );
  }
  delete[] vHits;
  
  //cout<<"Start CA reconstruction"<<endl;

  for( int iSlice=0; iSlice<fNSlices; iSlice++ ){
    TStopwatch timer;
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    slice.Reconstruct();
    timer.Stop();
    fStatTime[0] += timer.CpuTime();
    fStatTime[1]+=slice.Timers()[0];
    fStatTime[2]+=slice.Timers()[1];
    fStatTime[3]+=slice.Timers()[2];
    fStatTime[4]+=slice.Timers()[3];
    fStatTime[5]+=slice.Timers()[4];
    fStatTime[6]+=slice.Timers()[5];
    fStatTime[7]+=slice.Timers()[6];
#ifdef DRAW
    //SlicePerformance( iSlice );
    //if( slice.NTracks()>0 ) AliHLTTPCCADisplay::Instance().Ask();
#endif 
  }
  //cout<<"End CA reconstruction"<<endl;
  
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
  fStatTime[8]+=timerMerge.CpuTime();
  //cout<<"End CA merging"<<endl;

#ifdef DRAW
  AliHLTTPCCADisplay::Instance().Ask();
#endif //DRAW
}

void AliHLTTPCCAGBTracker::Merging()
{
  //* track merging between slices

  Double_t dalpha = fSlices[1].Param().Alpha() - fSlices[0].Param().Alpha();
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
    AliHLTTPCCATracker &iS = fSlices[iSlice];
    Int_t jSlice = iSlice+1;
    if( iSlice==fNSlices/2-1 ) jSlice = 0;
    else if( iSlice==fNSlices-1 ) jSlice = fNSlices/2;
    AliHLTTPCCATracker &jS = fSlices[jSlice];    
    int iNTracks = iS.NOutTracks();
    int jNTracks = jS.NOutTracks();
    if( iNTracks<=0 || jNTracks<=0 ) continue;
    
    //* prepare slice tracks for merging
    
    for (Int_t itr=0; itr<iNTracks; itr++) {      
      iOK[0][itr] = 0;
      iOK[1][itr] = 0;
      if( iS.OutTracks()[itr].NHits()<30 ) continue;
      AliHLTTPCCATrackParam &iT1 = iTrParams[0][itr];
      AliHLTTPCCATrackParam &iT2 = iTrParams[1][itr];
      iT1 = iS.OutTracks()[itr].StartPoint();
      iT2 = iS.OutTracks()[itr].EndPoint();
      iOK[0][itr] = iT1.Rotate( dalpha/2 - TMath::Pi()/2 );
      iOK[1][itr] = iT2.Rotate( dalpha/2 - TMath::Pi()/2 );

      if( iOK[0][itr] ){
	iOK[0][itr] = iT1.TransportToX( 0 );
	if( iS.Param().RMin() > iT1.Y() || iS.Param().RMax() < iT1.Y() ) iOK[0][itr]=0;
      }
      if( iOK[1][itr] ){
	iOK[1][itr] = iT2.TransportToX( 0 );
	if( iS.Param().RMin() > iT2.Y() || iS.Param().RMax() < iT2.Y() ) iOK[1][itr]=0;
      }
    }

    for (Int_t jtr=0; jtr<jNTracks; jtr++) {      
      jOK[0][jtr] = 0;
      jOK[1][jtr] = 0;
      if( jS.OutTracks()[jtr].NHits()<30 ) continue;
      AliHLTTPCCATrackParam &jT1 = jTrParams[0][jtr];
      AliHLTTPCCATrackParam &jT2 = jTrParams[1][jtr];
      jT1 = jS.OutTracks()[jtr].StartPoint();
      jT2 = jS.OutTracks()[jtr].EndPoint();
      jOK[0][jtr] = jT1.Rotate( -dalpha/2 - TMath::Pi()/2 );
      jOK[1][jtr] = jT2.Rotate( -dalpha/2 - TMath::Pi()/2 );
      if( jOK[0][jtr] ){
	jOK[0][jtr] = jT1.TransportToX( 0 );
	if( jS.Param().RMin() > jT1.Y() || jS.Param().RMax() < jT1.Y() ) jOK[0][jtr]=0;
      }
      if( jOK[1][jtr] ){
	jOK[1][jtr] = jT2.TransportToX( 0 );
	if( jS.Param().RMin() > jT2.Y() || jS.Param().RMax() < jT2.Y() ) jOK[1][jtr]=0;
      }
    }

    //* start merging

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

  for( Int_t iSlice = 0; iSlice<fNSlices; iSlice++ ){
    
    AliHLTTPCCATracker &slice = fSlices[iSlice];
    for( Int_t itr=0; itr<slice.NOutTracks(); itr++ ){
      if( fSliceTrackInfos[iSlice][itr].fUsed ) continue;
      fSliceTrackInfos[iSlice][itr].fUsed = 1;
      AliHLTTPCCAOutTrack &tCA = slice.OutTracks()[itr];
      AliHLTTPCCAGBTrack &t = fTracks[fNTracks];
      t.Param() = tCA.StartPoint();
      t.NHits() = 0;
      t.FirstHitRef() = nTrackHits;
      t.Alpha() = slice.Param().Alpha();
            
      Int_t ihit = 0;
      Int_t jSlice = iSlice;
      Int_t jtr = itr;
      for( int jhit=0; jhit<tCA.NHits(); jhit++){
	int index = slice.OutTrackHits()[tCA.FirstHitRef()+jhit];
	fTrackHits[nTrackHits+ihit] = index;
	ihit++;	    
      }
      while( fSliceTrackInfos[jSlice][jtr].fNextNeighbour >=0 ){
	jtr = fSliceTrackInfos[jSlice][jtr].fNextNeighbour;
	if( jSlice==fNSlices/2-1 ) jSlice = 0;
	else if( jSlice==fNSlices-1 ) jSlice = fNSlices/2;
	else jSlice = jSlice + 1;
	if( fSliceTrackInfos[jSlice][jtr].fUsed ) break;
	fSliceTrackInfos[jSlice][jtr].fUsed = 1;
	AliHLTTPCCAOutTrack &jTr = fSlices[jSlice].OutTracks()[jtr];
	for( int jhit=0; jhit<jTr.NHits(); jhit++){
	  int index = fSlices[jSlice].OutTrackHits()[jTr.FirstHitRef()+jhit];	  
	  fTrackHits[nTrackHits+ihit] = index;
	  ihit++;
	}
      }
      t.NHits() = ihit;
      if( t.NHits()<50 ) continue;
      Int_t nHitsOld = t.NHits();

      // refit 
      {
	if( t.NHits()<10 ) continue;


	AliHLTTPCCAGBHit* *vhits = new AliHLTTPCCAGBHit*[nHitsOld];
	
	for( Int_t ih=0; ih<nHitsOld; ih++ ){
	  vhits[ih] = &(fHits[fTrackHits[t.FirstHitRef() + ih]]);
	}	
	
	sort(vhits, vhits+nHitsOld, AliHLTTPCCAGBHit::ComparePRowDown );

	AliHLTTPCCATrackParam t0;
	
	{	  
	  AliHLTTPCCAGBHit &h0 = *(vhits[0]);
	  AliHLTTPCCAGBHit &h1 = *(vhits[nHitsOld/2]);
	  AliHLTTPCCAGBHit &h2 = *(vhits[nHitsOld-1]);
	  if( h0.IRow()==h1.IRow() || h0.IRow()==h2.IRow() || h1.IRow()==h2.IRow() ) continue;
	  Double_t x1,y1,z1, x2, y2, z2, x3, y3, z3;
	  fSlices[h0.ISlice()].Param().Slice2Global(h0.X(), h0.Y(), h0.Z(), &x1,&y1,&z1);
	  fSlices[h1.ISlice()].Param().Slice2Global(h1.X(), h1.Y(), h1.Z(), &x2,&y2,&z2);
	  fSlices[h2.ISlice()].Param().Slice2Global(h2.X(), h2.Y(), h2.Z(), &x3,&y3,&z3);
	  fSlices[h0.ISlice()].Param().Global2Slice(x1, y1, z1, &x1,&y1,&z1);
	  fSlices[h0.ISlice()].Param().Global2Slice(x2, y2, z2, &x2,&y2,&z2);
	  fSlices[h0.ISlice()].Param().Global2Slice(x3, y3, z3, &x3,&y3,&z3);
	  Float_t sp0[5] = {x1, y1, z1, .5, .5 };	
	  Float_t sp1[5] = {x2, y2, z2, .5, .5 };
	  Float_t sp2[5] = {x3, y3, z3, .5, .5 };
	  t0.ConstructXYZ3(sp0,sp1,sp2,1., 0);
	}	      

	Int_t currslice = vhits[0]->ISlice();
	Int_t currrow = fSlices[currslice].Param().NRows()-1;
	Double_t currd2 = 1.e10;
	AliHLTTPCCAGBHit *currhit = 0;

	for( Int_t ih=0; ih<nHitsOld; ih++ ){
	  Double_t y0 = t0.GetY();
	  Double_t z0 = t0.GetZ();	
	  AliHLTTPCCAGBHit &h = *(vhits[ih]);
	  //cout<<"hit,slice,row="<<ih<<" "<<h.slice<<" "<<h.row<<endl;
	  if( h.ISlice() == currslice && h.IRow() == currrow ){
	    //* select the best hit in the row
	    Double_t dy = h.Y() - y0;
	    Double_t dz = h.Z() - z0;
	    Double_t d2 = dy*dy + dz*dz;
	    if( d2<currd2 ){
	      currhit = &h;
	      currd2 = d2;
	    }
	    continue;
	  }else{
	    if( currhit != 0 ){ 
	      //* update track	
	      t0.Filter(currhit->Y(), currhit->Z(), currhit->ErrY(), currhit->ErrZ());
	      currhit = 0;
	    }
	    if( h.ISlice() != currslice ){
	      //* Rotate to the new slice
	      currhit = 0;
	      if( !t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[h.ISlice()].Param().Alpha() ) ) break;	
	      currslice = h.ISlice();
	      //currrow = 300;
	      currd2 = 1.e10;
	    }
	    //* search for missed hits in rows
	    {
	      Double_t factor2 = 3.5*3.5;
	      AliHLTTPCCATracker &cslice = fSlices[currslice];	      
	      for( Int_t srow=currrow-1; srow>h.IRow(); srow--){
		AliHLTTPCCARow &row = cslice.Rows()[srow];
		if( !t0.TransportToX( row.X() ) ) continue;
		Int_t bestsh = -1;
		Double_t ds = 1.e10;
		for( Int_t ish=0; ish<row.NHits(); ish++ ){
		  AliHLTTPCCAHit &sh = row.Hits()[ish];
		  Double_t dy = sh.Y() - t0.GetY();
		  Double_t dz = sh.Z() - t0.GetZ();
		  Double_t dds = dy*dy+dz*dz;
		  if( dds<ds ){
		    ds = dds;
		    bestsh = ish;
		  }
		}
		if( bestsh<0 ) continue;
		AliHLTTPCCAHit &sh = row.Hits()[bestsh];
		Double_t dy = sh.Y() - t0.GetY();
		Double_t dz = sh.Z() - t0.GetZ();
		Double_t s2z = /*t0.GetErr2Z() + */ sh.ErrZ()*sh.ErrZ();
		if( dz*dz>factor2*s2z ) continue;		
		Double_t s2y = /*t0.GetErr2Y() + */ sh.ErrY()*sh.ErrY();
		if( dy*dy>factor2*s2y ) continue;
		//* update track	  
		t0.Filter(sh.Y(), sh.Z(), sh.ErrY()/.33, sh.ErrZ()/.33);
		fTrackHits[nTrackHits+t.NHits()] = sh.ID();
		t.NHits()++;
	      }
	    }
	    //* transport to the new row
	    currrow = h.IRow();  
	    Bool_t ok = t0.TransportToX( h.X() );	
	    if( ok ){
	      currrow = h.IRow();
	      Double_t dy = h.Y() - t0.GetY();
	      Double_t dz = h.Z() - t0.GetZ();
	      currd2 = dy*dy + dz*dz;
	      currhit = &h;
	    }
	  }
	}
	if( currhit != 0 ){ // update track
	  
	  t0.Filter(currhit->Y(), currhit->Z(), currhit->ErrY(), currhit->ErrZ());
	}
	
	//* search for missed hits in rows
	{
	  Double_t factor2 = 3.5*3.5;
	  for( Int_t srow=currrow-1; srow>=0; srow--){
	    AliHLTTPCCATracker *cslice = &(fSlices[currslice]);
	    AliHLTTPCCARow *row = &(cslice->Rows()[srow]);
	    if( !t0.TransportToX( row->X() ) ) continue;
	    if( t0.GetY() > row->MaxY() ){ //next slice
	      Int_t j = currslice+1;
	      if( currslice==fNSlices/2-1 ) j = 0;
	      else if( currslice==fNSlices-1 ) j = fNSlices/2;
	      //* Rotate to the new slice
	      if( !t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[j].Param().Alpha() ) ) break;
	      currslice = j;
	      cslice = &(fSlices[currslice]);
	      row = &(cslice->Rows()[srow]);
	      if( !t0.TransportToX( row->X() ) ) continue;		
	    }else if( t0.GetY() < -row->MaxY() ){ //prev slice
	      Int_t j = currslice-1;
	      if( currslice==0 ) j = fNSlices/2-1;
	      else if( currslice==fNSlices/2 ) j = fNSlices-1;
	      //* Rotate to the new slice
	      if( !t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[j].Param().Alpha() ) ) break;
	      currslice = j;
	      cslice = &(fSlices[currslice]);
	      row = &(cslice->Rows()[srow]);
	      if( !t0.TransportToX( row->X() ) ) continue;		
	    }
	    Int_t bestsh = -1;
	    Double_t ds = 1.e10;
	    for( Int_t ish=0; ish<row->NHits(); ish++ ){
	      AliHLTTPCCAHit &sh = row->Hits()[ish];
	      Double_t dy = sh.Y() - t0.GetY();
	      Double_t dz = sh.Z() - t0.GetZ();
	      Double_t dds = dy*dy+dz*dz;
	      if( dds<ds ){
		ds = dds;
		bestsh = ish;
	      }
	    }
	    if( bestsh<0 ) continue;
	    AliHLTTPCCAHit &sh = row->Hits()[bestsh];
	    Double_t dy = sh.Y() - t0.GetY();
	    Double_t dz = sh.Z() - t0.GetZ();
	    Double_t s2z = /*t0.GetErr2Z() + */ sh.ErrZ()*sh.ErrZ();
	    if( dz*dz>factor2*s2z ) continue;		
	    Double_t s2y = /*t0.GetErr2Y() + */ sh.ErrY()*sh.ErrY();
	    if( dy*dy>factor2*s2y ) continue;
	    //* update track	  
	    t0.Filter(sh.Y(), sh.Z(), sh.ErrY()/.33, sh.ErrZ()/.33);
	    fTrackHits[nTrackHits+t.NHits()] = sh.ID();
	    t.NHits()++;
	  }	
	}
	if( vhits ) delete[] vhits;

	//* search for missed hits in rows
	{
	  Double_t factor2 = 3.5*3.5;	
	  AliHLTTPCCAGBHit &h0 = fHits[fTrackHits[t.FirstHitRef()]];
	  
	  Bool_t ok = t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[h0.ISlice()].Param().Alpha() );
	  
	  currslice = h0.ISlice();
	  currrow = h0.IRow();
	  
	  for( Int_t srow=currrow+1; srow<fSlices[0].Param().NRows()&&ok; srow++){
	    AliHLTTPCCATracker *cslice = &(fSlices[currslice]);
	    AliHLTTPCCARow *row = &(cslice->Rows()[srow]);
	    if( !t0.TransportToX( row->X() ) ) continue;
	    if( t0.GetY() > row->MaxY() ){ //next slice
	      Int_t j = currslice+1;
	      if( currslice==fNSlices/2-1 ) j = 0;
	      else if( currslice==fNSlices-1 ) j = fNSlices/2;
	      //* Rotate to the new slice
	      if( !t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[j].Param().Alpha() ) ) break;
	      currslice = j;
	      cslice = &(fSlices[currslice]);
	      row = &(cslice->Rows()[srow]);
	      if( !t0.TransportToX( row->X() ) ) continue;		
	    }else if( t0.GetY() < -row->MaxY() ){ //prev slice
	      Int_t j = currslice-1;
	      if( currslice==0 ) j = fNSlices/2-1;
	      else if( currslice==fNSlices/2 ) j = fNSlices-1;
	      //* Rotate to the new slice
	      if( !t0.Rotate( -fSlices[currslice].Param().Alpha() +fSlices[j].Param().Alpha() ) ) break;
	      currslice = j;
	      cslice = &(fSlices[currslice]);
	      row = &(cslice->Rows()[srow]);
	      if( !t0.TransportToX( row->X() ) ) continue;		
	    }
	    Int_t bestsh = -1;
	    Double_t ds = 1.e10;
	    for( Int_t ish=0; ish<row->NHits(); ish++ ){
	      AliHLTTPCCAHit &sh = row->Hits()[ish];
	      Double_t dy = sh.Y() - t0.GetY();
	      Double_t dz = sh.Z() - t0.GetZ();
	      Double_t dds = dy*dy+dz*dz;
	      if( dds<ds ){
		ds = dds;
		bestsh = ish;
	      }
	    }
	    if( bestsh<0 ) continue;
	    AliHLTTPCCAHit &sh = row->Hits()[bestsh];
	    Double_t dy = sh.Y() - t0.GetY();
	    Double_t dz = sh.Z() - t0.GetZ();
	    Double_t s2z = /*t0.GetErr2Z() + */ sh.ErrZ()*sh.ErrZ();
	    if( dz*dz>factor2*s2z ) continue;		
	    Double_t s2y = /*t0.GetErr2Y() + */ sh.ErrY()*sh.ErrY();
	    if( dy*dy>factor2*s2y ) continue;
	    //* update track	  
	    t0.Filter(sh.Y(), sh.Z(), sh.ErrY()/.33, sh.ErrZ()/.33);
	    fTrackHits[nTrackHits+t.NHits()] = sh.ID();
	    t.NHits()++;
	  }	
	}
	
	Bool_t ok=1;
	{
	  for( int i=0; i<15; i++ ) ok = ok && finite(t0.Cov()[i]);
	  for( int i=0; i<5; i++ ) ok = ok && finite(t0.Par()[i]);
	  ok = ok && (t0.GetX()>50);
	}
	if(!ok) continue;
	if( TMath::Abs(t0.Kappa())<1.e-8 ) t0.Kappa() = 1.e-8;
	if( nHitsOld != t.NHits() ){
	  //cout<<"N matched hits = "<<(t.NHits() - nHitsOld )<<" / "<<nHitsOld<<endl;
	}

	t.Param() = t0;
	t.Alpha() = fSlices[currslice].Param().Alpha();
	if( t.NHits()<50 ) continue;
	nTrackHits+= t.NHits();
	fNTracks++;      
      }
    }
  }
  
  //* selection  

  {
    AliHLTTPCCAGBTrack *vtracks = new AliHLTTPCCAGBTrack [fNTracks];
    Int_t *vhits = new Int_t [fNHits];
    AliHLTTPCCAGBTrack **vptracks = new AliHLTTPCCAGBTrack* [fNTracks];

    for( Int_t itr=0; itr<fNTracks; itr++ ){
      vptracks[itr] = &(fTracks[itr]);
    }
    Int_t nTracks = 0;
    Int_t nHits = 0;
    sort(vptracks, vptracks+fNTracks, AliHLTTPCCAGBTrack::ComparePNClusters );
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
      if( to.NHits()<50 ) continue;
      nHits+=to.NHits();
      nTracks++;
    }
    fNTracks = nTracks;
    if( fTrackHits ) delete[] fTrackHits;
    if( fTracks ) delete[] fTracks;
    fTrackHits = vhits;
    fTracks = vtracks;
    delete[] vptracks;
  }

}

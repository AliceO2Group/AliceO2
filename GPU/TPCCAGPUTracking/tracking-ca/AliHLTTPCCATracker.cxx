// @(#) $Id$
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
//                                                                        *
// Primary Authors: Jochen Thaeder <thaeder@kip.uni-heidelberg.de>        *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>              *
//                  for The ALICE HLT Project.                            *
//                                                                        *
// Permission to use, copy, modify and distribute this software and its   *
// documentation strictly for non-commercial purposes is hereby granted   *
// without fee, provided that the above copyright notice appears in all   *
// copies and that both the copyright notice and this permission notice   *
// appear in the supporting documentation. The authors make no claims     *
// about the suitability of this software for any purpose. It is          *
// provided "as is" without express or implied warranty.                  *
//*************************************************************************

#include "AliHLTTPCCATracker.h"

#include "AliHLTTPCCAHit.h"
#include "AliHLTTPCCACell.h"
#include "AliHLTTPCCAOutTrack.h"

#include "TMath.h"
//#include "Riostream.h"
#include <vector>
#include <algorithm>
#include "TStopwatch.h"
//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#include "TApplication.h"
#endif //DRAW

ClassImp(AliHLTTPCCATracker)


AliHLTTPCCATracker::AliHLTTPCCATracker()
  :fParam(),fRows(0),fOutTrackHits(0),fNOutTrackHits(0),fOutTracks(0),fNOutTracks(0),fTrackCells(0),fNHitsTotal(0),fTracks(0),fNTracks(0),fCellHitPointers(0)
{
  // constructor
  fRows = new AliHLTTPCCARow[fParam.NRows()];
  Initialize( fParam );
}

AliHLTTPCCATracker::AliHLTTPCCATracker( const AliHLTTPCCATracker& )
  :fParam(),fRows(0),fOutTrackHits(0),fNOutTrackHits(0),fOutTracks(0),fNOutTracks(0),fTrackCells(0),fNHitsTotal(0),fTracks(0),fNTracks(0),fCellHitPointers(0)
{
  // dummy
}

AliHLTTPCCATracker &AliHLTTPCCATracker::operator=( const AliHLTTPCCATracker& )
{
  // dummy
  fRows=0;
  fOutTrackHits=0;
  fOutTracks=0;
  fNOutTracks=0;
  fTrackCells=0;
  return *this;
}

AliHLTTPCCATracker::~AliHLTTPCCATracker()
{
  // destructor
  StartEvent();
  delete[] fRows;
}

// ----------------------------------------------------------------------------------
void AliHLTTPCCATracker::Initialize( AliHLTTPCCAParam &param )
{
  // initialosation
  StartEvent();
  delete[] fRows;
  fRows = 0;
  fParam = param;
  fParam.Update();
  fRows = new AliHLTTPCCARow[fParam.NRows()];
  for( Int_t irow=0; irow<fParam.NRows(); irow++ ){
    fRows[irow].X() = fParam.RowX(irow);
  }
  StartEvent();
}

void AliHLTTPCCATracker::StartEvent()
{
  // start new event and fresh the memory  
  delete[] fTracks;
  delete[] fTrackCells;
  delete[] fOutTrackHits;
  delete[] fOutTracks;
  delete[] fCellHitPointers;
  fTracks = 0;
  fTrackCells = 0;
  fOutTrackHits = 0;
  fOutTracks = 0;
  fNTracks = 0;
  fNOutTrackHits = 0;
  fNOutTracks = 0;
  fNHitsTotal = 0;
  for( Int_t irow=0; irow<fParam.NRows(); irow++ ){
    fRows[irow].Clear();
  }
}


void AliHLTTPCCATracker::ReadHitRow( Int_t iRow, AliHLTTPCCAHit *Row, Int_t NHits )
{
  // read row of hits
  AliHLTTPCCARow &row = fRows[iRow];
  row.Hits() = new AliHLTTPCCAHit[NHits];
  for( Int_t i=0; i<NHits; i++ ){ 
    row.Hits()[i]=Row[i];
    row.Hits()[i].ErrY()*= fParam.YErrorCorrection();
    row.Hits()[i].ErrZ()*= fParam.ZErrorCorrection();
  }
  row.NHits() = NHits;
  fNHitsTotal += NHits;
}

void AliHLTTPCCATracker::Reconstruct()
{
  // reconstruction of event
  fTimers[0] = 0;
  fTimers[1] = 0;
  fTimers[2] = 0;
  fTimers[3] = 0;
  //cout<<"Find Cells..."<<endl;
  FindCells();
  //cout<<"Find Tracks..."<<endl;
   FindTracks();
   //cout<<"Find Tracks OK"<<endl;
 }


void AliHLTTPCCATracker::FindCells()
{
  // cell finder - neighbouring hits are grouped to cells

  TStopwatch timer;
  Bool_t *vUsed = new Bool_t [fNHitsTotal];
  fCellHitPointers = new Int_t [fNHitsTotal];
  AliHLTTPCCACell *vCells = new AliHLTTPCCACell[fNHitsTotal]; 

  Int_t lastCellHitPointer = 0;
  for( Int_t irow=0; irow<fParam.NRows(); irow++ ){
    AliHLTTPCCARow &row=fRows[irow];
    Int_t nHits = row.NHits();
    if( nHits<1 ) continue;
    row.CellHitPointers() = fCellHitPointers + lastCellHitPointer;
    Int_t nPointers = 0;
    Int_t nCells = 0;

    for (Int_t ih = 0; ih<nHits; ih++) vUsed[ih] = 0;
    
    for (Int_t ih = 0; ih<nHits; ih++){    
      if( vUsed[ih] ) continue;
      // cell start
      AliHLTTPCCACell &cell = vCells[nCells++];
      cell.FirstHitRef() = nPointers;
      cell.NHits() = 1;
      cell.IDown() = -1;
      cell.IUp() = -1;
      cell.IUsed() = 0;
      row.CellHitPointers()[nPointers++] = ih;
      vUsed[ih] = 1;
      Int_t jLast = ih;
      while( jLast<nHits-1 ){
	AliHLTTPCCAHit &h  = row.Hits()[jLast];    
	Double_t d2min = 1.e10;
	Int_t jBest = -1;
	for (Int_t j = jLast+1; j<nHits; j++){    
	  if( vUsed[j] ) continue;
	  AliHLTTPCCAHit &h1  = row.Hits()[j];
	  Double_t dy = TMath::Abs(h.Y() - h1.Y() );
	  if( dy>(h.ErrY()+h1.ErrY())*3.5*2 ) break;
	  Double_t dz = TMath::Abs(h.Z() - h1.Z() );
	  if( dz>(h.ErrZ()+h1.ErrZ())*3.5*2 ) continue;
	  Double_t d2 = dz*dz+dy*dy;
	  if( d2<d2min ){
	    d2min = d2;
	    jBest = j;
	  }
	}
	if( jBest<0 ) break;
	row.CellHitPointers()[nPointers++] = jBest;
	cell.NHits()++;
	vUsed[jBest] = 1;     
	jLast = jBest;
      }

      AliHLTTPCCAHit &h = row.GetCellHit(cell,0);
      AliHLTTPCCAHit &h1= row.GetCellHit(cell,cell.NHits()-1);
    
      cell.Y() = .5*(h.Y() + h1.Y());
      cell.Z() = .5*(h.Z() + h1.Z());
      cell.ErrY() = .5*(TMath::Abs(h.Y() - h1.Y())/3 + h.ErrY() + h1.ErrY());
      cell.ErrZ() = .5*(TMath::Abs(h.Z() - h1.Z())/3 + h.ErrZ() + h1.ErrZ());
    }
    
    row.Cells() = new AliHLTTPCCACell[nCells];
    row.NCells() = nCells;
    lastCellHitPointer += nPointers;
    for( Int_t i=0; i<nCells; i++ ) row.Cells()[i]=vCells[i];  
  }
  delete[] vUsed;
  delete[] vCells;
  timer.Stop();
  fTimers[0] = timer.CpuTime();
}


void AliHLTTPCCATracker::FindTracks()
{
  // the Cellular Automaton track finder
  
  if( fNHitsTotal < 1 ) return;

#ifdef DRAW
  if( !gApplication ){
    TApplication *myapp = new TApplication("myapp",0,0);
  }    
  //AliHLTTPCCADisplay::Instance().Init();
  AliHLTTPCCADisplay::Instance().SetCurrentSector( this );
  AliHLTTPCCADisplay::Instance().DrawSector( this );
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NHits(); i++) 
      AliHLTTPCCADisplay::Instance().DrawHit( iRow, i );
  cout<<"hits"<<endl;
  AliHLTTPCCADisplay::Instance().Ask();
  AliHLTTPCCADisplay::Instance().Clear();
  AliHLTTPCCADisplay::Instance().DrawSector( this );
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NCells(); i++) 
      AliHLTTPCCADisplay::Instance().DrawCell( iRow, i );
  cout<<"cells"<<endl;
  AliHLTTPCCADisplay::Instance().Ask();  
  Int_t nConnectedCells = 0;
#endif 

  std::vector<AliHLTTPCCATrack> vTracks; 
  std::vector<Int_t> vTrackCells; 
  fTrackCells = new Int_t[2*fNHitsTotal];

  TStopwatch timer1;
  Double_t factor2 = 3.5*fParam.CellConnectionFactor()*3.5*fParam.CellConnectionFactor();

  for( Int_t iRow1=0; iRow1<fParam.NRows()-1; iRow1++ ){
    AliHLTTPCCARow &row1 = fRows[iRow1];
    Int_t lastRow2 = iRow1+3;
    if( lastRow2>=fParam.NRows() ) lastRow2 = fParam.NRows()-1;    
    for( Int_t iRow2=iRow1+1; iRow2<=lastRow2; iRow2++ ){
      AliHLTTPCCARow &row2 = fRows[iRow2];
      for (Int_t i1 = 0; i1<row1.NCells(); i1++){
	AliHLTTPCCACell *c1  = &(row1.Cells()[i1]);
	if( c1->IUp()>=0 ) continue;
	Double_t sy1 = c1->ErrY()*c1->ErrY();
	Double_t sz1 = c1->ErrZ()*c1->ErrZ();
	for (Int_t i2 = 0; i2<row2.NCells(); i2++){
	  AliHLTTPCCACell *c2  = &(row2.Cells()[i2]);
	  Double_t sy2 = c2->ErrY()*c2->ErrY();
	  Double_t dy = c1->Y()-c2->Y();
	  if( dy*dy>factor2*(sy1+sy2) ){
	    if( dy>0 ) continue;
	    else break;
	  }
	  Double_t sz2 = c2->ErrZ()*c2->ErrZ();
	  Double_t dz = c1->Z()-c2->Z();
	  if( dz*dz>factor2*(sz1+sz2) ) continue;
	  if( c1->IUp() ==-1 ) c1->IUp() = (i2<<8)+iRow2;
	  else c1->IUp() = -2;
	  if( c2->IDown() ==-1 ) c2->IDown() = (i1<<8)+iRow1;
	  else c2->IDown() = -2;
	}
      }
    }
  }

  timer1.Stop();
  fTimers[1] = timer1.CpuTime();

  TStopwatch timer2;

  Int_t nOutTrackHits = 0;
  Int_t nTrackCells = 0;
  for( Int_t iRow1=0; iRow1<fParam.NRows(); iRow1++ ){
    AliHLTTPCCARow &row1 = fRows[iRow1];    
    for (Int_t i1 = 0; i1<row1.NCells(); i1++){ 
      AliHLTTPCCACell *c1  = &(row1.Cells()[i1]);
      //if( c1->IDown()==-2 || c1->IUp()==-2 ) continue;
      if( c1->IUsed()>0 ) continue;
      c1->IUsed() = 1;
      AliHLTTPCCATrack track;
      track.Used() = 0;
      track.NCells() = 1;
      track.IFirstCell() = nTrackCells;
      fTrackCells[nTrackCells++] = (i1<<8)+iRow1;
      AliHLTTPCCACell *last = c1;
      Int_t lastRow = iRow1;
      while( last->IUp() >=0 ){
	Int_t iRow2 = last->IUp()%256;
	AliHLTTPCCARow &row2 = fRows[iRow2];
	AliHLTTPCCACell *next = &(row2.Cells()[last->IUp()>>8]);
	if( next->IDown()==-2 || next->IUp()==-2 ) break;
#ifdef DRAW
 	AliHLTTPCCADisplay::Instance().ConnectCells( lastRow,*last,iRow2,*next );
	nConnectedCells++;
#endif 
	next->IUsed() = 1;      
	fTrackCells[nTrackCells++] = last->IUp();
	track.NCells()++;
	last = next;
	lastRow = iRow2;
      } 
      vTracks.push_back(track);
    }
  }

  timer2.Stop();
  fTimers[2] = timer2.CpuTime();
  
  Int_t nTracks = vTracks.size();
  std::sort( vTracks.begin(), vTracks.end(), AliHLTTPCCATrack::CompareSize);

#ifdef DRAW
  if( nConnectedCells>0 ) AliHLTTPCCADisplay::Instance().Ask();  
#endif 

  
  fTracks = new AliHLTTPCCATrack[nTracks];
  fNTracks = 0;
  vTrackCells.clear();
 
  Int_t *vMatchedTracks = new Int_t[nTracks];

  //cout<<"nTracks = "<<nTracks<<endl;
  TStopwatch timer3;
  
  for( Int_t itr=0; itr<nTracks; itr++ ){
    AliHLTTPCCATrack &iTrack = vTracks[itr];
    if( iTrack.NCells()<3 ) break;

#ifdef DRAW
    FitTrack( iTrack );
    //AliHLTTPCCADisplay::Instance().Ask();
    AliHLTTPCCADisplay::Instance().DrawTrack( iTrack );
    //AliHLTTPCCADisplay::Instance().Ask();
#endif 
    if( iTrack.Used() ) continue;    
    
   
    FitTrack( iTrack, 1 );
    if( iTrack.Param().Chi2() > fParam.TrackChi2Cut()*iTrack.Param().NDF() ) continue;

    Int_t iFirstRow =  GetTrackCellIRow( iTrack, 0);
    Int_t iLastRow  =  GetTrackCellIRow( iTrack, iTrack.NCells()-1);
    AliHLTTPCCACell *iFirstCell = &GetTrackCell( iTrack, 0);
    AliHLTTPCCACell *iLastCell = &GetTrackCell( iTrack, iTrack.NCells()-1);   
    Bool_t updated = 1;
    Int_t nMatched = 0;
    std::vector<Int_t> vMatchedCellsFront;
    std::vector<Int_t> vMatchedCellsBack;
    while( updated ){
      updated = 0;
      for( Int_t jtr=itr+1; jtr<nTracks; jtr++ ){
	AliHLTTPCCATrack &jTrack = vTracks[jtr];
	if( jTrack.Used() ) continue;
	Int_t jFirstRow =  GetTrackCellIRow( jTrack, 0);
	Int_t jLastRow =  GetTrackCellIRow( jTrack, jTrack.NCells()-1);
	AliHLTTPCCACell *jFirstCell = &GetTrackCell( jTrack, 0);
	AliHLTTPCCACell *jLastCell = &GetTrackCell( jTrack, jTrack.NCells()-1);

	Int_t dFirstRow1 = TMath::Abs(iFirstRow-jLastRow);
	Int_t dFirstRow2 = TMath::Abs(iFirstRow-jFirstRow);
	Int_t dLastRow1 = TMath::Abs(iLastRow-jLastRow);
	Int_t dLastRow2 = TMath::Abs(iLastRow-jFirstRow);
	if( dFirstRow1 > fParam.MaxTrackMatchDRow() && 
	    dFirstRow2 > fParam.MaxTrackMatchDRow() &&
	    dLastRow1  > fParam.MaxTrackMatchDRow() && 
	    dLastRow2  > fParam.MaxTrackMatchDRow()    ) continue;
	Int_t iCase=0;
	AliHLTTPCCACell *iC, *jC;
	if( dFirstRow1<dFirstRow2 && dFirstRow1<dLastRow1 && dFirstRow1<dLastRow2 ){
	  iCase = 0;
	  iC = iFirstCell;
	  jC = jLastCell;
	}else if( dFirstRow2<dLastRow1 && dFirstRow2<dLastRow2 ){
	  iCase = 1;
	  iC = iFirstCell;
	  jC = jFirstCell;
	}else if( dLastRow1<dLastRow2 ){
	  iCase = 2;
	  iC = iLastCell; 
	  jC = jLastCell;
	}else{
	  iCase = 3;
	  iC = iLastCell; 
	  jC = jFirstCell;
	}
	{
	  Double_t dy = TMath::Abs(iC->Y() - jC->Y());
	  Double_t dz = TMath::Abs(iC->Z() - jC->Z());
	  Double_t sy1 = iC->ErrY()*iC->ErrY();
	  Double_t sz1 = iC->ErrZ()*iC->ErrZ();
	  Double_t sy2 = jC->ErrY()*jC->ErrY();
	  Double_t sz2 = jC->ErrZ()*jC->ErrZ();
	  Double_t dist1 = sqrt( (dy*dy)/(sy1+sy2) );
	  Double_t dist2 = sqrt( (dz*dz)/(sz1+sz2) );
	  if( dist1>3.5*fParam.TrackConnectionFactor() ) continue;
	  if( dist2>3.5*fParam.TrackConnectionFactor() ) continue;
	}
	AliHLTTPCCATrackPar t = iTrack.Param();
	//t.Chi2() = 0;
	//t.NDF() = 0;
	for( Int_t i=0; i<jTrack.NCells() ; i++){
	  AliHLTTPCCACell &c = GetTrackCell(jTrack,i);	
	  AliHLTTPCCARow &row = GetTrackCellRow(jTrack,i);
	  for( Int_t j=0; j<c.NHits(); j++){
	    AliHLTTPCCAHit &h = row.GetCellHit(c,j);
	    Double_t m[3] = {row.X(), h.Y(), h.Z() };
	    Double_t mV[6] = {fParam.ErrX()*fParam.ErrX(), 0, h.ErrY()*h.ErrY(), 0, 0, h.ErrZ()*h.ErrZ() };
	    Double_t mV1[6];
	    t.TransportBz(fParam.Bz(),m);
	    t.GetConnectionMatrix(fParam.Bz(),m, mV1);
	    t.Filter(m, mV, mV1);
	  }
	}
	if( t.Chi2() > fParam.TrackChi2Cut()*t.NDF() ) continue;
	if( iCase==0 ){
	  iFirstRow = jFirstRow;
	  iFirstCell = jFirstCell;
	  for( Int_t i=jTrack.NCells()-1; i>=0; i-- ){
	    vMatchedCellsBack.push_back(fTrackCells[jTrack.IFirstCell()+i]);
	  }
	}else if( iCase ==1 ){
	  iFirstRow = jLastRow;
	  iFirstCell = jLastCell;
	  for( Int_t i=0; i<jTrack.NCells(); i++){
	    vMatchedCellsBack.push_back(fTrackCells[jTrack.IFirstCell()+i]);
	  }
	}else if( iCase == 2 ){
	  iLastRow = jFirstRow;
	  iLastCell = jFirstCell;
	  for( Int_t i=jTrack.NCells()-1; i>=0; i-- ){
	    vMatchedCellsFront.push_back(fTrackCells[jTrack.IFirstCell()+i]);
	  }
	}else{
	  iLastRow = jLastRow;
	  iLastCell = jLastCell;
	  for( Int_t i=0; i<jTrack.NCells(); i++){
	    vMatchedCellsFront.push_back(fTrackCells[jTrack.IFirstCell()+i]);
	  }
	}
	t.Normalize();

	//t.NDF()+= iTrack.Param().NDF();
	//t.Chi2()+= iTrack.Param().Chi2();
	iTrack.Param() = t;
	vMatchedTracks[nMatched++] = jtr;
	jTrack.Used()=1;
	updated = 1;
	break;
      }
    }// while updated

    if(0){
      Double_t t0[7];
      for( Int_t i=0; i<7; i++ ) t0[i] = iTrack.Param().Par()[i];
      iTrack.Param().Init();
      for( Int_t i=0; i<7; i++ ) iTrack.Param().Par()[i] = t0[i];
      for( Int_t i=0; i<iTrack.NCells() ; i++){
	AliHLTTPCCACell &c = GetTrackCell( iTrack, i);
	AliHLTTPCCARow &row = GetTrackCellRow( iTrack, i);
	for( Int_t j=0; j<c.NHits(); j++){
	  AliHLTTPCCAHit &h = row.GetCellHit(c,j);
	  Double_t m[3] = {row.X(), h.Y(), h.Z() };
	  Double_t mV[6] = {0, 0, h.ErrY()*h.ErrY(), 0, 0, h.ErrZ()*h.ErrZ() };
	  Double_t mV1[6];
	  iTrack.Param().TransportBz(fParam.Bz(), m, t0);
	  iTrack.Param().GetConnectionMatrix(fParam.Bz(), m, mV1, t0);
	  iTrack.Param().Filter(m, mV, mV1);
	}
      }
      iTrack.Param().Normalize();
    }
    //FitTrack(iTrack,5);
    Int_t nHits = 0;
    for( Int_t iCell=0; iCell<iTrack.NCells(); iCell++){
      Int_t ind = fTrackCells[iTrack.IFirstCell()+iCell];
      AliHLTTPCCARow &row = fRows[ind%256];
      AliHLTTPCCACell &c = row.Cells()[ind>>8];
      nHits+=c.NHits();
    }
    for( UInt_t i=0; i<vMatchedCellsBack.size(); i++){
      Int_t ind = vMatchedCellsBack[i];
      AliHLTTPCCARow &row = fRows[ind%256];
      AliHLTTPCCACell &c = row.Cells()[ind>>8];
      nHits+=c.NHits();
    }
    for( UInt_t i=0; i<vMatchedCellsFront.size(); i++){
      Int_t ind = vMatchedCellsFront[i];
      AliHLTTPCCARow &row = fRows[ind%256];
      AliHLTTPCCACell &c = row.Cells()[ind>>8];
      nHits+=c.NHits();
    }
    Int_t nCells = iTrack.NCells()+vMatchedCellsBack.size()+vMatchedCellsFront.size();
    if( nHits<5 || nCells<3){
      for( Int_t i=0; i<nMatched; i++ ) vTracks[vMatchedTracks[i]].Used()=0;
      continue;
    }
    iTrack.Used() = 1;

    AliHLTTPCCATrack &mTrack = fTracks[fNTracks++];
    mTrack = iTrack;
    mTrack.IFirstCell() = vTrackCells.size();
    mTrack.NCells() = 0;

    for( Int_t i=vMatchedCellsBack.size()-1; i>=0; i-- ){
      vTrackCells.push_back(vMatchedCellsBack[i]);
      mTrack.NCells()++;
    }
    for( Int_t i=0; i<iTrack.NCells() ; i++){
      vTrackCells.push_back(fTrackCells[iTrack.IFirstCell()+i]);
      mTrack.NCells()++;
    }
    for( UInt_t i=0; i<vMatchedCellsFront.size(); i++){
      vTrackCells.push_back(vMatchedCellsFront[i]);
      mTrack.NCells()++;
    }     
    nOutTrackHits+= nHits;  
  }

  timer3.Stop();
  fTimers[3] = timer3.CpuTime();

  delete[] vMatchedTracks;

  //fTrackCells = new Int_t[vTrackCells.size()];
  for( UInt_t i=0; i<vTrackCells.size(); i++ ) fTrackCells[i] = vTrackCells[i];
  
#ifdef DRAW
  if( nTracks>0 ) AliHLTTPCCADisplay::Instance().Ask();
#endif 

  //cout<<"n out Tracks = "<<fNTracks<<endl;

  fOutTrackHits = new Int_t[nOutTrackHits];
  fNOutTrackHits = 0;
  fNOutTracks = fNTracks;
  fOutTracks = new AliHLTTPCCAOutTrack[fNOutTracks];
  for( Int_t itr=0; itr<fNOutTracks; itr++ ){
    AliHLTTPCCATrack &t = fTracks[itr];
#ifdef DRAW
    AliHLTTPCCADisplay::Instance().DrawTrack( t );
    //AliHLTTPCCADisplay::Instance().Ask();
#endif 
    AliHLTTPCCAOutTrack &tmp = fOutTracks[itr];
    tmp.FirstHitRef() = fNOutTrackHits;
    tmp.NHits() = 0;
    tmp.Param() = t.Param();
    for( Int_t iCell=0; iCell<t.NCells(); iCell++){
      AliHLTTPCCACell &cell = GetTrackCell(t,iCell);
      AliHLTTPCCARow &row = GetTrackCellRow(t,iCell);
      for( Int_t iHit=0; iHit<cell.NHits(); iHit++ ){
	AliHLTTPCCAHit &hit = row.GetCellHit(cell,iHit);
	fOutTrackHits[fNOutTrackHits] = hit.ID();
	fNOutTrackHits++;
	tmp.NHits()++;
      }
    }
  }  
  
#ifdef DRAW
  AliHLTTPCCADisplay::Instance().Ask();
  //AliHLTTPCCADisplay::Instance().DrawMCTracks(fParam.fISec);
  //AliHLTTPCCADisplay::Instance().Update();
  //AliHLTTPCCADisplay::Instance().Ask();
#endif 
}



void AliHLTTPCCATracker::FitTrack( AliHLTTPCCATrack &track, Int_t nIter )
{    
  // fit the track with nIter iterations

  AliHLTTPCCATrackPar &t = track.Param();
  t.Init();
  
  AliHLTTPCCACell &c1 = GetTrackCell(track,0);
  AliHLTTPCCACell &c2 = GetTrackCell(track,track.NCells()-1);
  AliHLTTPCCARow &row1 = GetTrackCellRow(track,0);
  AliHLTTPCCARow &row2 = GetTrackCellRow(track,track.NCells()-1);
  Double_t t0[7];
  t0[0]=row1.X();
  t0[1]=c1.Y();
  t0[2]=c1.Z();
  t0[3]= row2.X() - row1.X();
  t0[4]= c2.Y() - c1.Y();
  t0[5]= c2.Z() - c1.Z();
  Double_t tt = sqrt(t0[3]*t0[3]+t0[4]*t0[4]+t0[5]*t0[5]);
  if( TMath::Abs(tt)>1.e-4 ){
    t0[3]/=tt;
    t0[4]/=tt;
    t0[5]/=tt;
  }else{
    t0[4]=1;
  }
  t0[6] = 0;

  Int_t step = track.NCells()/3;

  for( Int_t iter=0; iter<nIter; iter++ ){
    t.Init();
    for( Int_t i=0; i<7; i++) t.Par()[i] = t0[i];
    {
      Double_t m[3] = {row1.X(), c1.Y(), c1.Z() };
      t.TransportBz(fParam.Bz(),m,t0);
    }
    t.Init();
    for( Int_t i=0; i<7; i++ ) t.Par()[i] = t0[i];

    //AliHLTTPCCATrackPar tt = t;
  
    for( Int_t i=0; i<track.NCells() ; i+=step){
      AliHLTTPCCACell &c = GetTrackCell(track,i);
      AliHLTTPCCARow &row = GetTrackCellRow(track,i);
      for( Int_t j=0; j<c.NHits(); j++){
	AliHLTTPCCAHit &h = row.GetCellHit(c,j);
	Double_t m[3] = {row.X(), h.Y(), h.Z() };
	Double_t mV[6] = {fParam.ErrX()*fParam.ErrX(), 0, h.ErrY()*h.ErrY(), 0, 0, h.ErrZ()*h.ErrZ() };
	Double_t mV1[6];
	t.TransportBz(fParam.Bz(),m, t0);
	t.GetConnectionMatrix(fParam.Bz(),m, mV1, t0);
	t.Filter(m, mV, mV1);
	//tt.TransportBz(fParam.Bz(),m, t0);
	//tt.GetConnectionMatrix(fParam.Bz(),m, mV1, t0);
	//tt.Filter(m, mV, mV1);
      }
    }
    t.Normalize();
    for( Int_t i=0; i<7; i++ ) t0[i] = t.Par()[i];
  }
}

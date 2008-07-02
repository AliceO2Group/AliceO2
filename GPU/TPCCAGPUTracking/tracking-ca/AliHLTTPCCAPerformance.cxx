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

#include "AliHLTTPCCAPerformance.h"
#include "AliHLTTPCCAGBHit.h"
#include "AliHLTTPCCAMCTrack.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCATracker.h"

#include "TMath.h"
#include "TROOT.h"
#include "Riostream.h"
#include "TFile.h"
#include "TH1.h"
#include "TProfile.h"


ClassImp(AliHLTTPCCAPerformance)



AliHLTTPCCAPerformance::AliHLTTPCCAPerformance()
  : TObject(),
    fTracker(0),
    fHitLabels(0), 
    fNHits(0),
    fMCTracks(0),
    fNMCTracks(0),
    fStatNEvents(0),
    fStatNRecTot(0),
    fStatNRecOut(0),
    fStatNGhost(0),
    fStatNMCAll(0),
    fStatNRecAll(0),
    fStatNClonesAll(0),
    fStatNMCRef(0),
    fStatNRecRef(0),
    fStatNClonesRef(0),
    fHistoDir(0),
    fhHitErrY(0),
    fhHitErrZ(0),
    fhHitResX(0),
    fhHitResY(0),
    fhHitResZ(0),
    fhHitPullX(0),
    fhHitPullY(0),
    fhHitPullZ(0),
    fhCellPurity(0),
    fhCellNHits(0),
    fhCellPurityVsN(0), 
    fhCellPurityVsPt(0),
    fhEffVsP(0)
{
  //* constructor
}


AliHLTTPCCAPerformance::AliHLTTPCCAPerformance(const AliHLTTPCCAPerformance&)
  : TObject(),
    fTracker(0),
    fHitLabels(0), 
    fNHits(0),
    fMCTracks(0),
    fNMCTracks(0),
    fStatNEvents(0),
    fStatNRecTot(0),
    fStatNRecOut(0),
    fStatNGhost(0),
    fStatNMCAll(0),
    fStatNRecAll(0),
    fStatNClonesAll(0),
    fStatNMCRef(0),
    fStatNRecRef(0),
    fStatNClonesRef(0),
    fHistoDir(0),
    fhHitErrY(0),
    fhHitErrZ(0),
    fhHitResX(0),
    fhHitResY(0),
    fhHitResZ(0),
    fhHitPullX(0),
    fhHitPullY(0),
    fhHitPullZ(0),
    fhCellPurity(0),
    fhCellNHits(0),
    fhCellPurityVsN(0), 
    fhCellPurityVsPt(0),
    fhEffVsP(0)
{
  //* dummy
}

AliHLTTPCCAPerformance &AliHLTTPCCAPerformance::operator=(const AliHLTTPCCAPerformance&)
{
  //* dummy
  return *this;
}

AliHLTTPCCAPerformance::~AliHLTTPCCAPerformance()
{
  //* destructor
  StartEvent();
}

void AliHLTTPCCAPerformance::SetTracker( AliHLTTPCCAGBTracker *Tracker )
{
  //* set pointer to HLT CA Global tracker
  fTracker = Tracker;
}

void AliHLTTPCCAPerformance::StartEvent()
{
  //* clean up arrays
  if( !fHistoDir )  CreateHistos();
  if( fHitLabels ) delete[] fHitLabels;
  fHitLabels = 0;
  fNHits = 0;
  if( fMCTracks ) delete[] fMCTracks;
  fMCTracks = 0;
  fNMCTracks = 0;
}

void AliHLTTPCCAPerformance::SetNHits( Int_t NHits )
{
  //* set number of hits
  if( fHitLabels ) delete[] fHitLabels;
  fHitLabels = 0;
  fHitLabels = new AliHLTTPCCAHitLabel[ NHits ];
  fNHits = NHits;
}  

void AliHLTTPCCAPerformance::SetNMCTracks( Int_t NMCTracks )
{
  //* set number of MC tracks
  if( fMCTracks ) delete[] fMCTracks;
  fMCTracks = 0;
  fMCTracks = new AliHLTTPCCAMCTrack[ NMCTracks ];
  fNMCTracks = NMCTracks;
}  

void AliHLTTPCCAPerformance::ReadHitLabel( Int_t HitID, 
					   Int_t lab0, Int_t lab1, Int_t lab2 )
{
  //* read the hit labels
  AliHLTTPCCAHitLabel hit;
  hit.fLab[0] = lab0;
  hit.fLab[1] = lab1;
  hit.fLab[2] = lab2;
  fHitLabels[HitID] = hit;
}

void AliHLTTPCCAPerformance::ReadMCTrack( Int_t index, const TParticle *part )
{
  //* read mc track to local array
  fMCTracks[index] = AliHLTTPCCAMCTrack(part);
}

void AliHLTTPCCAPerformance::CreateHistos()
{
  //* create performance histogramms
  TDirectory *curdir = gDirectory;
  fHistoDir = gROOT->mkdir("HLTTPCCATrackerPerformance");
  fHistoDir->cd();
  gDirectory->mkdir("Fit");
  gDirectory->cd("Fit");  
  /*
  fhResY = new TH1D("resY", "Y resoltion [cm]", 100, -5., 5.);
  fhResZ = new TH1D("resZ", "Z resoltion [cm]", 100, -5., 5.);
  fhResTy = new TH1D("resTy", "Ty resoltion ", 100, -1., 1.);
  fhResTz = new TH1D("resTz", "Tz resoltion ", 100, -1., 1.);
  fhResP = new TH1D("resP", "P resoltion ", 100, -10., 10.);
  fhPullY = new TH1D("pullY", "Y pull", 100, -10., 10.);
  fhPullZ = new TH1D("pullZ", "Z pull", 100, -10., 10.);
  fhPullTy = new TH1D("pullTy", "Ty pull", 100, -10., 10.);
  fhPullTz = new TH1D("pullTz", "Tz pull", 100, -10., 10.);
  fhPullQp = new TH1D("pullQp", "Qp pull", 100, -10., 10.);
  */
  gDirectory->cd("..");  

  fhEffVsP = new TProfile("EffVsP", "Eff vs P", 100, 0., 5.);

  fhHitResX = new TH1D("resHitX", "X cluster resoltion [cm]", 100, -2., 2.);
  fhHitResY = new TH1D("resHitY", "Y cluster resoltion [cm]", 100, -2., 2.);
  fhHitResZ = new TH1D("resHitZ", "Z cluster resoltion [cm]", 100, -2., 2.);
  fhHitPullX = new TH1D("pullHitX", "X cluster pull", 100, -10., 10.);
  fhHitPullY = new TH1D("pullHitY", "Y cluster pull", 100, -10., 10.);
  fhHitPullZ = new TH1D("pullHitZ", "Z cluster pull", 100, -10., 10.);


  fhHitErrY = new TH1D("HitErrY", "Y cluster error [cm]", 100, 0., 1.);
  fhHitErrZ = new TH1D("HitErrZ", "Z cluster error [cm]", 100, 0., 1.);

  fhCellPurity = new TH1D("CellPurity", "Cell Purity", 100, -0.1, 1.1);
  fhCellNHits = new TH1D("CellNHits", "Cell NHits", 40, 0., 40.);
  fhCellPurityVsN = new TProfile("CellPurityVsN", "Cell purity Vs N hits", 40, 2., 42.);
  fhCellPurityVsPt = new TProfile("CellPurityVsPt", "Cell purity Vs Pt", 100, 0., 5.);

  curdir->cd();  
}

void AliHLTTPCCAPerformance::WriteDir2Current( TObject *obj )
{
  //* recursive function to copy the directory 'obj' to the current one
  if( !obj->IsFolder() ) obj->Write();
  else{
    TDirectory *cur = gDirectory;
    TDirectory *sub = cur->mkdir(obj->GetName());
    sub->cd();
    TList *listSub = ((TDirectory*)obj)->GetList();
    TIter it(listSub);
    while( TObject *obj1=it() ) WriteDir2Current(obj1);
    cur->cd();
  }
}

void AliHLTTPCCAPerformance::WriteHistos()
{
  //* write histograms to the file
  TDirectory *curr = gDirectory;
  // Open output file and write histograms
  TFile* outfile = new TFile("HLTTPCCATrackerPerformance.root","RECREATE");
  outfile->cd();
  WriteDir2Current(fHistoDir);
  outfile->Close();
  curr->cd();
}


void AliHLTTPCCAPerformance::SlicePerformance( Int_t iSlice, Bool_t PrintFlag )
{ 
  //* calculate slice tracker performance
  if( !fTracker ) return;

  int nRecTot = 0, nGhost=0, nRecOut=0;
  int nMCAll = 0, nRecAll=0, nClonesAll=0;
  int nMCRef = 0, nRecRef=0, nClonesRef=0;
  AliHLTTPCCATracker &slice = fTracker->Slices()[iSlice];

  int firstSliceHit = 0;
  for( ; firstSliceHit<fTracker->NHits(); firstSliceHit++){
    if( fTracker->Hits()[firstSliceHit].ISlice()==iSlice ) break;
  }
  int endSliceHit = firstSliceHit;

  for( ; endSliceHit<fTracker->NHits(); endSliceHit++){
    if( fTracker->Hits()[endSliceHit].ISlice()!=iSlice ) break;
  }

  { // Cell construction performance
    
    for( Int_t iRow=0; iRow<slice.Param().NRows(); iRow++ ){
      AliHLTTPCCARow &row = slice.Rows()[iRow];      
      for (Int_t ic = 0; ic<row.NCells(); ic++){
	AliHLTTPCCACell &c  = row.Cells()[ic];
	Int_t *lb = new Int_t[c.NHits()*3];
	Int_t nla = 0;
	//cout<<11<<" "<<c.NHits()<<endl;
	for( Int_t j=0; j<c.NHits(); j++){
	  AliHLTTPCCAHit &h = row.GetCellHit(c,j);
	  //cout<<"hit ID="<<h.ID()<<" of"<<fTracker->NHits()<<endl;
	  //cout<<"gb hit ID="<<fTracker->Hits()[h.ID()].ID()<<endl;
	  AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[h.ID()].ID()];
	  if( l.fLab[0]>=0 ) lb[nla++]= l.fLab[0];
	  if( l.fLab[1]>=0 ) lb[nla++]= l.fLab[1];
	  if( l.fLab[2]>=0 ) lb[nla++]= l.fLab[2];
	}
	//cout<<12<<endl;
	sort( lb, lb+nla );
	int labmax = -1, labcur=-1, lmax = 0, lcurr=0;	
	for( int i=0; i<nla; i++ ){
	  if( lb[i]!=labcur ){
	    if( labcur>=0 && lmax<lcurr ){
	      lmax = lcurr;
	      labmax = labcur;
	    }
	    labcur = lb[i];
	    lcurr = 0;
	  }
	  lcurr++;
	}
	if( labcur>=0 && lmax<lcurr ){
	  lmax = lcurr;
	  labmax = labcur;
	}

	int label = labmax;
	lmax = 0;
	for( Int_t j=0; j<c.NHits(); j++){
	  AliHLTTPCCAHit &h = row.GetCellHit(c,j);
	  AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[h.ID()].ID()];
	  if( l.fLab[0]==label || l.fLab[1]==label || l.fLab[2]==label ) lmax++;
	}

	nla = c.NHits();
	if( nla>0 && label>=0 ){
	  double purity = double(lmax)/double(nla);
	  fhCellPurity->Fill(purity);
	  fhCellPurityVsN->Fill(c.NHits(),purity);
	  fhCellPurityVsPt->Fill(fMCTracks[label].Pt(),purity);
	}
	fhCellNHits->Fill(c.NHits());
	if(lb) delete[] lb;
      }
    }
  }

  // Select reconstructable MC tracks

  {
    for (Int_t imc=0; imc<fNMCTracks; imc++) fMCTracks[imc].NHits() = 0;
          
    for( int ih=firstSliceHit; ih<endSliceHit; ih++){
      AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[ih].ID()];
      if( l.fLab[0]>=0 ) fMCTracks[l.fLab[0]].NHits()++;
      if( l.fLab[1]>=0 ) fMCTracks[l.fLab[1]].NHits()++;
      if( l.fLab[2]>=0 ) fMCTracks[l.fLab[2]].NHits()++;
    }
    
    for (Int_t imc=0; imc<fNMCTracks; imc++) {		
      AliHLTTPCCAMCTrack &mc = fMCTracks[imc];
      mc.Set() = 0;
      mc.NReconstructed() = 0;
      mc.NTurns() = 1;
      if( mc.NHits() >=  10 && mc.P()>=.05 ){
	mc.Set() = 1;
	nMCAll++;
	if( mc.P()>=1. ){
	  mc.Set() = 2;
	  nMCRef++;
	}
      }
    }
  }

  int traN = slice.NOutTracks();
  Int_t *traLabels = new Int_t[traN];
  Double_t *traPurity = new Double_t[traN];
  {
    for (Int_t itr=0; itr<traN; itr++) {
      traLabels[itr]=-1;
      traPurity[itr]= 0;
      AliHLTTPCCAOutTrack &tCA = slice.OutTracks()[itr];
      int nhits = tCA.NHits();
      int *lb = new Int_t[nhits*3];
      int nla=0;
      for( int ihit=0; ihit<nhits; ihit++){
	int index = slice.OutTrackHits()[tCA.FirstHitRef()+ihit];
	AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[index].ID()];
	if(l.fLab[0]>=0 ) lb[nla++]= l.fLab[0];
	if(l.fLab[1]>=0 ) lb[nla++]= l.fLab[1];
	if(l.fLab[2]>=0 ) lb[nla++]= l.fLab[2];
      }
      sort( lb, lb+nla );
      int labmax = -1, labcur=-1, lmax = 0, lcurr=0;
      for( int i=0; i<nla; i++ ){
	if( lb[i]!=labcur ){
	  if( labcur>=0 && lmax<lcurr ){
	    lmax = lcurr;
	    labmax = labcur;
	  }
	  labcur = lb[i];
	  lcurr = 0;
	}
	lcurr++;
      }
      if( labcur>=0 && lmax<lcurr ){
	lmax = lcurr;
	labmax = labcur;
      }
      lmax = 0;
      for( int ihit=0; ihit<nhits; ihit++){
	int index = slice.OutTrackHits()[tCA.FirstHitRef()+ihit];
	AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[index].ID()];
	if( l.fLab[0] == labmax || l.fLab[1] == labmax || l.fLab[2] == labmax 
	    ) lmax++;
      }
      traLabels[itr] = labmax;
      traPurity[itr] = ( (nhits>0) ?double(lmax)/double(nhits) :0 );
      if( lb ) delete[] lb;
    }
  }

  nRecTot+= traN;
  for(int itr=0; itr<traN; itr++){      
    if( traPurity[itr]<.7 || traLabels[itr]<0 || traLabels[itr]>=fNMCTracks){
      nGhost++;
      continue;
    }

    AliHLTTPCCAMCTrack &mc = fMCTracks[traLabels[itr]];	
    mc.NReconstructed()++;
    if( mc.Set()== 0 ) nRecOut++;
    else{
      if( mc.NReconstructed()==1 ) nRecAll++;
      else if(mc.NReconstructed() > mc.NTurns() ) nClonesAll++;
      if( mc.Set()==2 ){
	if( mc.NReconstructed()==1 ) nRecRef++;
	else if(mc.NReconstructed() > mc.NTurns() ) nClonesRef++;
      }
    }      
  }

  for (Int_t ipart=0; ipart<fNMCTracks; ipart++) {		
    AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
    if( mc.Set()>0 ) fhEffVsP->Fill(mc.P(), ( mc.NReconstructed()>0 ?1 :0));
  }  


  if( traLabels ) delete[] traLabels;
  if( traPurity ) delete[] traPurity;

  fStatNRecTot += nRecTot;
  fStatNRecOut += nRecOut;
  fStatNGhost  += nGhost;
  fStatNMCAll  += nMCAll;
  fStatNRecAll  += nRecAll;
  fStatNClonesAll  += nClonesAll;
  fStatNMCRef  += nMCRef;
  fStatNRecRef  += nRecRef;
  fStatNClonesRef  += nClonesRef;

  if( nMCAll ==0 ) return;

  if( PrintFlag ){
    cout<<"Performance for slice "<<iSlice<<" : "<<endl;
    cout<<" N tracks : "
	<<nMCAll<<" mc all, "
	<<nMCRef<<" mc ref, "
	<<nRecTot<<" rec total, "
	<<nRecAll<<" rec all, "
	<<nClonesAll<<" clones all, "
	<<nRecRef<<" rec ref, "
	<<nClonesRef<<" clones ref, "
	<<nRecOut<<" out, "
	<<nGhost<<" ghost"<<endl;
  
    Int_t nRecExtr = nRecAll - nRecRef;
    Int_t nMCExtr = nMCAll - nMCRef;
    Int_t nClonesExtr = nClonesAll - nClonesRef;
  
    Double_t dRecTot = (nRecTot>0 ) ? nRecTot :1;
    Double_t dMCAll = (nMCAll>0 ) ? nMCAll :1;
    Double_t dMCRef = (nMCRef>0 ) ? nMCRef :1;
    Double_t dMCExtr = (nMCExtr>0 ) ? nMCExtr :1;
    Double_t dRecAll = (nRecAll+nClonesAll>0 ) ? nRecAll+nClonesAll :1;
    Double_t dRecRef = (nRecRef+nClonesRef>0 ) ? nRecRef+nClonesRef :1;
    Double_t dRecExtr = (nRecExtr+nClonesExtr>0 ) ? nRecExtr+nClonesExtr :1;
    
    cout<<" EffRef = ";
    if( nMCRef>0 ) cout<<nRecRef/dMCRef; else cout<<"_";
    cout<<", CloneRef = ";
    if( nRecRef >0 ) cout << nClonesRef/dRecRef; else cout<<"_";
    cout<<endl;
    cout<<" EffExtra = ";
    if( nMCExtr>0 ) cout << nRecExtr/dMCExtr; else cout<<"_";
    cout <<", CloneExtra = ";
    if( nRecExtr>0 ) cout << nClonesExtr/dRecExtr; else cout<<"_";
    cout<<endl;
    cout<<" EffAll = ";
    if( nMCAll>0 ) cout<<nRecAll/dMCAll; else cout<<"_";
    cout <<", CloneAll = ";
    if( nRecAll>0 ) cout << nClonesAll/dRecAll; else cout<<"_";
    cout <<endl;
    cout<<" Out = ";
    if( nRecTot>0 ) cout <<nRecOut/dRecTot; else cout<<"_";
    cout <<", Ghost = ";
    if( nRecTot>0 ) cout<<nGhost/dRecTot; else cout<<"_";
    cout<<endl;
  }
}


void AliHLTTPCCAPerformance::Performance()
{ 
  // main routine for performance calculation  
  fStatNEvents++;
  for( int islice=0; islice<fTracker->NSlices(); islice++){
    SlicePerformance(islice,0);
  }
  
  // distribution of cluster errors

  {    
    Int_t nHits = fTracker->NHits();
    for( Int_t ih=0; ih<nHits; ih++ ){
      AliHLTTPCCAGBHit &hit = fTracker->Hits()[ih];
      fhHitErrY->Fill(hit.ErrY());
      fhHitErrZ->Fill(hit.ErrZ());
    }
  }
  
  cout<<" N tracks : "
      <<fStatNMCAll/fStatNEvents<<" mc all, "
      <<fStatNMCRef/fStatNEvents<<" mc ref, "
      <<fStatNRecTot/fStatNEvents<<" rec total, "
      <<fStatNRecAll/fStatNEvents<<" rec all, "
      <<fStatNClonesAll/fStatNEvents<<" clones all, "
      <<fStatNRecRef/fStatNEvents<<" rec ref, "
      <<fStatNClonesRef/fStatNEvents<<" clones ref, "
      <<fStatNRecOut/fStatNEvents<<" out, "
      <<fStatNGhost/fStatNEvents<<" ghost"<<endl;
  
  Int_t nRecExtr = fStatNRecAll - fStatNRecRef;
  Int_t nMCExtr = fStatNMCAll - fStatNMCRef;
  Int_t nClonesExtr = fStatNClonesAll - fStatNClonesRef;
  
  Double_t dRecTot = (fStatNRecTot>0 ) ? fStatNRecTot :1;
  Double_t dMCAll = (fStatNMCAll>0 ) ? fStatNMCAll :1;
  Double_t dMCRef = (fStatNMCRef>0 ) ? fStatNMCRef :1;
  Double_t dMCExtr = (nMCExtr>0 ) ? nMCExtr :1;
  Double_t dRecAll = (fStatNRecAll+fStatNClonesAll>0 ) ? fStatNRecAll+fStatNClonesAll :1;
  Double_t dRecRef = (fStatNRecRef+fStatNClonesRef>0 ) ? fStatNRecRef+fStatNClonesRef :1;
  Double_t dRecExtr = (nRecExtr+nClonesExtr>0 ) ? nRecExtr+nClonesExtr :1;

  cout<<" EffRef = "<< fStatNRecRef/dMCRef
      <<", CloneRef = " << fStatNClonesRef/dRecRef <<endl;
  cout<<" EffExtra = "<<nRecExtr/dMCExtr
      <<", CloneExtra = " << nClonesExtr/dRecExtr<<endl;
  cout<<" EffAll = "<<fStatNRecAll/dMCAll
      <<", CloneAll = " << fStatNClonesAll/dRecAll<<endl;
  cout<<" Out = "<<fStatNRecOut/dRecTot
      <<", Ghost = "<<fStatNGhost/dRecTot<<endl;
  cout<<" Time = "<<fTracker->StatTime(0)/fTracker->StatNEvents()*1.e3<<" msec/event "<<endl;
  cout<<" Local timers = "
      <<fTracker->StatTime(1)/fTracker->StatNEvents()*1.e3<<" "
      <<fTracker->StatTime(2)/fTracker->StatNEvents()*1.e3<<" "
      <<fTracker->StatTime(3)/fTracker->StatNEvents()*1.e3<<" "
      <<fTracker->StatTime(4)/fTracker->StatNEvents()*1.e3<<" "
      <<fTracker->StatTime(5)/fTracker->StatNEvents()*1.e3<<"["
      <<fTracker->StatTime(6)/fTracker->StatNEvents()*1.e3<<"/"
      <<fTracker->StatTime(7)/fTracker->StatNEvents()*1.e3<<"], merge:"
      <<fTracker->StatTime(8)/fTracker->StatNEvents()*1.e3
      <<" msec/event "<<endl;
  WriteHistos();
}

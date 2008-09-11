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
#include "AliHLTTPCCAMCPoint.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAGBTrack.h"
#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCATracker.h"


#include "TMath.h"
#include "TROOT.h"
#include "Riostream.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"


ClassImp(AliHLTTPCCAPerformance)



AliHLTTPCCAPerformance::AliHLTTPCCAPerformance()
  : TObject(),
    fTracker(0),
    fHitLabels(0), 
    fNHits(0),
    fMCTracks(0),
    fNMCTracks(0),
    fMCPoints(0),
    fNMCPoints(0),
    fDoClusterPulls(1),
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
    fStatGBNRecTot(0),
    fStatGBNRecOut(0),
    fStatGBNGhost(0),
    fStatGBNMCAll(0),
    fStatGBNRecAll(0),
    fStatGBNClonesAll(0),
    fStatGBNMCRef(0),
    fStatGBNRecRef(0),
    fStatGBNClonesRef(0),
    fHistoDir(0),
    fhResY(0),
    fhResZ(0),
    fhResSinPhi(0),
    fhResDzDs(0),
    fhResPt(0),
    fhPullY(0),
    fhPullZ(0),
    fhPullSinPhi(0),
    fhPullDzDs(0),
    fhPullQPt(0),
    fhHitErrY(0),
    fhHitErrZ(0),
    fhHitResY(0),
    fhHitResZ(0),
    fhHitPullY(0),
    fhHitPullZ(0),
    fhHitResY1(0),
    fhHitResZ1(0),
    fhHitPullY1(0),
    fhHitPullZ1(0),
    fhCellPurity(0),
    fhCellNHits(0),
    fhCellPurityVsN(0), 
    fhCellPurityVsPt(0),
    fhEffVsP(0),
    fhGBEffVsP(0)
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
    fMCPoints(0),
    fNMCPoints(0),
    fDoClusterPulls(1),
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
    fStatGBNRecTot(0),
    fStatGBNRecOut(0),
    fStatGBNGhost(0),
    fStatGBNMCAll(0),
    fStatGBNRecAll(0),
    fStatGBNClonesAll(0),
    fStatGBNMCRef(0),
    fStatGBNRecRef(0),
    fStatGBNClonesRef(0),
    fHistoDir(0),
    fhResY(0),
    fhResZ(0),
    fhResSinPhi(0),
    fhResDzDs(0),
    fhResPt(0),
    fhPullY(0),
    fhPullZ(0),
    fhPullSinPhi(0),
    fhPullDzDs(0),
    fhPullQPt(0),
    fhHitErrY(0),
    fhHitErrZ(0),    
    fhHitResY(0),
    fhHitResZ(0),
    fhHitPullY(0),
    fhHitPullZ(0),
    fhHitResY1(0),
    fhHitResZ1(0),
    fhHitPullY1(0),
    fhHitPullZ1(0),
    fhCellPurity(0),
    fhCellNHits(0),
    fhCellPurityVsN(0), 
    fhCellPurityVsPt(0),
    fhEffVsP(0),
    fhGBEffVsP(0)
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
  if( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fNMCPoints = 0;
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

void AliHLTTPCCAPerformance::SetNMCPoints( Int_t NMCPoints )
{
  //* set number of MC points
  if( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fMCPoints = new AliHLTTPCCAMCPoint[ NMCPoints ];
  fNMCPoints = 0;
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
  //* read mc track to the local array
  fMCTracks[index] = AliHLTTPCCAMCTrack(part);
}

void AliHLTTPCCAPerformance::ReadMCTPCTrack( Int_t index, Float_t X, Float_t Y, Float_t Z, 
					     Float_t Px, Float_t Py, Float_t Pz )
{
  //* read mc track parameters at TPC
  fMCTracks[index].SetTPCPar(X,Y,Z,Px,Py,Pz);
}

void AliHLTTPCCAPerformance::ReadMCPoint( Int_t TrackID, Float_t X, Float_t Y, Float_t Z, Float_t Time, Int_t iSlice )
{
  //* read mc point to the local array
  AliHLTTPCCAMCPoint &p = fMCPoints[fNMCPoints];
  p.TrackID() = TrackID;
  p.X() = X;
  p.Y() = Y;
  p.Z() = Z;
  p.Time() = Time;
  p.ISlice() = iSlice;
  fTracker->Slices()[iSlice].Param().Global2Slice( X, Y, Z, 
						   &p.Sx(), &p.Sy(), &p.Sz() ); 
  if( X*X + Y*Y>10.) fNMCPoints++;
}

void AliHLTTPCCAPerformance::CreateHistos()
{
  //* create performance histogramms
  TDirectory *curdir = gDirectory;
  fHistoDir = gROOT->mkdir("HLTTPCCATrackerPerformance");
  fHistoDir->cd();
  gDirectory->mkdir("TrackFit");
  gDirectory->cd("TrackFit");  
  
  fhResY = new TH1D("resY", "track Y resoltion [cm]", 30, -.5, .5);
  fhResZ = new TH1D("resZ", "track Z resoltion [cm]", 30, -.5, .5);
  fhResSinPhi = new TH1D("resSinPhi", "track SinPhi resoltion ", 30, -.03, .03);
  fhResDzDs = new TH1D("resDzDs", "track DzDs resoltion ", 30, -.01, .01);
  fhResPt = new TH1D("resPt", "track telative Pt resoltion", 30, -.2, .2);
  fhPullY = new TH1D("pullY", "track Y pull", 30, -10., 10.);
  fhPullZ = new TH1D("pullZ", "track Z pull", 30, -10., 10.);
  fhPullSinPhi = new TH1D("pullSinPhi", "track SinPhi pull", 30, -10., 10.);
  fhPullDzDs = new TH1D("pullDzDs", "track DzDs pull", 30, -10., 10.);
  fhPullQPt = new TH1D("pullQPt", "track Q/Pt pull", 30, -10., 10.);

  gDirectory->cd("..");  

  fhEffVsP = new TProfile("EffVsP", "Eff vs P", 100, 0., 5.);
  fhGBEffVsP = new TProfile("GBEffVsP", "Global tracker: Eff vs P", 100, 0., 5.);

  gDirectory->mkdir("Clusters");
  gDirectory->cd("Clusters");  
  fhHitResY = new TH1D("resHitY", "Y cluster resoltion [cm]", 100, -2., 2.);
  fhHitResZ = new TH1D("resHitZ", "Z cluster resoltion [cm]", 100, -2., 2.);
  fhHitPullY = new TH1D("pullHitY", "Y cluster pull", 50, -10., 10.);
  fhHitPullZ = new TH1D("pullHitZ", "Z cluster pull", 50, -10., 10.);

  fhHitResY1 = new TH1D("resHitY1", "Y cluster resoltion [cm]", 100, -2., 2.);
  fhHitResZ1 = new TH1D("resHitZ1", "Z cluster resoltion [cm]", 100, -2., 2.);
  fhHitPullY1 = new TH1D("pullHitY1", "Y cluster pull", 50, -10., 10.);
  fhHitPullZ1 = new TH1D("pullHitZ1", "Z cluster pull", 50, -10., 10.);

  fhHitErrY = new TH1D("HitErrY", "Y cluster error [cm]", 100, 0., 1.);
  fhHitErrZ = new TH1D("HitErrZ", "Z cluster error [cm]", 100, 0., 1.);

  gDirectory->cd("..");  

  gDirectory->mkdir("Cells");
  gDirectory->cd("Cells");  
  fhCellPurity = new TH1D("CellPurity", "Cell Purity", 100, -0.1, 1.1);
  fhCellNHits = new TH1D("CellNHits", "Cell NHits", 40, 0., 40.);
  fhCellPurityVsN = new TProfile("CellPurityVsN", "Cell purity Vs N hits", 40, 2., 42.);
  fhCellPurityVsPt = new TProfile("CellPurityVsPt", "Cell purity Vs Pt", 100, 0., 5.);
  gDirectory->cd("..");  

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
  
  Int_t nRecTot = 0, nGhost=0, nRecOut=0;
  Int_t nMCAll = 0, nRecAll=0, nClonesAll=0;
  Int_t nMCRef = 0, nRecRef=0, nClonesRef=0;
  AliHLTTPCCATracker &slice = fTracker->Slices()[iSlice];

  Int_t firstSliceHit = 0;
  for( ; firstSliceHit<fTracker->NHits(); firstSliceHit++){
    if( fTracker->Hits()[firstSliceHit].ISlice()==iSlice ) break;
  }
  Int_t endSliceHit = firstSliceHit;  

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
	for( Int_t j=0; j<c.NHits(); j++){
	  AliHLTTPCCAHit &h = row.GetCellHit(c,j);
	  //cout<<"hit ID="<<h.ID()<<" of"<<fTracker->NHits()<<endl;
	  //cout<<"gb hit ID="<<fTracker->Hits()[h.ID()].ID()<<endl;
	  AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[h.ID()].ID()];
	  if( l.fLab[0]>=0 ) lb[nla++]= l.fLab[0];
	  if( l.fLab[1]>=0 ) lb[nla++]= l.fLab[1];
	  if( l.fLab[2]>=0 ) lb[nla++]= l.fLab[2];
	}
	sort( lb, lb+nla );
	Int_t labmax = -1, labcur=-1, lmax = 0, lcurr=0;	
	for( Int_t i=0; i<nla; i++ ){
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

	Int_t label = labmax;
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
          
    for( Int_t ih=firstSliceHit; ih<endSliceHit; ih++){
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

  Int_t traN = slice.NOutTracks();
  Int_t *traLabels = 0; 
  Double_t *traPurity = 0;
  traLabels = new Int_t[traN];
  traPurity = new Double_t[traN];
  {
    for (Int_t itr=0; itr<traN; itr++) {
      traLabels[itr]=-1;
      traPurity[itr]= 0;
      AliHLTTPCCAOutTrack &tCA = slice.OutTracks()[itr];
      Int_t nhits = tCA.NHits();
      Int_t *lb = new Int_t[nhits*3];
      Int_t nla=0;
      //cout<<"\nHit labels:"<<endl;
      for( Int_t ihit=0; ihit<nhits; ihit++){
	Int_t index = slice.OutTrackHits()[tCA.FirstHitRef()+ihit];
	AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[index].ID()];
	//cout<<l.fLab[0]<<" "<<l.fLab[1]<<" "<<l.fLab[2]<<endl;
	if(l.fLab[0]>=0 ) lb[nla++]= l.fLab[0];
	if(l.fLab[1]>=0 ) lb[nla++]= l.fLab[1];
	if(l.fLab[2]>=0 ) lb[nla++]= l.fLab[2];
      }
      sort( lb, lb+nla );
      Int_t labmax = -1, labcur=-1, lmax = 0, lcurr=0;
      for( Int_t i=0; i<nla; i++ ){
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
      for( Int_t ihit=0; ihit<nhits; ihit++){
	Int_t index = slice.OutTrackHits()[tCA.FirstHitRef()+ihit];
	AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[index].ID()];
	if( l.fLab[0] == labmax || l.fLab[1] == labmax || l.fLab[2] == labmax 
	    ) lmax++;
      }
      traLabels[itr] = labmax;
      traPurity[itr] = ( (nhits>0) ?double(lmax)/double(nhits) :0 );
      //cout<<"perf track "<<itr<<": "<<nhits<<" "<<labmax<<" "<<traPurity[itr]<<endl;
      if( lb ) delete[] lb;
    }
  }

  nRecTot+= traN;

  for(Int_t itr=0; itr<traN; itr++){      
    if( traPurity[itr]<.9 || traLabels[itr]<0 || traLabels[itr]>=fNMCTracks){
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
  //SG!!!
  /*  
  fStatNEvents=0;
    fStatNRecTot=0;
    fStatNRecOut=0;
    fStatNGhost=0;
    fStatNMCAll=0;
    fStatNRecAll=0;
    fStatNClonesAll=0;
    fStatNMCRef=0;
    fStatNRecRef=0;
    fStatNClonesRef=0;
  */
  fStatNEvents++;
  for( Int_t islice=0; islice<fTracker->NSlices(); islice++){ 
    SlicePerformance(islice,0);
  }

  // global tracker performance
  {
      if( !fTracker ) return;

      Int_t nRecTot = 0, nGhost=0, nRecOut=0;
      Int_t nMCAll = 0, nRecAll=0, nClonesAll=0;
      Int_t nMCRef = 0, nRecRef=0, nClonesRef=0;

      // Select reconstructable MC tracks
   
      {
	for (Int_t imc=0; imc<fNMCTracks; imc++) fMCTracks[imc].NHits() = 0;
          
	for( Int_t ih=0; ih<fNHits; ih++){
	  AliHLTTPCCAHitLabel &l = fHitLabels[ih];
	  if( l.fLab[0]>=0 ) fMCTracks[l.fLab[0]].NHits()++;
	  if( l.fLab[1]>=0 ) fMCTracks[l.fLab[1]].NHits()++;
	  if( l.fLab[2]>=0 ) fMCTracks[l.fLab[2]].NHits()++;
	}
    
	for (Int_t imc=0; imc<fNMCTracks; imc++) {		
	  AliHLTTPCCAMCTrack &mc = fMCTracks[imc];
	  mc.Set() = 0;
	  mc.NReconstructed() = 0;
	  mc.NTurns() = 1;
	  if( mc.NHits() >=  50 && mc.P()>=.05 ){
	    mc.Set() = 1;
	    nMCAll++;
	    if( mc.P()>=1. ){
	      mc.Set() = 2;
	      nMCRef++;
	    }
	  }
	}
      }

      Int_t traN = fTracker->NTracks();
      Int_t *traLabels = 0;
      Double_t *traPurity = 0;
      traLabels = new Int_t[traN];
      traPurity = new Double_t[traN];
      {
	for (Int_t itr=0; itr<traN; itr++) {
	  traLabels[itr]=-1;
	  traPurity[itr]= 0;
	  AliHLTTPCCAGBTrack &tCA = fTracker->Tracks()[itr];
	  Int_t nhits = tCA.NHits();
	  Int_t *lb = new Int_t[nhits*3];
	  Int_t nla=0;
	  for( Int_t ihit=0; ihit<nhits; ihit++){
	    Int_t index = fTracker->TrackHits()[tCA.FirstHitRef()+ihit];
	    AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[index].ID()];
	    if(l.fLab[0]>=0 ) lb[nla++]= l.fLab[0];
	    if(l.fLab[1]>=0 ) lb[nla++]= l.fLab[1];
	    if(l.fLab[2]>=0 ) lb[nla++]= l.fLab[2];
	  }
	  sort( lb, lb+nla );
	  Int_t labmax = -1, labcur=-1, lmax = 0, lcurr=0;
	  for( Int_t i=0; i<nla; i++ ){
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
	  for( Int_t ihit=0; ihit<nhits; ihit++){
	    Int_t index = fTracker->TrackHits()[tCA.FirstHitRef()+ihit];
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
      for(Int_t itr=0; itr<traN; itr++){      
	if( traPurity[itr]<.9 || traLabels[itr]<0 || traLabels[itr]>=fNMCTracks){
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

	// track resolutions
	while( TMath::Abs(mc.TPCPar()[0]) + TMath::Abs(mc.TPCPar()[1])>1 ){
	  if( traPurity[itr]<.90 ) break;
	  AliHLTTPCCAGBTrack &t = fTracker->Tracks()[itr];
	  AliHLTTPCCATrackParam p = t.Param();
	  Double_t cosA = TMath::Cos( t.Alpha() );
	  Double_t sinA = TMath::Sin( t.Alpha() );
	  Double_t mcX =  mc.TPCPar()[0]*cosA + mc.TPCPar()[1]*sinA;
	  Double_t mcY = -mc.TPCPar()[0]*sinA + mc.TPCPar()[1]*cosA;
	  Double_t mcZ =  mc.TPCPar()[2];
	  Double_t mcEx =  mc.TPCPar()[3]*cosA + mc.TPCPar()[4]*sinA;
	  Double_t mcEy = -mc.TPCPar()[3]*sinA + mc.TPCPar()[4]*cosA;
	  Double_t mcEz =  mc.TPCPar()[5];
	  Double_t mcEt = TMath::Sqrt(mcEx*mcEx + mcEy*mcEy);
	  if( TMath::Abs(mcEt)<1.e-4 ) break;
	  Double_t mcSinPhi = mcEy / mcEt;
	  Double_t mcDzDs   = mcEz / mcEt;
	  Double_t mcQPt = mc.TPCPar()[6]/ mcEt;
	  if( TMath::Abs(mcQPt)<1.e-4 ) break;
	  Double_t mcPt = 1./TMath::Abs(mcQPt);
	  if( mcPt<1. ) break;
	  if( t.NHits() <  50 ) break;
	  Double_t bz = fTracker->Slices()[0].Param().Bz();
	  if( !p.TransportToXWithMaterial( mcX, bz ) ) break;
	  if( p.GetCosPhi()*mcEx < 0 ){ // change direction
	    mcSinPhi = -mcSinPhi;
	    mcDzDs = -mcDzDs;
	    mcQPt = -mcQPt;
	  }
	  const Double_t kCLight = 0.000299792458;  
	  Double_t k2QPt = 100;
	  if( TMath::Abs(bz)>1.e-4 ) k2QPt= 1./(bz*kCLight);
	  Double_t qPt = p.GetKappa()*k2QPt;
	  Double_t pt = 100;
	  if( TMath::Abs(qPt) >1.e-4 ) pt = 1./TMath::Abs(qPt);
	  
	  fhResY->Fill( p.GetY() - mcY ); 
	  fhResZ->Fill( p.GetZ() - mcZ );
	  fhResSinPhi->Fill( p.GetSinPhi() - mcSinPhi );
	  fhResDzDs->Fill( p.GetDzDs() - mcDzDs );
	  fhResPt->Fill( ( pt - mcPt )/mcPt );

	  if( p.GetErr2Y()>0 ) fhPullY->Fill( (p.GetY() - mcY)/TMath::Sqrt(p.GetErr2Y()) ); 
	  if( p.GetErr2Z()>0 ) fhPullZ->Fill( (p.GetZ() - mcZ)/TMath::Sqrt(p.GetErr2Z()) ); 
	  if( p.GetErr2SinPhi()>0 ) fhPullSinPhi->Fill( (p.GetSinPhi() - mcSinPhi)/TMath::Sqrt(p.GetErr2SinPhi()) ); 
	  if( p.GetErr2DzDs()>0 ) fhPullDzDs->Fill( (p.DzDs() - mcDzDs)/TMath::Sqrt(p.GetErr2DzDs()) ); 
	  if( p.GetErr2Kappa()>0 ) fhPullQPt->Fill( (qPt - mcQPt)/TMath::Sqrt(p.GetErr2Kappa()*k2QPt*k2QPt) ); 
	  break;
	}
      }

      for (Int_t ipart=0; ipart<fNMCTracks; ipart++) {		
	AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
	if( mc.Set()>0 ) fhGBEffVsP->Fill(mc.P(), ( mc.NReconstructed()>0 ?1 :0));
      }

      if( traLabels ) delete[] traLabels;
      if( traPurity ) delete[] traPurity;

      fStatGBNRecTot += nRecTot;
      fStatGBNRecOut += nRecOut;
      fStatGBNGhost  += nGhost;
      fStatGBNMCAll  += nMCAll;
      fStatGBNRecAll  += nRecAll;
      fStatGBNClonesAll  += nClonesAll;
      fStatGBNMCRef  += nMCRef;
      fStatGBNRecRef  += nRecRef;
      fStatGBNClonesRef  += nClonesRef;
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

  // cluster pulls

  if( fDoClusterPulls && fNMCPoints>0 ) {

    {
      for (Int_t ipart=0; ipart<fNMCTracks; ipart++) {		
	AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
	mc.NMCPoints() = 0;
      }
      sort(fMCPoints, fMCPoints+fNMCPoints, AliHLTTPCCAMCPoint::Compare );
      
      for( Int_t ip=0; ip<fNMCPoints; ip++ ){
	AliHLTTPCCAMCPoint &p = fMCPoints[ip];
	AliHLTTPCCAMCTrack &t = fMCTracks[p.TrackID()];
	if( t.NMCPoints()==0 ) t.FirstMCPointID() = ip;
	t.NMCPoints()++;
      }
    }

    for( Int_t ih=0; ih<fNHits; ih++ ){

      AliHLTTPCCAGBHit &hit = fTracker->Hits()[ih];
      AliHLTTPCCAHitLabel &l = fHitLabels[ih];

      if( l.fLab[0]<0 || l.fLab[0]>=fNMCTracks
	  || l.fLab[1]>=0 || l.fLab[2]>=0       ) continue;

      Int_t lab = l.fLab[0];

      AliHLTTPCCAMCTrack &track = fMCTracks[lab];
      //if( track.Pt()<1. ) continue;
      Int_t ip1=-1, ip2=-1;
      Double_t d1 = 1.e20, d2=1.e20;
      for( Int_t ip=0; ip<track.NMCPoints(); ip++ ){
        AliHLTTPCCAMCPoint &p = fMCPoints[track.FirstMCPointID() + ip];
        if( p.ISlice() != hit.ISlice() ) continue;        
        Double_t dx = p.Sx()-hit.X();
        Double_t dy = p.Sy()-hit.Y();
        Double_t dz = p.Sz()-hit.Z();
        Double_t d = dx*dx + dy*dy + dz*dz;
        if( p.Sx()< hit.X() ){
	  if( d<d1 ){
	    d1 = d;
	    ip1 = ip;
	  }
	}else{
	  if( d<d2 ){
	    d2 = d;
	    ip2 = ip;
	  }
	}
      }

      if( ip1<0 || ip2<0 ) continue;

      AliHLTTPCCAMCPoint &p1 = fMCPoints[track.FirstMCPointID() + ip1];
      AliHLTTPCCAMCPoint &p2 = fMCPoints[track.FirstMCPointID() + ip2];
      Double_t dx = p2.Sx() - p1.Sx();
      Double_t dy = p2.Sy() - p1.Sy();
      Double_t dz = p2.Sz() - p1.Sz();
      if( TMath::Abs(dx)>1.e-8 && TMath::Abs(p1.Sx()-hit.X())<2. && TMath::Abs(p2.Sx()-hit.X())<2.  ){
        Double_t sx = hit.X();
        Double_t sy = p1.Sy() + dy/dx*(sx-p1.Sx());
        Double_t sz = p1.Sz() + dz/dx*(sx-p1.Sx());
	
	Float_t errY, errZ;
	{
	  AliHLTTPCCATrackParam t;
	  t.Z() = sz;
	  t.SinPhi() = dy/TMath::Sqrt(dx*dx+dy*dy);
	  t.CosPhi() = dx/TMath::Sqrt(dx*dx+dy*dy);
	  t.DzDs() = dz/TMath::Sqrt(dx*dx+dy*dy);
	  fTracker->GetErrors2(hit,t,errY, errZ );
	  errY = TMath::Sqrt(errY);
	  errZ = TMath::Sqrt(errZ);
	} 
             
	fhHitResY->Fill((hit.Y()-sy));
	fhHitResZ->Fill((hit.Z()-sz));
	fhHitPullY->Fill((hit.Y()-sy)/errY);
	fhHitPullZ->Fill((hit.Z()-sz)/errZ);
	if( track.Pt()>=1. ){
	  fhHitResY1->Fill((hit.Y()-sy));
	  fhHitResZ1->Fill((hit.Z()-sz));
	  fhHitPullY1->Fill((hit.Y()-sy)/errY);
	  fhHitPullZ1->Fill((hit.Z()-sz)/errZ);
	}
      }
    }   
  }

  {
    cout<<"\nSlice tracker performance: \n"<<endl;
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
	<<fTracker->StatTime(7)/fTracker->StatNEvents()*1.e3<<"] "
	<<fTracker->StatTime(8)/fTracker->StatNEvents()*1.e3<<" "
	<<" msec/event "<<endl;
  }

  {
    cout<<"\nGlobal tracker performance: \n"<<endl;
    cout<<" N tracks : "
	<<fStatGBNMCAll/fStatNEvents<<" mc all, "
	<<fStatGBNMCRef/fStatNEvents<<" mc ref, "
	<<fStatGBNRecTot/fStatNEvents<<" rec total, "
	<<fStatGBNRecAll/fStatNEvents<<" rec all, "
	<<fStatGBNClonesAll/fStatNEvents<<" clones all, "
	<<fStatGBNRecRef/fStatNEvents<<" rec ref, "
	<<fStatGBNClonesRef/fStatNEvents<<" clones ref, "
	<<fStatGBNRecOut/fStatNEvents<<" out, "
	<<fStatGBNGhost/fStatNEvents<<" ghost"<<endl;
  
    Int_t nRecExtr = fStatGBNRecAll - fStatGBNRecRef;
    Int_t nMCExtr = fStatGBNMCAll - fStatGBNMCRef;
    Int_t nClonesExtr = fStatGBNClonesAll - fStatGBNClonesRef;
    
    Double_t dRecTot = (fStatGBNRecTot>0 ) ? fStatGBNRecTot :1;
    Double_t dMCAll = (fStatGBNMCAll>0 ) ? fStatGBNMCAll :1;
    Double_t dMCRef = (fStatGBNMCRef>0 ) ? fStatGBNMCRef :1;
    Double_t dMCExtr = (nMCExtr>0 ) ? nMCExtr :1;
    Double_t dRecAll = (fStatGBNRecAll+fStatGBNClonesAll>0 ) ? fStatGBNRecAll+fStatGBNClonesAll :1;
    Double_t dRecRef = (fStatGBNRecRef+fStatGBNClonesRef>0 ) ? fStatGBNRecRef+fStatGBNClonesRef :1;
    Double_t dRecExtr = (nRecExtr+nClonesExtr>0 ) ? nRecExtr+nClonesExtr :1;
    
    cout<<" EffRef = "<< fStatGBNRecRef/dMCRef
	<<", CloneRef = " << fStatGBNClonesRef/dRecRef <<endl;
    cout<<" EffExtra = "<<nRecExtr/dMCExtr
	<<", CloneExtra = " << nClonesExtr/dRecExtr<<endl;
    cout<<" EffAll = "<<fStatGBNRecAll/dMCAll
	<<", CloneAll = " << fStatGBNClonesAll/dRecAll<<endl;
    cout<<" Out = "<<fStatGBNRecOut/dRecTot
	<<", Ghost = "<<fStatGBNGhost/dRecTot<<endl;
    cout<<" Time = "<<( fTracker->StatTime(0)+fTracker->StatTime(9) )/fTracker->StatNEvents()*1.e3<<" msec/event "<<endl;
    cout<<" Local timers: "<<endl;
    cout<<" slice tracker "<<fTracker->StatTime(0)/fTracker->StatNEvents()*1.e3<<": "
	<<fTracker->StatTime(1)/fTracker->StatNEvents()*1.e3<<" "
	<<fTracker->StatTime(2)/fTracker->StatNEvents()*1.e3<<" "
	<<fTracker->StatTime(3)/fTracker->StatNEvents()*1.e3<<" "
	<<fTracker->StatTime(4)/fTracker->StatNEvents()*1.e3<<" "
	<<fTracker->StatTime(5)/fTracker->StatNEvents()*1.e3<<"["
	<<fTracker->StatTime(6)/fTracker->StatNEvents()*1.e3<<"/"
	<<fTracker->StatTime(7)/fTracker->StatNEvents()*1.e3<<"] "
	<<fTracker->StatTime(8)/fTracker->StatNEvents()*1.e3
	<<" msec/event "<<endl;
    cout<<" GB merger "<<fTracker->StatTime(9)/fTracker->StatNEvents()*1.e3<<": "
	<<fTracker->StatTime(10)/fTracker->StatNEvents()*1.e3<<", "
	<<fTracker->StatTime(11)/fTracker->StatNEvents()*1.e3<<", "
	<<fTracker->StatTime(12)/fTracker->StatNEvents()*1.e3<<" "
	<<" msec/event "<<endl;
  }

  WriteHistos();
}

void AliHLTTPCCAPerformance::WriteMCEvent( ostream &out )
{
  out<<fNMCTracks<<endl;
  for( Int_t it=0; it<fNMCTracks; it++ ){
    AliHLTTPCCAMCTrack &t = fMCTracks[it];
    out<<it<<" ";
    out<<t.PDG()<<endl;
    for( Int_t i=0; i<7; i++ ) out<<t.Par()[i]<<" ";
    out<<endl<<"    ";
    for( Int_t i=0; i<7; i++ ) out<<t.TPCPar()[i]<<" ";
    out<<endl<<"    ";
    out<< t.P()<<" ";
    out<< t.Pt()<<" ";
    out<< t.NMCPoints()<<" ";
    out<< t.FirstMCPointID()<<" ";
    out<< t.NHits()<<" ";
    out<< t.NReconstructed()<<" ";
    out<< t.Set()<<" ";
    out<< t.NTurns()<<endl;
  }

  out<<fNHits<<endl;
  for( Int_t ih=0; ih<fNHits; ih++ ){
    AliHLTTPCCAHitLabel &l = fHitLabels[ih];
    out<<l.fLab[0]<<" "<<l.fLab[1]<<" "<<l.fLab[2]<<endl;
  }
}

void AliHLTTPCCAPerformance::WriteMCPoints( ostream &out )
{  
  out<<fNMCPoints<<endl;
  for( Int_t ip=0; ip<fNMCPoints; ip++ ){
    AliHLTTPCCAMCPoint &p = fMCPoints[ip];
    out<< p.X()<<" ";
    out<< p.Y()<<" ";
    out<< p.Z()<<" ";
    out<< p.Sx()<<" ";
    out<< p.Sy()<<" ";
    out<< p.Sz()<<" ";
    out<< p.Time()<<" ";
    out<< p.ISlice()<<" ";
    out<< p.TrackID()<<endl;
  }
}

void AliHLTTPCCAPerformance::ReadMCEvent( istream &in )
{
  StartEvent();
  if( fMCTracks ) delete[] fMCTracks;
  fMCTracks = 0;
  fNMCTracks = 0;
  if( fHitLabels ) delete[] fHitLabels;
  fHitLabels = 0;
  fNHits = 0;
  if( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fNMCPoints = 0;

  in>>fNMCTracks;
  fMCTracks = new AliHLTTPCCAMCTrack[fNMCTracks];
  for( Int_t it=0; it<fNMCTracks; it++ ){
    AliHLTTPCCAMCTrack &t = fMCTracks[it];
    Int_t j;
    in>>j;
    in>> t.PDG();
    for( Int_t i=0; i<7; i++ ) in>>t.Par()[i];
    for( Int_t i=0; i<7; i++ ) in>>t.TPCPar()[i];
    in>> t.P();
    in>> t.Pt();
    in>> t.NHits();
    in>> t.NMCPoints();
    in>> t.FirstMCPointID();
    in>> t.NReconstructed();
    in>> t.Set();
    in>> t.NTurns();
  }
  
  in>>fNHits;
  fHitLabels = new AliHLTTPCCAHitLabel[fNHits];
  for( Int_t ih=0; ih<fNHits; ih++ ){
    AliHLTTPCCAHitLabel &l = fHitLabels[ih];
    in>>l.fLab[0]>>l.fLab[1]>>l.fLab[2];
  }
}

void AliHLTTPCCAPerformance::ReadMCPoints( istream &in )
{
  if( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fNMCPoints = 0;
  
  in>>fNMCPoints;
  fMCPoints = new AliHLTTPCCAMCPoint[fNMCPoints];
  for( Int_t ip=0; ip<fNMCPoints; ip++ ){
    AliHLTTPCCAMCPoint &p = fMCPoints[ip];
    in>> p.X();
    in>> p.Y();
    in>> p.Z();
    in>> p.Sx();
    in>> p.Sy();
    in>> p.Sz();
    in>> p.Time();
    in>> p.ISlice();
    in>> p.TrackID();
  }
}

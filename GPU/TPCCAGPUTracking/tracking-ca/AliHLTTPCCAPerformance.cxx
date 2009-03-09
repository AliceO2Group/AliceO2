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

AliHLTTPCCAPerformance &AliHLTTPCCAPerformance::Instance()
{
  // reference to static object
  static AliHLTTPCCAPerformance gAliHLTTPCCAPerformance;
  return gAliHLTTPCCAPerformance;
}

AliHLTTPCCAPerformance::AliHLTTPCCAPerformance()
  : 
  fTracker(0),
  fHitLabels(0),
  fNHits(0),                   
  fMCTracks(0),  
  fNMCTracks(0),               
  fMCPoints(0),  
  fNMCPoints(0),               
  fDoClusterPulls(0),         
  fStatNEvents(0),
  fStatTime(0),
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
  fhPullYS(0),      
  fhPullZT(0),      
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
  fhGBEffVsP(0),
  fhGBEffVsPt(0),
  fhNeighQuality(0),
  fhNeighEff(0),
  fhNeighQualityVsPt(0),
  fhNeighEffVsPt(0),
  fhNeighDy(0),
  fhNeighDz(0),
  fhNeighChi(0),
  fhNeighDyVsPt(0),
  fhNeighDzVsPt(0),
  fhNeighChiVsPt(0), 
  fhNeighNCombVsArea(0),
  fhNHitsPerSeed (0),
  fhNHitsPerTrackCand(0),
  fhTrackLengthRef(0),
  fhRefRecoX(0),
  fhRefRecoY(0),
  fhRefRecoZ(0),
  fhRefRecoP(0),
  fhRefRecoPt(0),
  fhRefRecoAngleY(0),
  fhRefRecoAngleZ(0),
  fhRefRecoNHits(0),
  fhRefNotRecoX(0),
  fhRefNotRecoY(0),
  fhRefNotRecoZ(0),
  fhRefNotRecoP(0),
  fhRefNotRecoPt(0),
  fhRefNotRecoAngleY(0),
  fhRefNotRecoAngleZ(0),
  fhRefNotRecoNHits(0)
{
  //* constructor
}


AliHLTTPCCAPerformance::AliHLTTPCCAPerformance(const AliHLTTPCCAPerformance&)
  :
  fTracker(0),
  fHitLabels(0),
  fNHits(0),                   
  fMCTracks(0),  
  fNMCTracks(0),               
  fMCPoints(0),  
  fNMCPoints(0),               
  fDoClusterPulls(0),         
  fStatNEvents(0),
  fStatTime(0),
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
  fhPullYS(0),      
  fhPullZT(0),      
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
  fhGBEffVsP(0),
  fhGBEffVsPt(0),
  fhNeighQuality(0),
  fhNeighEff(0),
  fhNeighQualityVsPt(0),
  fhNeighEffVsPt(0),
  fhNeighDy(0),
  fhNeighDz(0),
  fhNeighChi(0),
  fhNeighDyVsPt(0),
  fhNeighDzVsPt(0),
  fhNeighChiVsPt(0), 
  fhNeighNCombVsArea(0),
  fhNHitsPerSeed (0),
  fhNHitsPerTrackCand(0),
  fhTrackLengthRef(0),
  fhRefRecoX(0),
  fhRefRecoY(0),
  fhRefRecoZ(0),
  fhRefRecoP(0),
  fhRefRecoPt(0),
  fhRefRecoAngleY(0),
  fhRefRecoAngleZ(0),
  fhRefRecoNHits(0),
  fhRefNotRecoX(0),
  fhRefNotRecoY(0),
  fhRefNotRecoZ(0),
  fhRefNotRecoP(0),
  fhRefNotRecoPt(0),
  fhRefNotRecoAngleY(0),
  fhRefNotRecoAngleZ(0),
  fhRefNotRecoNHits(0)
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

void AliHLTTPCCAPerformance::SetNMCTracks( Int_t NumberOfMCTracks )
{
  //* set number of MC tracks
  if( fMCTracks ) delete[] fMCTracks;
  fMCTracks = 0;
  fMCTracks = new AliHLTTPCCAMCTrack[ NumberOfMCTracks ];
  fNMCTracks = NumberOfMCTracks;
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
  p.SetTrackID( TrackID );
  p.SetX( X );
  p.SetY( Y );
  p.SetZ( Z );
  p.SetTime( Time );
  p.SetISlice( iSlice );
  Float_t sx, sy, sz;
  fTracker->Slices()[iSlice].Param().Global2Slice( X, Y, Z, &sx, &sy, &sz );
  p.SetSx(sx);
  p.SetSy(sy); 
  p.SetSz(sz); 
  if( X*X + Y*Y>10.) fNMCPoints++;
}

void AliHLTTPCCAPerformance::CreateHistos()
{
  //* create performance histogramms
  TDirectory *curdir = gDirectory;
  fHistoDir = gROOT->mkdir("HLTTPCCATrackerPerformance");
  fHistoDir->cd();

  gDirectory->mkdir("Neighbours");
  gDirectory->cd("Neighbours");  
  
  fhNeighQuality = new TProfile("NeighQuality", "Neighbours Quality vs row", 160, 0., 160.); 
  fhNeighEff = new TProfile("NeighEff", "Neighbours Efficiency vs row", 160, 0., 160.); 
  fhNeighQualityVsPt = new TProfile("NeighQualityVsPt", "Neighbours Quality vs Pt", 100, 0., 5.); 
  fhNeighEffVsPt = new TProfile("NeighEffVsPt", "Neighbours Efficiency vs Pt", 100, 0., 5.); 
  fhNeighDy = new TH1D("NeighDy","Neighbours dy",100,-10,10);
  fhNeighDz =  new TH1D("NeighDz","Neighbours dz",100,-10,10);
  fhNeighChi = new TH1D("NeighChi","Neighbours chi",100,0,20);

  fhNeighDyVsPt = new TH2D("NeighDyVsPt","NeighDyVsPt", 100,0,5, 100, -20,20);
  fhNeighDzVsPt = new TH2D("NeighDzVsPt","NeighDzVsPt", 100,0,5, 100, -20,20);
  fhNeighChiVsPt = new TH2D("NeighChiVsPt","NeighChiVsPt", 100,0,5, 100, 0,40);
  fhNeighNCombVsArea = new TH2D("NeighNCombVsArea","NeighNCombVsArea", 15,0,3, 40, 0,40);

  gDirectory->cd(".."); 

  gDirectory->mkdir("Tracklets");
  gDirectory->cd("Tracklets");

  fhNHitsPerSeed = new TH1D("NHitsPerSeed","NHitsPerSeed", 160,0,160);
  fhNHitsPerTrackCand = new TH1D("NHitsPerTrackCand","NHitsPerTrackCand", 160,0,160);
  gDirectory->cd(".."); 
  
  gDirectory->mkdir("Tracks");
  gDirectory->cd("Tracks");  

  fhTrackLengthRef = new TH1D("TrackLengthRef", "TrackLengthRef", 100,0,1);

  fhRefRecoX = new TH1D("fhRefRecoX","fhRefRecoX",100,0,200.);
  fhRefRecoY = new TH1D("fhRefRecoY","fhRefRecoY",100,-200,200.);
  fhRefRecoZ = new TH1D("fhRefRecoZ","fhRefRecoZ",100,-250,250.);


  fhRefRecoP = new TH1D("fhRefRecoP","fhRefRecoP",100,0,10.);
  fhRefRecoPt = new TH1D("fhRefRecoPt","fhRefRecoPt",100,0,10.);
  fhRefRecoAngleY = new TH1D("fhRefRecoAngleY","fhRefRecoAngleY",100,-180.,180.);
  fhRefRecoAngleZ = new TH1D("fhRefRecoAngleZ","fhRefRecoAngleZ",100,-180.,180);
  fhRefRecoNHits = new TH1D("fhRefRecoNHits","fhRefRecoNHits",100,0.,200);

  fhRefNotRecoX = new TH1D("fhRefNotRecoX","fhRefNotRecoX",100,0,200.);
  fhRefNotRecoY = new TH1D("fhRefNotRecoY","fhRefNotRecoY",100,-200,200.);
  fhRefNotRecoZ = new TH1D("fhRefNotRecoZ","fhRefNotRecoZ",100,-250,250.);


  fhRefNotRecoP = new TH1D("fhRefNotRecoP","fhRefNotRecoP",100,0,10.);
  fhRefNotRecoPt = new TH1D("fhRefNotRecoPt","fhRefNotRecoPt",100,0,10.);
  fhRefNotRecoAngleY = new TH1D("fhRefNotRecoAngleY","fhRefNotRecoAngleY",100,-180.,180.);
  fhRefNotRecoAngleZ = new TH1D("fhRefNotRecoAngleZ","fhRefNotRecoAngleZ",100,-180.,180);
  fhRefNotRecoNHits = new TH1D("fhRefNotRecoNHits","fhRefNotRecoNHits",100,0.,200);
  
  gDirectory->cd(".."); 

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
  fhPullYS = new TH1D("pullYS", "track Y+SinPhi chi deviation", 100, 0., 30.);
  fhPullZT = new TH1D("pullZT", "track Z+DzDs chi deviation ", 100, 0., 30.);

  gDirectory->cd("..");  

  fhEffVsP = new TProfile("EffVsP", "Eff vs P", 100, 0., 5.);
  fhGBEffVsP = new TProfile("GBEffVsP", "Global tracker: Eff vs P", 100, 0., 5.);
  fhGBEffVsPt = new TProfile("GBEffVsPt", "Global tracker: Eff vs Pt", 100, 0.2, 5.);

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
  
  { // Efficiency and quality of found neighbours
#ifdef XXX
    for( Int_t iRow=1; iRow<slice.Param().NRows()-1; iRow++ ){      
      AliHLTTPCCARow &row = slice.Rows()[iRow];
      AliHLTTPCCARow &rowP = slice.Rows()[iRow-1];
      AliHLTTPCCARow &rowN = slice.Rows()[iRow+1];      

      Int_t nHitsRow[fNMCTracks][3];
      Bool_t foundRow[fNMCTracks];
      Bool_t isPrim[fNMCTracks];
      for (Int_t imc=0; imc<fNMCTracks; imc++){ 
	for( Int_t i=0; i<3; i++ )nHitsRow[imc][i]=0;
	foundRow[imc]=0;
	Float_t y = fMCTracks[imc].Par()[1];
	Float_t z = fMCTracks[imc].Par()[2];
	isPrim[imc] = (y*y + z*z < 100. );
      }
    
      for( Int_t i=0; i<3; i++ ){
	AliHLTTPCCARow &r = slice.Rows()[iRow-1+i];      
	for (Int_t ih = 0; ih<r.NHits(); ih++){
	  AliHLTTPCCAHit &h = r.Hits()[ih];
	  AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[h.ID()].ID()];
	  if( l.fLab[0] <0 || l.fLab[1]>=0 ) continue;
	  nHitsRow[l.fLab[0]][i]++;
	}
      }

      for( Int_t it=0; it<row.NTriplets(); it++ ){
	AliHLTTPCCATriplet &t = row.Triplets()[it];
	if( !t.Alive() ) continue;
	AliHLTTPCCAHit &h = row.Hits()[t.HitMid()];
	AliHLTTPCCAHit &hP = rowP.Hits()[t.HitDown()];
	AliHLTTPCCAHit &hN = rowN.Hits()[t.HitUp()];
	AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[h.ID()].ID()];
	AliHLTTPCCAHitLabel &lP = fHitLabels[fTracker->Hits()[hP.ID()].ID()];
	AliHLTTPCCAHitLabel &lN = fHitLabels[fTracker->Hits()[hN.ID()].ID()];	
	Bool_t found = ( l.fLab[0]>=0 && lP.fLab[0]==l.fLab[0] && lN.fLab[0]==l.fLab[0] );
	if( found ) foundRow[l.fLab[0]] = 1;
	Bool_t isGhost = 1;
	for( Int_t il=0; il<3; il++ ){
	  if( l.fLab[il]<0 ) continue;
	  Bool_t okP=0, okN=0;
	  for( Int_t jl=0; jl<3; jl++ ){
	    if( lP.fLab[jl]==l.fLab[il] ) okP = 1;
	    if( lN.fLab[jl]==l.fLab[il] ) okN = 1;
	  }
	  if( okP && okN ) isGhost = 0; 
	}
	fhNeighQuality->Fill(iRow, !isGhost );
	if( l.fLab[0]>=0 ) fhNeighQualityVsPt->Fill(fMCTracks[l.fLab[0]].Pt(), !isGhost );
      }
    
      for (Int_t imc=0; imc<fNMCTracks; imc++){ 
	if( nHitsRow[imc][0]<=0 || nHitsRow[imc][1]<=0 ||nHitsRow[imc][2]<=0 ) continue;
	if( !isPrim[imc] ) continue;
	if( fMCTracks[imc].Pt()>0.2 ) fhNeighEff->Fill(iRow, foundRow[imc] );	
	fhNeighEffVsPt->Fill( fMCTracks[imc].Pt(), foundRow[imc] );	
      }
    }
#endif    
  } // efficiency and quality of found neighbours




  if(0){ // Local efficiency of found neighbours
#ifdef XXX    
    for( Int_t iRow=1; iRow<slice.Param().NRows()-1; iRow++ ){
      //cout<<"iRow="<<iRow<<endl;
      AliHLTTPCCARow &row = slice.Rows()[iRow];      
      AliHLTTPCCARow &rowP = slice.Rows()[iRow-1];      
      AliHLTTPCCARow &rowN = slice.Rows()[iRow+1];      
      Float_t xP = rowP.X() - row.X();
      Float_t xN = rowN.X() - row.X();
      Float_t tN = rowN.X()/row.X();
      Float_t tP = rowP.X()/row.X();
      Float_t chi2Cut = 2*3.*3.*(xN*xN+xP*xP);
      for (Int_t ih = 0; ih<row.NHits(); ih++){
	AliHLTTPCCAHit &h = row.Hits()[ih];
	AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[h.ID()].ID()];
	if( l.fLab[0] <0 || l.fLab[1]>=0 ) continue;
	Float_t yyP = h.Y()*tP;
	Float_t zzP = h.Z()*tP;
	Float_t yyN = h.Y()*tN;
	Float_t zzN = h.Z()*tN;
	Float_t mcPt = fMCTracks[l.fLab[0]].Pt();
	if(0){
	  Int_t jhP=-1;
	  Float_t bestDP = 1.e10;
	  for (Int_t ihP = 0; ihP<rowP.NHits(); ihP++){
	    AliHLTTPCCAHit &hP = rowP.Hits()[ihP];
	    AliHLTTPCCAHitLabel &lP = fHitLabels[fTracker->Hits()[hP.ID()].ID()];
	    Bool_t ok = 0;
	    for( Int_t il=0; il<3; il++ ){
	      if( lP.fLab[il]==l.fLab[0] ) ok = 1;
	    }
	    if( !ok ) continue;
	    Float_t dy = hP.Y()-yyP;
	    Float_t dz = hP.Z()-zzP;
	    Float_t d = dy*dy + dz*dz;
	    if( d<bestDP ){
	      bestDP = d;
	      jhP = ihP;
	    }
	  }

	  Int_t jhN=-1;
	  Float_t bestDN = 1.e10;
	  for (Int_t ihN = 0; ihN<rowN.NHits(); ihN++){
	    AliHLTTPCCAHit &hN = rowN.Hits()[ihN];
	    AliHLTTPCCAHitLabel &lN = fHitLabels[fTracker->Hits()[hN.ID()].ID()];
	    Bool_t ok = 0;
	    for( Int_t il=0; il<3; il++ ){
	      if( lN.fLab[il]==l.fLab[0] ) ok = 1;
	    }
	    if( !ok ) continue;
	    Float_t dy = hN.Y()-yyN;
	    Float_t dz = hN.Z()-zzN;
	    Float_t d = dy*dy + dz*dz;
	    if( d<bestDN ){
	      bestDN = d;
	      jhN = ihN;
	    }
	  }
	  if( jhP>=0 && jhN>=0 ){
	    AliHLTTPCCAHit &hP = rowP.Hits()[jhP];
	    AliHLTTPCCAHit &hN = rowN.Hits()[jhN];
	    fhNeighDyVsPt->Fill(mcPt, hP.Y()-yyP);
	    fhNeighDyVsPt->Fill(mcPt, hN.Y()-yyN);
	    fhNeighDzVsPt->Fill(mcPt, hP.Z()-zzP);
	    fhNeighDzVsPt->Fill(mcPt, hN.Z()-zzN);
	    Float_t dy = xP*(hN.Y()-h.Y()) - xN*(hP.Y()-h.Y());
	    Float_t dz = xP*(hN.Z()-h.Z()) - xN*(hP.Z()-h.Z());
	    Float_t chi2 = (dy*dy + dz*dz)/(xN*xN+xP*xP);
	    fhNeighChiVsPt->Fill(mcPt, TMath::Sqrt(chi2/2));	    
	  }
	  {
	    Float_t darea = .2;
	    Int_t nAreas = 15;
	    Int_t nComb[nAreas];
	    for( Int_t i=0; i<nAreas; i++ ) nComb[i]=0;
	    for (Int_t ihP = 0; ihP<rowP.NHits(); ihP++){
	      AliHLTTPCCAHit &hP = rowP.Hits()[ihP];
	      Float_t yyy = -xP*h.Y() - xN*(hP.Y()-h.Y());
	      Float_t zzz = -xP*h.Z() - xN*(hP.Z()-h.Z());
	      for (Int_t ihN = 0; ihN<rowN.NHits(); ihN++){
		AliHLTTPCCAHit &hN = rowN.Hits()[ihN];
		Float_t dy = xP*hN.Y()+yyy;
		Float_t dz = xP*hN.Z()+zzz;
		Float_t chi2 = (dy*dy + dz*dz);
		if( chi2 >chi2Cut ) continue;
		Float_t D = TMath::Abs( hP.Y()-yyP );
		Float_t d = TMath::Abs( hP.Z()-zzP );
		if( d>D ) D = d;
		d = TMath::Abs( hN.Y()-yyN );
		if( d>D ) D = d;
		d = TMath::Abs( hN.Z()-zzN );
		if( d>D ) D = d;		
		Int_t ic = (int) (D/darea);
		if( ic<nAreas ) nComb[ic]++;
	      }
	    }
	    Int_t j=0;
	    for( Int_t i=0; i<nAreas; i++ ){
	      j+=nComb[i];
	      if(j>0 ) fhNeighNCombVsArea->Fill(i*darea,j);
	    }	  
	  }
	}

	Float_t area = 5;
	const Int_t maxN = 1000;
	Int_t neighP[maxN];
	Int_t neighN[maxN];
	Float_t zP[maxN];
	Float_t yP[maxN];
	Float_t zN[maxN];
	Float_t yN[maxN];
	Int_t nNeighP = 0;
	Int_t nNeighN = 0;     
 	
	for (Int_t ihP = 0; ihP<rowP.NHits(); ihP++){
	  AliHLTTPCCAHit &hP = rowP.Hits()[ihP];
	  if( TMath::Abs(hP.Y()-yyP)>area || TMath::Abs(hP.Z()-zzP)>area ) continue;
	  AliHLTTPCCAHitLabel &lP = fHitLabels[fTracker->Hits()[hP.ID()].ID()];
	  Bool_t ok = 0;
	  for( Int_t il=0; il<3; il++ ){
	    if( lP.fLab[il]==l.fLab[0] ) ok = 1;
	  }
	  if( !ok ) continue;
	  neighP[nNeighP] = ihP;
	  zP[nNeighP] = xN*(hP.Z()-h.Z());
	  yP[nNeighP] = xN*(hP.Y()-h.Y());
	  nNeighP++;
	  if( nNeighP>=maxN ) break;
	}
	for (Int_t ihN = 0; ihN<rowN.NHits(); ihN++){
	  AliHLTTPCCAHit &hN = rowN.Hits()[ihN];
	  if( TMath::Abs(hN.Y()-yyN)>area || TMath::Abs(hN.Z()-zzN)>area ) continue;
	  AliHLTTPCCAHitLabel &lN = fHitLabels[fTracker->Hits()[hN.ID()].ID()];
	  Bool_t ok = 0;
	  for( Int_t il=0; il<3; il++ ){
	    if( lN.fLab[il]==l.fLab[0] ) ok = 1;
	  }
	  if( !ok ) continue;
	  neighN[nNeighN] = ihN;
	  zN[nNeighN] = xP*(hN.Z()-h.Z());
	  yN[nNeighN] = xP*(hN.Y()-h.Y());
	  nNeighN++;
	  if( nNeighN>=maxN ) break;
	}

	if( nNeighN<=0 || nNeighP<=0 ) continue;

	{
	  // neighbours found, look for the straight line connection
	  Int_t bestP=-1, bestN=-1;
	  Float_t bestD=1.e10;
	  for( Int_t iP=0; iP<nNeighP; iP++ ){
	    for( Int_t iN=0; iN<nNeighN; iN++ ){
	      Float_t dy = yP[iP]-yN[iN];
	      Float_t dz = zP[iP]-zN[iN];
	      Float_t d = dy*dy + dz*dz;
	      if( d<bestD){
		bestD = d;
		bestP = iP;
		bestN = iN;
	      }
	    }	
	  }
	  if( bestP>=0 ){
	    AliHLTTPCCAHit &hP = rowP.Hits()[neighP[bestP]];
	    AliHLTTPCCAHit &hN = rowN.Hits()[neighN[bestN]];
	    Float_t yP = hP.Y() - h.Y();
	    Float_t zP = hP.Z() - h.Z();
	    Float_t yN = hN.Y() - h.Y();
	    Float_t zN = hN.Z() - h.Z();
	    Float_t dy = yP+yN;//(xN*yP - xP*yN)/TMath::Sqrt(xN*xN+xP*xP);
	    Float_t dz = zP+zN;//(xN*zP - xP*zN)/TMath::Sqrt(xN*xN+xP*xP);
	    //if( fMCTracks[l.fLab[0]].Pt()>1 ){
	    fhNeighChi->Fill(TMath::Sqrt(bestD/(xN*xN+xP*xP)/4.));
	    fhNeighDy->Fill(dy);
	    fhNeighDz->Fill(dz);	  
	    //}
	  }
	}

	Bool_t okP=0, okN=0;
	if( h.FirstTriplet()>=0 ){
	  for( Int_t it=h.FirstTriplet(); it<row.NTriplets(); it++ ){
	    AliHLTTPCCATriplet &t = row.Triplets()[it];
	    if( t.HitMid()!=ih ) break;
	    //if( !t.Alive() ) continue;
	    AliHLTTPCCAHit &hP = rowP.Hits()[t.HitDown()];
	    AliHLTTPCCAHit &hN = rowN.Hits()[t.HitUp()];
	    AliHLTTPCCAHitLabel &lP = fHitLabels[fTracker->Hits()[hP.ID()].ID()];
	    AliHLTTPCCAHitLabel &lN = fHitLabels[fTracker->Hits()[hN.ID()].ID()];
	    for( Int_t jl=0; jl<3; jl++ ){
	      if( lP.fLab[jl]==l.fLab[0] ) okP = 1;
	      if( lN.fLab[jl]==l.fLab[0] ) okN = 1;
	    }
	  }	
	}
	//fhNeighEff->Fill(iRow, okP&&okN );	
	//fhNeighEffVsPt->Fill( fMCTracks[l.fLab[0]].Pt(), okP&&okN );	
      }
    }
#endif
  }
  

  // Select reconstructable MC tracks

  {
    for (Int_t imc=0; imc<fNMCTracks; imc++) fMCTracks[imc].NHits() = 0;
        
    for( Int_t ih=firstSliceHit; ih<endSliceHit; ih++){
      Int_t id = fTracker->Hits()[ih].ID();
      if( id<0 || id>fNHits ) break;
      AliHLTTPCCAHitLabel &l = fHitLabels[id];
      if( l.fLab[0]>=0 ) fMCTracks[l.fLab[0]].NHits()++;
      if( l.fLab[1]>=0 ) fMCTracks[l.fLab[1]].NHits()++;
      if( l.fLab[2]>=0 ) fMCTracks[l.fLab[2]].NHits()++;
    }
    
    for (Int_t imc=0; imc<fNMCTracks; imc++) {		
      AliHLTTPCCAMCTrack &mc = fMCTracks[imc];
      mc.Set() = 0;
      mc.NReconstructed() = 0;
      mc.NTurns() = 1;
      if( mc.NHits() >=  30 && mc.P()>=.05 ){
	mc.Set() = 1;
	nMCAll++;
	if( mc.NHits() >=  30 && mc.P()>=1. ){
	  mc.Set() = 2;
	  nMCRef++;
	}
      }
    }
  }

  Int_t traN = *slice.NOutTracks();
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
	Int_t index = firstSliceHit + slice.OutTrackHits()[tCA.FirstHitRef()+ihit];
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
	Int_t index = firstSliceHit + slice.OutTrackHits()[tCA.FirstHitRef()+ihit];
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


void AliHLTTPCCAPerformance::Performance( fstream *StatFile )
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
	AliHLTTPCCAGBTrack &tCA = fTracker->Tracks()[itr];
	AliHLTTPCCAMCTrack &mc = fMCTracks[traLabels[itr]];	
	
	mc.NReconstructed()++;
	if( mc.Set()== 0 ) nRecOut++;
	else{
	  if( mc.NReconstructed()==1 ) nRecAll++;
	  else if(mc.NReconstructed() > mc.NTurns() ) nClonesAll++;
	  if( mc.Set()==2 ){
	    if( mc.NReconstructed()==1 ) nRecRef++;
	    else if(mc.NReconstructed() > mc.NTurns() ) nClonesRef++;
	    fhTrackLengthRef->Fill( tCA.NHits()/((Double_t) mc.NHits()));
	  }
	}      

	// track resolutions
	while( mc.Set()==2 && TMath::Abs(mc.TPCPar()[0]) + TMath::Abs(mc.TPCPar()[1])>1 ){
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
	  fhPullYS->Fill( TMath::Sqrt(fTracker->GetChi2( p.GetY(), p.GetSinPhi(), p.GetCov()[0], p.GetCov()[3], p.GetCov()[5], 
					     mcY, mcSinPhi, 0,0,0 )));
	  fhPullZT->Fill( TMath::Sqrt(fTracker->GetChi2( p.GetZ(), p.GetDzDs(), p.GetCov()[2], p.GetCov()[7], p.GetCov()[9], 
			  mcZ, mcDzDs, 0,0,0 ) ));

	  break;
	}
      }

      for (Int_t ipart=0; ipart<fNMCTracks; ipart++) {		
	AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
	if( mc.Set()>0 ) fhGBEffVsP->Fill(mc.P(), ( mc.NReconstructed()>0 ?1 :0));
	if( mc.Set()>0 ) fhGBEffVsPt->Fill(mc.Pt(), ( mc.NReconstructed()>0 ?1 :0));
	if( mc.Set()==2 ){ 
	  const Double_t *p = mc.TPCPar();
	  Double_t r = TMath::Sqrt(p[0]*p[0] + p[1]*p[1]);
	  Double_t cosA = p[0]/r;
	  Double_t sinA = p[1]/r;


	  Double_t phipos = TMath::Pi()+TMath::ATan2(-p[1], -p[0]);
	  Double_t alpha =  TMath::Pi()*(20*((((Int_t)(phipos*180/TMath::Pi()))/20))+10)/180.;
	  cosA = TMath::Cos(alpha);
	  sinA = TMath::Sin(alpha);

	  Double_t mcX =  p[0]*cosA + p[1]*sinA;
	  Double_t mcY = -p[0]*sinA + p[1]*cosA;
	  Double_t mcZ =  p[2];
	  Double_t mcEx =  p[3]*cosA + p[4]*sinA;
	  Double_t mcEy = -p[3]*sinA + p[4]*cosA;
	  Double_t mcEz =  p[5];
	  //Double_t mcEt = TMath::Sqrt(mcEx*mcEx + mcEy*mcEy);
	  Double_t angleY = TMath::ATan2(mcEy, mcEx)*180./TMath::Pi();
	  Double_t angleZ = TMath::ATan2(mcEz, mcEx)*180./TMath::Pi();

	  if( mc.NReconstructed()>0 ){
	    fhRefRecoX->Fill(mcX);
	    fhRefRecoY->Fill(mcY);
	    fhRefRecoZ->Fill(mcZ);
	    fhRefRecoP->Fill(mc.P());
	    fhRefRecoPt->Fill(mc.Pt());
	    fhRefRecoAngleY->Fill(angleY);
	    fhRefRecoAngleZ->Fill(angleZ);
	    fhRefRecoNHits->Fill(mc.NHits());
	  } else {
	    fhRefNotRecoX->Fill(mcX);
	    fhRefNotRecoY->Fill(mcY);
	    fhRefNotRecoZ->Fill(mcZ);
	    fhRefNotRecoP->Fill(mc.P());
	    fhRefNotRecoPt->Fill(mc.Pt());
	    fhRefNotRecoAngleY->Fill(angleY);
	    fhRefNotRecoAngleZ->Fill(angleZ);
	    fhRefNotRecoNHits->Fill(mc.NHits());
	  }
	}
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
	  t.SetZ( sz );
	  t.SetSinPhi( dy/TMath::Sqrt(dx*dx+dy*dy) );
	  t.SetCosPhi( dx/TMath::Sqrt(dx*dx+dy*dy) );
	  t.SetDzDs( dz/TMath::Sqrt(dx*dx+dy*dy) );
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
	<<fTracker->StatTime(5)/fTracker->StatNEvents()*1.e3<<" "
	<<fTracker->StatTime(6)/fTracker->StatNEvents()*1.e3<<" "
	<<fTracker->StatTime(7)/fTracker->StatNEvents()*1.e3<<" "
	<<fTracker->StatTime(8)/fTracker->StatNEvents()*1.e3<<" "
	<<" msec/event "<<endl;
  }

  {
    cout<<"\nGlobal tracker performance for "<<fStatNEvents<<" events: \n"<<endl;
    cout<<" N tracks : "
	<<fStatGBNMCAll<<" mc all, "
	<<fStatGBNMCRef<<" mc ref, "
	<<fStatGBNRecTot<<" rec total, "
	<<fStatGBNRecAll<<" rec all, "
	<<fStatGBNClonesAll<<" clones all, "
	<<fStatGBNRecRef<<" rec ref, "
	<<fStatGBNClonesRef<<" clones ref, "
	<<fStatGBNRecOut<<" out, "
	<<fStatGBNGhost<<" ghost"<<endl;
     cout<<" N tracks average : "
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

    if( StatFile && StatFile->is_open() ){
      fstream &out = *StatFile;

      //out<<"\nGlobal tracker performance for "<<fStatNEvents<<" events: \n"<<endl;
      //out<<" N tracks : "
      //<<fStatGBNMCAll/fStatNEvents<<" mc all, "
      //<<fStatGBNMCRef/fStatNEvents<<" mc ref, "
      // <<fStatGBNRecTot/fStatNEvents<<" rec total, "
      // <<fStatGBNRecAll/fStatNEvents<<" rec all, "
      // <<fStatGBNClonesAll/fStatNEvents<<" clones all, "
      // <<fStatGBNRecRef/fStatNEvents<<" rec ref, "
      // <<fStatGBNClonesRef/fStatNEvents<<" clones ref, "
      // <<fStatGBNRecOut/fStatNEvents<<" out, "
      // <<fStatGBNGhost/fStatNEvents<<" ghost"<<endl;
      fStatTime+=fTracker->SliceTrackerTime();
      double timeHz=0;
      if( fStatTime>1.e-4 ) timeHz = 1./fStatTime*fStatNEvents;

      out<<"<table border>"<<endl;
      out<<"<tr>"<<endl; 
      out<<"<td>      </td> <td align=center> RefSet </td> <td align=center> AllSet </td> <td align=center> ExtraSet </td>"<<endl;
      out<<"</tr>"<<endl;
      out<<"<tr>"<<endl;
      out<<"<td>Efficiency</td> <td align=center>"<<fStatGBNRecRef/dMCRef
	 <<"</td> <td align=center>"<<fStatGBNRecAll/dMCAll
	 <<"</td> <td align=center>"<<nRecExtr/dMCExtr
	 <<"</td>"<<endl;
      out<<"</tr>"<<endl;
      out<<"<tr> "<<endl;
      out<<"<td>Clone</td>      <td align=center>"<<fStatGBNClonesRef/dRecRef
	 <<"</td> <td align=center>"<<fStatGBNClonesAll/dRecAll
	 <<"</td> <td align=center>"<<nClonesExtr/dRecExtr
	 <<"</td>"<<endl;
      out<<"</tr>"<<endl;
      out<<"<tr> "<<endl;
      out<<"<td>Ghost</td>      <td colspan=3 align=center>"<<fStatGBNGhost/dRecTot
	 <<"</td>"<<endl;
      out<<"</tr>"<<endl;
      out<<"<tr> "<<endl;
      out<<"<td>Time</td>      <td colspan=3 align=center>"<<timeHz 
	 <<" ev/s</td>"<<endl;
      out<<"</tr>"<<endl;
      out<<"<tr> "<<endl;
      out<<"<td>N Events</td>      <td colspan=3 align=center>"<<fStatNEvents
	 <<"</td>"<<endl;
      out<<"</tr>"<<endl;
      out<<"</table>"<<endl;
    }

  }

  WriteHistos();


}

void AliHLTTPCCAPerformance::WriteMCEvent( ostream &out ) const
{
  // write MC information to the file
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

void AliHLTTPCCAPerformance::WriteMCPoints( ostream &out ) const
{  
  // write Mc points to the file
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
  // read mc info from the file
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
  // read mc points from the file
  if( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fNMCPoints = 0;
  
  in>>fNMCPoints;
  fMCPoints = new AliHLTTPCCAMCPoint[fNMCPoints];
  for( Int_t ip=0; ip<fNMCPoints; ip++ ){
    AliHLTTPCCAMCPoint &p = fMCPoints[ip];
    Float_t f;
    Int_t i;
    in>> f;
    p.SetX( f );
    in >> f;
    p.SetY( f );
    in >> f;
    p.SetZ( f );
    in >> f;
    p.SetSx( f );
    in >> f;
    p.SetSy( f );
    in >> f;
    p.SetSz( f );
    in >> f;
    p.SetTime( f );
    in >> i;
    p.SetISlice( i );
    in >> i;
    p.SetTrackID( i );
  }
}

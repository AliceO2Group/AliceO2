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

#include "AliTPCtrackerCA.h"

#include <TTree.h>
#include <Riostream.h>
#include "AliCluster.h"
#include "AliTPCClustersRow.h"
#include "AliTPCParam.h"
#include "AliRun.h"
#include "AliRunLoader.h"
#include "AliStack.h"

#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGBHit.h"
#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCAGBTrack.h"
#include "AliHLTTPCCAMCTrack.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAPerformance.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackConvertor.h"

#include "TMath.h"
#include "AliTPCLoader.h"
#include "AliTPC.h"
#include "AliTPCclusterMI.h"
#include "AliTPCTransform.h"
#include "AliTPCcalibDB.h"
#include "AliTPCReconstructor.h"
#include "AliTPCtrack.h"
#include "AliESDtrack.h"
#include "AliESDEvent.h"
#include "AliTrackReference.h"

//#include <fstream.h>

ClassImp(AliTPCtrackerCA)

AliTPCtrackerCA::AliTPCtrackerCA()
  :AliTracker(),fParam(0), fClusters(0), fNClusters(0), fHLTTracker(0),fHLTPerformance(0),fDoHLTPerformance(0),fDoHLTPerformanceClusters(0),fStatNEvents(0)
{
  //* default constructor
}

AliTPCtrackerCA::AliTPCtrackerCA(const AliTPCtrackerCA &):
  AliTracker(),fParam(0), fClusters(0), fNClusters(0), fHLTTracker(0),fHLTPerformance(0),fDoHLTPerformance(0),fDoHLTPerformanceClusters(0),fStatNEvents(0)
{
  //* dummy
}

AliTPCtrackerCA & AliTPCtrackerCA::operator=(const AliTPCtrackerCA& )
{
  //* dummy 
  return *this;
}


AliTPCtrackerCA::~AliTPCtrackerCA() 
{
  //* destructor
  if( fClusters ) delete[] fClusters;
  if( fHLTTracker ) delete fHLTTracker;
  if( fHLTPerformance ) delete fHLTPerformance;
}

AliTPCtrackerCA::AliTPCtrackerCA(const AliTPCParam *par): 
  AliTracker(),fParam(par), fClusters(0), fNClusters(0), fHLTTracker(0), fHLTPerformance(0),fDoHLTPerformance(0),fDoHLTPerformanceClusters(0),fStatNEvents(0)
{
  //* constructor

  DoHLTPerformance() = 1;
  DoHLTPerformanceClusters() = 1;

  fHLTTracker = new AliHLTTPCCAGBTracker;
  fHLTTracker->SetNSlices( fParam->GetNSector()/2 );

  if( fDoHLTPerformance ){
    fHLTPerformance = new AliHLTTPCCAPerformance;
    fHLTPerformance->SetTracker( fHLTTracker );
  }

  for( int iSlice=0; iSlice<fHLTTracker->NSlices(); iSlice++ ){
  
    Float_t bz = AliTracker::GetBz();

    Float_t inRmin = fParam->GetInnerRadiusLow();
    //Float_t inRmax = fParam->GetInnerRadiusUp();
    //Float_t outRmin = fParam->GetOuterRadiusLow(); 
    Float_t outRmax = fParam->GetOuterRadiusUp();
    Float_t plusZmin = 0.0529937; 
    Float_t plusZmax = 249.778; 
    Float_t minusZmin = -249.645; 
    Float_t minusZmax = -0.0799937; 
    Float_t dalpha = 0.349066;
    Float_t alpha = 0.174533 + dalpha*iSlice;
    
    Bool_t zPlus = (iSlice<18 );
    Float_t zMin =  zPlus ?plusZmin :minusZmin;
    Float_t zMax =  zPlus ?plusZmax :minusZmax;
    //TPCZmin = -249.645, ZMax = 249.778    
    //Float_t rMin =  inRmin;
    //Float_t rMax =  outRmax;
        
    Float_t padPitch = 0.4;
    Float_t sigmaZ = 0.228808;

    Int_t NRows = fParam->GetNRowLow()+fParam->GetNRowUp();

    Float_t rowX[200];
    for( Int_t irow=0; irow<fParam->GetNRowLow(); irow++){
      rowX[irow] = fParam->GetPadRowRadiiLow(irow);
    }     
    for( Int_t irow=0; irow<fParam->GetNRowUp(); irow++){
      rowX[fParam->GetNRowLow()+irow] = fParam->GetPadRowRadiiUp(irow);
    }      
    AliHLTTPCCAParam param;
    param.Initialize( iSlice, NRows, rowX, alpha, dalpha,
		      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, bz );
    param.YErrorCorrection() = 1.;//.33;
    param.ZErrorCorrection() = 1.;//.33;
    param.MaxTrackMatchDRow() = 5;
    param.TrackConnectionFactor() = 3.5;
    fHLTTracker->Slices()[iSlice].Initialize( param ); 
  }
}



Int_t AliTPCtrackerCA::LoadClusters (TTree * tree)
{ 
  fNClusters = 0;
  if( fClusters ) delete[] fClusters;

  fHLTTracker->StartEvent();
  if( fDoHLTPerformance ) fHLTPerformance->StartEvent();

  if( !fParam ) return 1;

  // load mc tracks
  while( fDoHLTPerformance ){
    if( !gAlice ) break;
    AliRunLoader *rl = AliRunLoader::GetRunLoader(); 
    if( !rl ) break;
    rl->LoadKinematics();
    AliStack *stack = rl->Stack();
    if( !stack ) break;

    fHLTPerformance->SetNMCTracks( stack->GetNtrack() );
    
    for( Int_t itr=0; itr<stack->GetNtrack(); itr++ ){
      TParticle *part = stack->Particle(itr);
      fHLTPerformance->ReadMCTrack( itr, part );
    }

    { // check for MC tracks at the TPC entrance
      Bool_t *isTPC = 0;
      isTPC = new Bool_t [stack->GetNtrack()];
      for( Int_t i=0; i<stack->GetNtrack(); i++ ) isTPC[i] = 0;
      rl->LoadTrackRefs();
      TTree *TR = rl->TreeTR();
      if( !TR ) break;
      TBranch *branch=TR->GetBranch("TrackReferences");
      if (!branch ) break;	
      TClonesArray tpcdummy("AliTrackReference",1000), *tpcRefs=&tpcdummy;
      branch->SetAddress(&tpcRefs);
      Int_t nr=(Int_t)TR->GetEntries();
      for (Int_t r=0; r<nr; r++) {
	TR->GetEvent(r);
	Int_t nref = tpcRefs->GetEntriesFast();
	if (!nref) continue;
	AliTrackReference *tpcRef= 0x0;	 
	for (Int_t iref=0; iref<nref; ++iref) {
	  tpcRef = (AliTrackReference*)tpcRefs->UncheckedAt(iref);
	  if (tpcRef->DetectorId() == AliTrackReference::kTPC) break;
	  tpcRef = 0x0;
	}
	if (!tpcRef) continue;

	if( isTPC[tpcRef->Label()] ) continue;	

	fHLTPerformance->ReadMCTPCTrack(tpcRef->Label(),
					tpcRef->X(),tpcRef->Y(),tpcRef->Z(),
					tpcRef->Px(),tpcRef->Py(),tpcRef->Pz() );
	isTPC[tpcRef->Label()] = 1;
	tpcRefs->Clear();
      }	
      if( isTPC ) delete[] isTPC;
    }

    while( fDoHLTPerformanceClusters ){
      AliTPCLoader *tpcl = (AliTPCLoader*) rl->GetDetectorLoader("TPC");
      if( !tpcl ) break;
      if( tpcl->TreeH() == 0x0 ){
	if( tpcl->LoadHits() ) break;
      }
      if( tpcl->TreeH() == 0x0 ) break;
      
      AliTPC *tpc = (AliTPC*) gAlice->GetDetector("TPC");
      Int_t nEnt=(Int_t)tpcl->TreeH()->GetEntries();
      Int_t nPoints = 0;
      for (Int_t iEnt=0; iEnt<nEnt; iEnt++) {    
	tpc->ResetHits();
	tpcl->TreeH()->GetEvent(iEnt);
	AliTPChit *phit = (AliTPChit*)tpc->FirstHit(-1);
	for ( ; phit; phit=(AliTPChit*)tpc->NextHit() ) nPoints++;
      }
      fHLTPerformance->SetNMCPoints( nPoints );

      for (Int_t iEnt=0; iEnt<nEnt; iEnt++) {    
	tpc->ResetHits();
	tpcl->TreeH()->GetEvent(iEnt);
	AliTPChit *phit = (AliTPChit*)tpc->FirstHit(-1);
	for ( ; phit; phit=(AliTPChit*)tpc->NextHit() ){
	  fHLTPerformance->ReadMCPoint( phit->GetTrack(),phit->X(), phit->Y(),phit->Z(),phit->Time(), phit->fSector%36);
	}      
      }
      break;
    }
    break;
  }
  
  TBranch * br = tree->GetBranch("Segment");
  if( !br ) return 1;

  AliTPCClustersRow *clrow = new AliTPCClustersRow;
  clrow->SetClass("AliTPCclusterMI");
  clrow->SetArray(0);
  clrow->GetArray()->ExpandCreateFast(10000);
  
  br->SetAddress(&clrow);
  
  //
  Int_t NEnt=Int_t(tree->GetEntries());

  fNClusters = 0;
  for (Int_t i=0; i<NEnt; i++) {
    br->GetEntry(i);
    Int_t sec,row;
    fParam->AdjustSectorRow(clrow->GetID(),sec,row);
    fNClusters += clrow->GetArray()->GetEntriesFast();
  }

  fClusters = new AliTPCclusterMI [fNClusters];
  fHLTTracker->SetNHits( fNClusters );
  if( fDoHLTPerformance ) fHLTPerformance->SetNHits( fNClusters );
  int ind=0;
  for (Int_t i=0; i<NEnt; i++) {
    br->GetEntry(i);
    Int_t sec,row;
    fParam->AdjustSectorRow(clrow->GetID(),sec,row);
    int NClu = clrow->GetArray()->GetEntriesFast();
    Float_t x = fParam->GetPadRowRadii(sec,row);
    for (Int_t icl=0; icl<NClu; icl++){
      Int_t lab0 = -1;
      Int_t lab1 = -1;
      Int_t lab2 = -1;
      AliTPCclusterMI* cluster = (AliTPCclusterMI*)(clrow->GetArray()->At(icl));
      if( !cluster ) continue;
      lab0 = cluster->GetLabel(0);
      lab1 = cluster->GetLabel(1);
      lab2 = cluster->GetLabel(2);

      AliTPCTransform *transform = AliTPCcalibDB::Instance()->GetTransform() ;
      if (!transform) {
	AliFatal("Tranformations not in calibDB");
      }
      Double_t xx[3]={cluster->GetRow(),cluster->GetPad(),cluster->GetTimeBin()};
      Int_t id[1]={cluster->GetDetector()};
      transform->Transform(xx,id,0,1);  
      //if (!AliTPCReconstructor::GetRecoParam()->GetBYMirror()){
      if (cluster->GetDetector()%36>17){
	xx[1]*=-1;
      }
      //}

      cluster->SetX(xx[0]);
      cluster->SetY(xx[1]);
      cluster->SetZ(xx[2]);

      TGeoHMatrix  *mat = fParam->GetClusterMatrix(cluster->GetDetector());
      Double_t pos[3]= {cluster->GetX(),cluster->GetY(),cluster->GetZ()};
      Double_t posC[3]={cluster->GetX(),cluster->GetY(),cluster->GetZ()};
      if (mat) mat->LocalToMaster(pos,posC);
      else{
	// chack Loading of Geo matrices from GeoManager - TEMPORARY FIX
      }
      cluster->SetX(posC[0]);
      cluster->SetY(posC[1]);
      cluster->SetZ(posC[2]);

      Float_t y = cluster->GetY();
      Float_t z = cluster->GetZ();        

      if( sec>=36 ){
	sec = sec - 36;
	row = row + fParam->GetNRowLow(); 
      }
      
      Int_t index = ind++;
      fClusters[index] = *cluster;
      fHLTTracker->ReadHit( x, y, z, 
			    TMath::Sqrt(cluster->GetSigmaY2()), TMath::Sqrt(cluster->GetSigmaZ2()),			    
			    cluster->GetMax(), index, sec, row );
      if( fDoHLTPerformance ) fHLTPerformance->ReadHitLabel(index, lab0, lab1, lab2 );
    }
  }
  delete clrow;
  return 0;
}

AliCluster * AliTPCtrackerCA::GetCluster(Int_t index) const
{
  return &(fClusters[index]);
}

Int_t AliTPCtrackerCA::Clusters2Tracks( AliESDEvent *event )
{
  //cout<<"Start of AliTPCtrackerCA"<<endl;

  fHLTTracker->FindTracks();
  if( fDoHLTPerformance ) fHLTPerformance->Performance();

  if( 0 ) {// Write Event    
    if( fStatNEvents == 0 ){
      fstream geo;
      geo.open("CAEvents/settings.dat", ios::out);
      if( geo.is_open() ){
	fHLTTracker->WriteSettings(geo);	
      }
      geo.close();
    }

    fstream eventStream;
    char name[255];
    sprintf( name,"CAEvents/%i.event.dat",fStatNEvents ); 
    eventStream.open(name, ios::out);
    if( eventStream.is_open() ){
      fHLTTracker->WriteEvent(eventStream);	
      fstream tracks;
      sprintf( name,"CAEvents/%i.tracks.dat",fStatNEvents ); 
      tracks.open(name, ios::out);
      fHLTTracker->WriteTracks(tracks);	
    }
    eventStream.close();   
    if( fDoHLTPerformance ){
      fstream mcevent, mcpoints;
      char mcname[255];
      sprintf( mcname,"CAEvents/%i.mcevent.dat",fStatNEvents ); 
      mcevent.open(mcname, ios::out);
      if( mcevent.is_open() ){      
	fHLTPerformance->WriteMCEvent(mcevent);
      }
      if(fDoHLTPerformanceClusters ){
	sprintf( mcname,"CAEvents/%i.mcpoints.dat",fStatNEvents ); 
	mcpoints.open(mcname, ios::out);
	if( mcpoints.is_open() ){      
	  fHLTPerformance->WriteMCPoints(mcpoints);
	}
	mcpoints.close();
      }
      mcevent.close();   
    }
  }
  fStatNEvents++;

  if( event ){
   
    Float_t bz = fHLTTracker->Slices()[0].Param().Bz();

    for( Int_t itr=0; itr<fHLTTracker->NTracks(); itr++ ){
      AliTPCtrack tTPC;
      AliHLTTPCCAGBTrack &tCA = fHLTTracker->Tracks()[itr];
      AliHLTTPCCATrackParam &par = tCA.Param();	
      AliHLTTPCCATrackConvertor::GetExtParam( par, tTPC, tCA.Alpha(), bz );
      tTPC.SetMass(0.13957);
      tTPC.SetdEdx( tCA.DeDx() );
      if( TMath::Abs(tTPC.GetSigned1Pt())>1./0.1 ) continue;
      int nhits = tCA.NHits();
      if( nhits>kMaxRow ) nhits = kMaxRow;
      tTPC.SetNumberOfClusters(nhits);      
      for( Int_t ih=0; ih<nhits; ih++ ){
	Int_t index = fHLTTracker->TrackHits()[tCA.FirstHitRef()+ih];
	Int_t ext_index = fHLTTracker->Hits()[index].ID();
	tTPC.SetClusterIndex(ih, ext_index);
      }
      CookLabel(&tTPC,0.1);	      
      {
	Double_t xTPC=fParam->GetInnerRadiusLow();
	Double_t dAlpha = fParam->GetInnerAngle()/180.*TMath::Pi();
	if (tTPC.AliExternalTrackParam::PropagateTo(xTPC,bz)) {	  
	  Double_t y=tTPC.GetY();
	  Double_t ymax=xTPC*TMath::Tan(dAlpha/2.); 
	  if (y > ymax) {
	    if (tTPC.Rotate(dAlpha)) tTPC.AliExternalTrackParam::PropagateTo(xTPC,bz);
	  } else if (y <-ymax) {
	    if (tTPC.Rotate(-dAlpha)) tTPC.AliExternalTrackParam::PropagateTo(xTPC,bz);
	  }	    
	}
      }

      AliESDtrack tESD;
      tESD.UpdateTrackParams( &(tTPC),AliESDtrack::kTPCin);
      //tESD.SetStatus( AliESDtrack::kTPCrefit );
      //tESD.SetTPCPoints(tTPC.GetPoints());
      //tESD.myTPC = tTPC;            
      event->AddTrack(&tESD);
    }
  }

  //cout<<"End of AliTPCtrackerCA"<<endl;
  return 0;
}


Int_t AliTPCtrackerCA::RefitInward (AliESDEvent *event)
{ 
  //* back propagation of ESD tracks (not fully functional)

  Float_t bz = fHLTTracker->Slices()[0].Param().Bz();
  Float_t xTPC = fParam->GetInnerRadiusLow();
  Float_t dAlpha = fParam->GetInnerAngle()/180.*TMath::Pi();
  Float_t yMax = xTPC*TMath::Tan(dAlpha/2.); 

  Int_t nentr=event->GetNumberOfTracks();
     
  for (Int_t i=0; i<nentr; i++) {
    AliESDtrack *esd=event->GetTrack(i);
    ULong_t status=esd->GetStatus();
    if (!(status&AliESDtrack::kTPCin)) continue;
    AliHLTTPCCATrackParam t0;
    AliHLTTPCCATrackConvertor::SetExtParam(t0,*esd, bz );
    Float_t alpha = esd->GetAlpha();
    if( t0.TransportToXWithMaterial( xTPC, bz) ){
      if (t0.GetY() > yMax) {
	if (t0.Rotate(dAlpha)){ 
	  alpha+=dAlpha;  
	  t0.TransportToXWithMaterial( xTPC, bz);
	}
      } else if (t0.GetY() <-yMax) {
	if (t0.Rotate(-dAlpha)){
	  alpha+=-dAlpha;
	  t0.TransportToXWithMaterial( xTPC, bz);
	}
      }    
    }
    AliTPCtrack tt(*esd);
    AliHLTTPCCATrackConvertor::GetExtParam(t0,tt,alpha,bz);
    esd->UpdateTrackParams( &tt,AliESDtrack::kTPCrefit); 
  }
  return 0;
}

Int_t AliTPCtrackerCA::PropagateBack(AliESDEvent *)
{ 
  //* not implemented yet
  return 0; 
}

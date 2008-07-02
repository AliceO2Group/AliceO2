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


ClassImp(AliTPCtrackerCA)

AliTPCtrackerCA::AliTPCtrackerCA()
  :AliTracker(),fParam(0), fClusters(0), fNClusters(0), fHLTTracker(0),fHLTPerformance(0),fDoHLTPerformance(0)
{
  //* default constructor
}

AliTPCtrackerCA::AliTPCtrackerCA(const AliTPCtrackerCA &):
  AliTracker(),fParam(0), fClusters(0), fNClusters(0), fHLTTracker(0),fHLTPerformance(0),fDoHLTPerformance(0)
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
  AliTracker(),fParam(par), fClusters(0), fNClusters(0), fHLTTracker(0), fHLTPerformance(0),fDoHLTPerformance(0)
{
  //* constructor

  DoHLTPerformance() = 0;

  fHLTTracker = new AliHLTTPCCAGBTracker;
  fHLTTracker->SetNSlices( fParam->GetNSector()/2 );

  if( fDoHLTPerformance ){
    fHLTPerformance = new AliHLTTPCCAPerformance;
    fHLTPerformance->SetTracker( fHLTTracker );
  }

  for( int iSlice=0; iSlice<fHLTTracker->NSlices(); iSlice++ ){
  
    Double_t bz = AliTracker::GetBz();

    Double_t inRmin = fParam->GetInnerRadiusLow();
    //Double_t inRmax = fParam->GetInnerRadiusUp();
    //Double_t outRmin = fParam->GetOuterRadiusLow(); 
    Double_t outRmax = fParam->GetOuterRadiusUp();
    Double_t plusZmin = 0.0529937; 
    Double_t plusZmax = 249.778; 
    Double_t minusZmin = -249.645; 
    Double_t minusZmax = -0.0799937; 
    Double_t dalpha = 0.349066;
    Double_t alpha = 0.174533 + dalpha*iSlice;
    
    Bool_t zPlus = (iSlice<18 );
    Double_t zMin =  zPlus ?plusZmin :minusZmin;
    Double_t zMax =  zPlus ?plusZmax :minusZmax;
    //TPCZmin = -249.645, ZMax = 249.778    
    //Double_t rMin =  inRmin;
    //Double_t rMax =  outRmax;
        
    Double_t padPitch = 0.4;
    Double_t sigmaZ = 0.228808;

    Int_t NRows = fParam->GetNRowLow()+fParam->GetNRowUp();

    Double_t rowX[200];
    for( Int_t irow=0; irow<fParam->GetNRowLow(); irow++){
      rowX[irow] = fParam->GetPadRowRadiiLow(irow);
    }     
    for( Int_t irow=0; irow<fParam->GetNRowUp(); irow++){
      rowX[fParam->GetNRowLow()+irow] = fParam->GetPadRowRadiiUp(irow);
    }      
    AliHLTTPCCAParam param;
    param.Initialize( iSlice, NRows, rowX, alpha, dalpha,
		      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, bz );
    param.YErrorCorrection() = .33;//1;
    param.ZErrorCorrection() = .33;//2;
    param.MaxTrackMatchDRow() = 5;
    param.TrackConnectionFactor() = 5.;
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
  if( fDoHLTPerformance ){
    if( !gAlice ) return 0;
    AliRunLoader *rl = gAlice->GetRunLoader(); 
    if( !rl ) return 0;
    rl->LoadKinematics();
    AliStack *stack = rl->Stack();
    if( !stack ) return 0 ;

    fHLTPerformance->SetNMCTracks( stack->GetNtrack() );
    
    for( Int_t itr=0; itr<stack->GetNtrack(); itr++ ){
      TParticle *part = stack->Particle(itr);
      fHLTPerformance->ReadMCTrack( itr, part );
    }
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
    Double_t x = fParam->GetPadRowRadii(sec,row);
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

      Double_t y = cluster->GetY();
      Double_t z = cluster->GetZ();        

      if( sec>=36 ){
	sec = sec - 36;
	row = row + fParam->GetNRowLow(); 
      }
      
      Int_t index = ind++;
      fClusters[index] = *cluster;
      fHLTTracker->ReadHit( x, y, z, 
			    TMath::Sqrt(cluster->GetSigmaY2()), TMath::Sqrt(cluster->GetSigmaZ2()), 
			    index, sec, row );
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

  if( event ){
   
    for( Int_t itr=0; itr<fHLTTracker->NTracks(); itr++ ){
      AliTPCtrack tTPC;
      AliHLTTPCCAGBTrack &tCA = fHLTTracker->Tracks()[itr];
      AliHLTTPCCATrackParam &par = tCA.Param();	

      par.GetExtParam( tTPC, tCA.Alpha(), fHLTTracker->Slices()[0].Param().Bz() );
      
      tTPC.SetMass(0.13957);
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
	Double_t xTPC=83.65;
	if (tTPC.AliExternalTrackParam::PropagateTo(xTPC,5)) {	  
	  Double_t y=tTPC.GetY();
	  Double_t ymax=xTPC*TMath::Tan(1.74532920122146606e-01); 
	  if (y > ymax) {
	    if (tTPC.Rotate(2*1.74532920122146606e-01)) tTPC.AliExternalTrackParam::PropagateTo(xTPC,5);
	  } else if (y <-ymax) {
	    if (tTPC.Rotate(-2*1.74532920122146606e-01)) tTPC.AliExternalTrackParam::PropagateTo(xTPC,5);
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


Int_t AliTPCtrackerCA::RefitInward (AliESDEvent *)
{ 
  //* not implemented yet
  return 0; 
}

Int_t AliTPCtrackerCA::PropagateBack(AliESDEvent *)
{ 
  //* not implemented yet
  return 0; 
}

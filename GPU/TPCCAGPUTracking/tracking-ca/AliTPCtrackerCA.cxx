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

#include "AliTPCtrackerCA.h"

#include "TTree.h"
#include "Riostream.h"
//#include "AliCluster.h"
#include "AliTPCClustersRow.h"
#include "AliTPCParam.h"
#include "AliTPCClusterParam.h"

#include "AliRun.h"
#include "AliRunLoader.h"
#include "AliStack.h"

#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCAGBHit.h"
#include "AliHLTTPCCAGBTrack.h"
#include "AliHLTTPCCAPerformance.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackConvertor.h"
#include "AliHLTTPCCATracker.h"

#include "TMath.h"
#include "AliTPCLoader.h"
#include "AliTPC.h"
#include "AliTPCclusterMI.h"
#include "AliTPCTransform.h"
#include "AliTPCcalibDB.h"
#include "AliTPCtrack.h"
#include "AliTPCseed.h"
#include "AliESDtrack.h"
#include "AliESDEvent.h"
#include "AliTrackReference.h"
#include "TStopwatch.h"
#include "AliTPCReconstructor.h"

//#include <fstream.h>

ClassImp( AliTPCtrackerCA )

AliTPCtrackerCA::AliTPCtrackerCA()
    : AliTracker(), fkParam( 0 ), fClusters( 0 ), fNClusters( 0 ), fHLTTracker( 0 ), fDoHLTPerformance( 0 ), fDoHLTPerformanceClusters( 0 ), fStatNEvents( 0 )
{
  //* default constructor
}

AliTPCtrackerCA::AliTPCtrackerCA( const AliTPCtrackerCA & ):
    AliTracker(), fkParam( 0 ), fClusters( 0 ), fNClusters( 0 ), fHLTTracker( 0 ), fDoHLTPerformance( 0 ), fDoHLTPerformanceClusters( 0 ), fStatNEvents( 0 )
{
  //* dummy
}

const AliTPCtrackerCA & AliTPCtrackerCA::operator=( const AliTPCtrackerCA& ) const
{
  //* dummy
  return *this;
}


AliTPCtrackerCA::~AliTPCtrackerCA()
{
  //* destructor
  if ( fClusters ) delete[] fClusters;
  if ( fHLTTracker ) delete fHLTTracker;
}

//#include "AliHLTTPCCADisplay.h"

AliTPCtrackerCA::AliTPCtrackerCA( const AliTPCParam *par ):
    AliTracker(), fkParam( par ), fClusters( 0 ), fNClusters( 0 ), fHLTTracker( 0 ), fDoHLTPerformance( 0 ), fDoHLTPerformanceClusters( 0 ), fStatNEvents( 0 )
{
  //* constructor

  fDoHLTPerformance = 0;
  fDoHLTPerformanceClusters = 0;

  fHLTTracker = new AliHLTTPCCAGBTracker;
  fHLTTracker->SetNSlices( fkParam->GetNSector() / 2 );

  if ( fDoHLTPerformance ) {
    AliHLTTPCCAPerformance::Instance().SetTracker( fHLTTracker );
  }

  for ( int iSlice = 0; iSlice < fHLTTracker->NSlices(); iSlice++ ) {

    const double kCLight = 0.000299792458;

    float bz = AliTracker::GetBz() * kCLight;

    float inRmin = fkParam->GetInnerRadiusLow();
    //float inRmax = fkParam->GetInnerRadiusUp();
    //float outRmin = fkParam->GetOuterRadiusLow();
    float outRmax = fkParam->GetOuterRadiusUp();
    float plusZmin = 0.0529937;
    float plusZmax = 249.778;
    float minusZmin = -249.645;
    float minusZmax = -0.0799937;
    float dalpha = 0.349066;
    float alpha = 0.174533 + dalpha * iSlice;

    bool zPlus = ( iSlice < 18 );
    float zMin =  zPlus ? plusZmin : minusZmin;
    float zMax =  zPlus ? plusZmax : minusZmax;
    //TPCZmin = -249.645, ZMax = 249.778
    //float rMin =  inRmin;
    //float rMax =  outRmax;

    float padPitch = 0.4;
    float sigmaZ = 0.228808;

    int nRows = fkParam->GetNRowLow() + fkParam->GetNRowUp();
    float rowX[200];
    for ( int irow = 0; irow < fkParam->GetNRowLow(); irow++ ) {
      rowX[irow] = fkParam->GetPadRowRadiiLow( irow );
    }
    for ( int irow = 0; irow < fkParam->GetNRowUp(); irow++ ) {
      rowX[fkParam->GetNRowLow()+irow] = fkParam->GetPadRowRadiiUp( irow );
    }
    AliHLTTPCCAParam param;
    param.Initialize( iSlice, nRows, rowX, alpha, dalpha,
                      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, bz );
    param.SetHitPickUpFactor( 1. );
    param.SetMaxTrackMatchDRow( 5 );
    param.SetTrackConnectionFactor( 3.5 );

    AliTPCClusterParam * clparam = AliTPCcalibDB::Instance()->GetClusterParam();
    for ( int iRow = 0; iRow < nRows; iRow++ ) {
      int    type = ( iRow < 63 ) ? 0 : ( ( iRow > 126 ) ? 1 : 2 );
      for ( int iyz = 0; iyz < 2; iyz++ ) {
        for ( int k = 0; k < 7; k++ ) {
          //std::cout<<param.fParamS0Par[iyz][type][k]<<" "<<clparam->fParamS0Par[iyz][type][k] - param.fParamS0Par[iyz][type][k]<<std::endl;
          param.SetParamS0Par( iyz, type, k, clparam->fParamS0Par[iyz][type][k] );
        }
      }
    }
    fHLTTracker->Slices()[iSlice].Initialize( param );
  }
}



int AliTPCtrackerCA::LoadClusters ( TTree * fromTree )
{
  // load clusters to the local arrays
  fNClusters = 0;
  if ( fClusters ) delete[] fClusters;

  fHLTTracker->StartEvent();
  if ( fDoHLTPerformance ) AliHLTTPCCAPerformance::Instance().StartEvent();

  if ( !fkParam ) return 1;

  // load mc tracks
  while ( fDoHLTPerformance ) {
    if ( !gAlice ) break;
    AliRunLoader *rl = AliRunLoader::Instance();//gAlice->GetRunLoader();
    if ( !rl ) break;
    rl->LoadKinematics();
    AliStack *stack = rl->Stack();
    if ( !stack ) break;

    AliHLTTPCCAPerformance::Instance().SetNMCTracks( stack->GetNtrack() );

    for ( int itr = 0; itr < stack->GetNtrack(); itr++ ) {
      TParticle *part = stack->Particle( itr );
      AliHLTTPCCAPerformance::Instance().ReadMCTrack( itr, part );
    }

    { // check for MC tracks at the TPC entrance

      bool *isTPC = 0;
      isTPC = new bool [stack->GetNtrack()];
      for ( int i = 0; i < stack->GetNtrack(); i++ ) isTPC[i] = 0;
      rl->LoadTrackRefs();
      TTree *mcTree = rl->TreeTR();
      if ( !mcTree ) break;
      TBranch *branch = mcTree->GetBranch( "TrackReferences" );
      if ( !branch ) break;
      TClonesArray tpcdummy( "AliTrackReference", 1000 ), *tpcRefs = &tpcdummy;
      branch->SetAddress( &tpcRefs );
      int nr = ( int )mcTree->GetEntries();
      for ( int r = 0; r < nr; r++ ) {
        mcTree->GetEvent( r );
        int nref = tpcRefs->GetEntriesFast();
        if ( !nref ) continue;
        AliTrackReference *tpcRef = 0x0;
        for ( int iref = 0; iref < nref; ++iref ) {
          tpcRef = ( AliTrackReference* )tpcRefs->UncheckedAt( iref );
          if ( tpcRef->DetectorId() == AliTrackReference::kTPC ) break;
          tpcRef = 0x0;
        }
        if ( !tpcRef ) continue;

        if ( isTPC[tpcRef->Label()] ) continue;

        AliHLTTPCCAPerformance::Instance().ReadMCTPCTrack( tpcRef->Label(),
            tpcRef->X(), tpcRef->Y(), tpcRef->Z(),
            tpcRef->Px(), tpcRef->Py(), tpcRef->Pz() );
        isTPC[tpcRef->Label()] = 1;
        tpcRefs->Clear();
      }
      if ( isTPC ) delete[] isTPC;
    }

    while ( fDoHLTPerformanceClusters ) {
      AliTPCLoader *tpcl = ( AliTPCLoader* ) rl->GetDetectorLoader( "TPC" );
      if ( !tpcl ) break;
      if ( tpcl->TreeH() == 0x0 ) {
        if ( tpcl->LoadHits() ) break;
      }
      if ( tpcl->TreeH() == 0x0 ) break;

      AliTPC *tpc = ( AliTPC* ) gAlice->GetDetector( "TPC" );
      int nEnt = ( int )tpcl->TreeH()->GetEntries();
      int nPoints = 0;
      for ( int iEnt = 0; iEnt < nEnt; iEnt++ ) {
        tpc->ResetHits();
        tpcl->TreeH()->GetEvent( iEnt );
        AliTPChit *phit = ( AliTPChit* )tpc->FirstHit( -1 );
        for ( ; phit; phit = ( AliTPChit* )tpc->NextHit() ) nPoints++;
      }
      AliHLTTPCCAPerformance::Instance().SetNMCPoints( nPoints );

      for ( int iEnt = 0; iEnt < nEnt; iEnt++ ) {
        tpc->ResetHits();
        tpcl->TreeH()->GetEvent( iEnt );
        AliTPChit *phit = ( AliTPChit* )tpc->FirstHit( -1 );
        for ( ; phit; phit = ( AliTPChit* )tpc->NextHit() ) {
          AliHLTTPCCAPerformance::Instance().ReadMCPoint( phit->GetTrack(), phit->X(), phit->Y(), phit->Z(), phit->Time(), phit->fSector % 36 );
        }
      }
      break;
    }
    break;
  }

  TBranch * br = fromTree->GetBranch( "Segment" );
  if ( !br ) return 1;

  AliTPCClustersRow *clrow = new AliTPCClustersRow;
  clrow->SetClass( "AliTPCclusterMI" );
  clrow->SetArray( 0 );
  clrow->GetArray()->ExpandCreateFast( 10000 );

  br->SetAddress( &clrow );

  //
  int nEnt = int( fromTree->GetEntries() );

  fNClusters = 0;
  for ( int i = 0; i < nEnt; i++ ) {
    br->GetEntry( i );
    int sec, row;
    fkParam->AdjustSectorRow( clrow->GetID(), sec, row );
    fNClusters += clrow->GetArray()->GetEntriesFast();
  }

  fClusters = new AliTPCclusterMI [fNClusters];
  fHLTTracker->SetNHits( fNClusters );
  if ( fDoHLTPerformance ) AliHLTTPCCAPerformance::Instance().SetNHits( fNClusters );
  int ind = 0;
  for ( int i = 0; i < nEnt; i++ ) {
    br->GetEntry( i );
    int sec, row;
    fkParam->AdjustSectorRow( clrow->GetID(), sec, row );
    int nClu = clrow->GetArray()->GetEntriesFast();
    float x = fkParam->GetPadRowRadii( sec, row );
    for ( int icl = 0; icl < nClu; icl++ ) {
      int lab0 = -1;
      int lab1 = -1;
      int lab2 = -1;
      AliTPCclusterMI* cluster = ( AliTPCclusterMI* )( clrow->GetArray()->At( icl ) );
      if ( !cluster ) continue;
      lab0 = cluster->GetLabel( 0 );
      lab1 = cluster->GetLabel( 1 );
      lab2 = cluster->GetLabel( 2 );

      AliTPCTransform *transform = AliTPCcalibDB::Instance()->GetTransform() ;
      if ( !transform ) {
        AliFatal( "Tranformations not in calibDB" );
      }
      double xx[3] = {cluster->GetRow(), cluster->GetPad(), cluster->GetTimeBin()};
      int id[1] = {cluster->GetDetector()};
      transform->Transform( xx, id, 0, 1 );
      //if (!AliTPCReconstructor::GetRecoParam()->GetBYMirror()){
      //if (cluster->GetDetector()%36>17){
      //xx[1]*=-1;
      //}
      //}

      cluster->SetX( xx[0] );
      cluster->SetY( xx[1] );
      cluster->SetZ( xx[2] );

      TGeoHMatrix  *mat = fkParam->GetClusterMatrix( cluster->GetDetector() );
      double pos[3] = {cluster->GetX(), cluster->GetY(), cluster->GetZ()};
      double posC[3] = {cluster->GetX(), cluster->GetY(), cluster->GetZ()};
      if ( mat ) mat->LocalToMaster( pos, posC );
      else {
        // chack Loading of Geo matrices from GeoManager - TEMPORARY FIX
      }
      cluster->SetX( posC[0] );
      cluster->SetY( posC[1] );
      cluster->SetZ( posC[2] );

      x = cluster->GetX();
      float y = cluster->GetY();
      float z = cluster->GetZ();

      if ( sec >= 36 ) {
        sec = sec - 36;
        row = row + fkParam->GetNRowLow();
      }

      int index = ind++;
      fClusters[index] = *cluster;
      fHLTTracker->ReadHit( x, y, z,
                            TMath::Sqrt( cluster->GetSigmaY2() ), TMath::Sqrt( cluster->GetSigmaZ2() ),
                            cluster->GetMax(), index, sec, row );
      if ( fDoHLTPerformance ) AliHLTTPCCAPerformance::Instance().ReadHitLabel( index, lab0, lab1, lab2 );
    }
  }
  delete clrow;
  return 0;
}

AliCluster * AliTPCtrackerCA::GetCluster( int index ) const
{
  return &( fClusters[index] );
}

int AliTPCtrackerCA::Clusters2Tracks( AliESDEvent *event )
{
  // reconstruction
  //cout<<"Start of AliTPCtrackerCA"<<endl;
  TStopwatch timer;

  fHLTTracker->FindTracks();
  //cout<<"Do performance.."<<endl;
  if ( fDoHLTPerformance ) AliHLTTPCCAPerformance::Instance().Performance();

  if ( 0 ) {// Write Event
    if ( fStatNEvents == 0 ) {
      fstream geo;
      geo.open( "CAEvents/settings.dat", ios::out );
      if ( geo.is_open() ) {
        fHLTTracker->WriteSettings( geo );
      }
      geo.close();
    }

    fstream hits;
    char name[255];
    sprintf( name, "CAEvents/%i.event.dat", fStatNEvents );
    hits.open( name, ios::out );
    if ( hits.is_open() ) {
      fHLTTracker->WriteEvent( hits );
      fstream tracks;
      sprintf( name, "CAEvents/%i.tracks.dat", fStatNEvents );
      tracks.open( name, ios::out );
      fHLTTracker->WriteTracks( tracks );
    }
    hits.close();
    if ( fDoHLTPerformance ) {
      fstream mcevent, mcpoints;
      char mcname[255];
      sprintf( mcname, "CAEvents/%i.mcevent.dat", fStatNEvents );
      mcevent.open( mcname, ios::out );
      if ( mcevent.is_open() ) {
        AliHLTTPCCAPerformance::Instance().WriteMCEvent( mcevent );
      }
      if ( 1 && fDoHLTPerformanceClusters ) {
        sprintf( mcname, "CAEvents/%i.mcpoints.dat", fStatNEvents );
        mcpoints.open( mcname, ios::out );
        if ( mcpoints.is_open() ) {
          AliHLTTPCCAPerformance::Instance().WriteMCPoints( mcpoints );
        }
        mcpoints.close();
      }
      mcevent.close();
    }
  }
  fStatNEvents++;

  if ( event ) {

    for ( int itr = 0; itr < fHLTTracker->NTracks(); itr++ ) {
      //AliTPCtrack tTPC;
      AliTPCseed tTPC;
      AliHLTTPCCAGBTrack &tCA = fHLTTracker->Tracks()[itr];
      AliHLTTPCCATrackParam par = tCA.Param();
      AliHLTTPCCATrackConvertor::GetExtParam( par, tTPC, tCA.Alpha() );
      tTPC.SetMass( 0.13957 );
      tTPC.SetdEdx( tCA.DeDx() );
      if ( TMath::Abs( tTPC.GetSigned1Pt() ) > 1. / 0.02 ) continue;
      int nhits = tCA.NHits();
      int firstHit = 0;
      if ( nhits > 160 ) {
        firstHit = nhits - 160;
        nhits = 160;
      }

      tTPC.SetNumberOfClusters( nhits );

      float alpha = tCA.Alpha();
      AliHLTTPCCATrackParam t0 = par;
      for ( int ih = 0; ih < nhits; ih++ ) {
        int index = fHLTTracker->TrackHits()[tCA.FirstHitRef()+firstHit+ih];
        const AliHLTTPCCAGBHit &h = fHLTTracker->Hits()[index];
        int extIndex = h.ID();
        tTPC.SetClusterIndex( ih, extIndex );

        AliTPCclusterMI *c = &( fClusters[extIndex] );
        tTPC.SetClusterPointer( h.IRow(), c );
        AliTPCTrackerPoint &point = *( tTPC.GetTrackPoint( h.IRow() ) );
        {
          int iSlice = h.ISlice();
          AliHLTTPCCATracker &slice = fHLTTracker->Slices()[iSlice];
          if ( slice.Param().Alpha() != alpha ) {
            if ( ! t0.Rotate(  slice.Param().Alpha() - alpha, .999 ) ) continue;
            alpha = slice.Param().Alpha();
          }
          float x = slice.Row( h.IRow() ).X();
          if ( !t0.TransportToX( x, fHLTTracker->Slices()[0].Param().GetBz( t0 ), .999 ) ) continue;
          float sy2, sz2;
          slice.GetErrors2( h.IRow(), t0, sy2, sz2 );
          point.SetSigmaY( c->GetSigmaY2() / sy2 );
          point.SetSigmaZ( c->GetSigmaZ2() / sz2 );
          point.SetAngleY( TMath::Abs( t0.GetSinPhi() / t0.GetCosPhi() ) );
          point.SetAngleZ( TMath::Abs( t0.GetDzDs() ) );
        }
      }
      tTPC.CookdEdx( 0.02, 0.6 );

      CookLabel( &tTPC, 0.1 );

      if ( 1 ) { // correction like in off-line --- Adding systematic error

        const double *param = AliTPCReconstructor::GetRecoParam()->GetSystematicError();
        double covar[15];
        for ( int i = 0; i < 15; i++ ) covar[i] = 0;
        covar[0] = param[0] * param[0];
        covar[2] = param[1] * param[1];
        covar[5] = param[2] * param[2];
        covar[9] = param[3] * param[3];
        double facC =  AliTracker::GetBz() * kB2C;
        covar[14] = param[4] * param[4] * facC * facC;
        tTPC.AddCovariance( covar );
      }

      AliESDtrack tESD;
      tESD.UpdateTrackParams( &( tTPC ), AliESDtrack::kTPCin );
      //tESD.SetStatus( AliESDtrack::kTPCrefit );
      //tESD.SetTPCPoints(tTPC.GetPoints());
      int   ndedx = tTPC.GetNCDEDX( 0 );
      float sdedx = tTPC.GetSDEDX( 0 );
      float dedx  = tTPC.GetdEdx();
      tESD.SetTPCsignal( dedx, sdedx, ndedx );
      //tESD.myTPC = tTPC;

      event->AddTrack( &tESD );
    }
  }
  timer.Stop();
  static double time = 0, time1 = 0;
  static int ncalls = 0;
  time += timer.CpuTime();
  time1 += timer.RealTime();
  ncalls++;
  //cout<<"\n\nCA tracker speed: cpu = "<<time/ncalls*1.e3<<" [ms/ev], real = "<<time1/ncalls*1.e3<<" [ms/ev], n calls = "<<ncalls<<endl<<endl;

  //cout<<"End of AliTPCtrackerCA"<<endl;
  return 0;
}


int AliTPCtrackerCA::RefitInward ( AliESDEvent *event )
{
  //* forward propagation of ESD tracks

  //float bz = fHLTTracker->Slices()[0].Param().Bz();
  float xTPC = fkParam->GetInnerRadiusLow();
  float dAlpha = fkParam->GetInnerAngle() / 180.*TMath::Pi();
  float yMax = xTPC * TMath::Tan( dAlpha / 2. );

  int nentr = event->GetNumberOfTracks();

  for ( int itr = 0; itr < nentr; itr++ ) {
    AliESDtrack *esd = event->GetTrack( itr );
    ULong_t status = esd->GetStatus();
    if ( !( status&AliESDtrack::kTPCin ) ) continue;
    AliHLTTPCCATrackParam t0;
    AliHLTTPCCATrackConvertor::SetExtParam( t0, *esd );
    AliHLTTPCCATrackParam t = t0;
    float alpha = esd->GetAlpha();
    //float dEdX=0;
    int hits[1000];
    int nHits = esd->GetTPCclusters( hits );

    // convert clluster indices to AliHLTTPCCAGBHit indices

    for ( int i = 0; i < nHits; i++ ) hits[i] = fHLTTracker->Ext2IntHitID( hits[i] );

    bool ok = fHLTTracker->FitTrack( t, t0, alpha, hits, nHits, 0 );
    if ( ok &&  nHits > 15 ) {
      if ( t.TransportToXWithMaterial( xTPC, fHLTTracker->Slices()[0].Param().GetBz( t ) ) ) {
        if ( t.GetY() > yMax ) {
          if ( t.Rotate( dAlpha ) ) {
            alpha += dAlpha;
            t.TransportToXWithMaterial( xTPC, fHLTTracker->Slices()[0].Param().GetBz( t ) );
          }
        } else if ( t.GetY() < -yMax ) {
          if ( t.Rotate( -dAlpha ) ) {
            alpha += -dAlpha;
            t.TransportToXWithMaterial( xTPC, fHLTTracker->Slices()[0].Param().GetBz( t ) );
          }
        }
      }

      AliTPCtrack tt( *esd );
      if ( AliHLTTPCCATrackConvertor::GetExtParam( t, tt, alpha ) ) {
        if ( t.X() > 50 ) esd->UpdateTrackParams( &tt, AliESDtrack::kTPCrefit );
      }
    }
  }
  return 0;
}

int AliTPCtrackerCA::PropagateBack( AliESDEvent *event )
{

  //* backward propagation of ESD tracks

  //float bz = fHLTTracker->Slices()[0].Param().Bz();
  int nentr = event->GetNumberOfTracks();

  for ( int itr = 0; itr < nentr; itr++ ) {

    AliESDtrack *esd = event->GetTrack( itr );
    ULong_t status = esd->GetStatus();
    if ( !( status&AliESDtrack::kTPCin ) ) continue;

    AliHLTTPCCATrackParam t0;
    AliHLTTPCCATrackConvertor::SetExtParam( t0, *esd  );
    AliHLTTPCCATrackParam t = t0;
    float alpha = esd->GetAlpha();
    //float dEdX=0;
    int hits[1000];
    int nHits = esd->GetTPCclusters( hits );

    // convert clluster indices to AliHLTTPCCAGBHit indices

    for ( int i = 0; i < nHits; i++ ) hits[i] = fHLTTracker->Ext2IntHitID( hits[i] );

    bool ok = fHLTTracker->FitTrack( t, t0, alpha, hits, nHits, 1 );
    if ( ok &&  nHits > 15 ) {
      AliTPCtrack tt( *esd );
      if ( AliHLTTPCCATrackConvertor::GetExtParam( t, tt, alpha ) ) {
        if ( t.X() > 50 ) esd->UpdateTrackParams( &tt, AliESDtrack::kTPCout );
      }
    }
  }
  return 0;
}

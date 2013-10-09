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

#include "AliHLTTPCCAPerformance.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCGMMergedTrack.h"

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
#include <memory>

//#include <fstream.h>

ClassImp( AliTPCtrackerCA )

AliTPCtrackerCA::AliTPCtrackerCA()
    : AliTracker(), fkParam( 0 ), fClusters( 0 ), fClusterSliceRow( 0 ), fNClusters( 0 ), fDoHLTPerformance( 0 ), fDoHLTPerformanceClusters( 0 ), fStatCPUTime( 0 ), fStatRealTime( 0 ), fStatNEvents( 0 )
{
  //* default constructor
}

AliTPCtrackerCA::~AliTPCtrackerCA()
{
  //* destructor
  delete[] fClusters;
  delete[] fClusterSliceRow;
}

//#include "AliHLTTPCCADisplay.h"

AliTPCtrackerCA::AliTPCtrackerCA( const AliTPCParam *par ):
    AliTracker(), fkParam( par ), fClusters( 0 ), fClusterSliceRow( 0 ), fNClusters( 0 ), fDoHLTPerformance( 0 ), fDoHLTPerformanceClusters( 0 ), fStatCPUTime( 0 ), fStatRealTime( 0 ), fStatNEvents( 0 )
{
  //* constructor

  fDoHLTPerformance = 0;
  fDoHLTPerformanceClusters = 0;

  AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();


  for ( int iSlice = 0; iSlice < hlt.NSlices(); iSlice++ ) {

    float bz = AliTracker::GetBz();

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
    int dimRowX=fkParam->GetNRowLow()+fkParam->GetNRowUp();
    std::auto_ptr<float> rowX(new float[dimRowX]);
    for ( int irow = 0; irow < fkParam->GetNRowLow(); irow++ ) {
      (rowX.get())[irow] = fkParam->GetPadRowRadiiLow( irow );
    }
    for ( int irow = 0; irow < fkParam->GetNRowUp(); irow++ ) {
      (rowX.get())[fkParam->GetNRowLow()+irow] = fkParam->GetPadRowRadiiUp( irow );
    }
    AliHLTTPCCAParam param;
    param.Initialize( iSlice, nRows, rowX.get(), alpha, dalpha,
                      inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, bz );
    param.SetHitPickUpFactor( 1. );
    param.SetMaxTrackMatchDRow( 5 );
    param.SetTrackConnectionFactor( 3.5 );
    param.SetMinNTrackClusters( 30 );
    param.SetMinTrackPt(0.2);
    AliTPCClusterParam * clparam = AliTPCcalibDB::Instance()->GetClusterParam();
    for ( int iRow = 0; iRow < nRows; iRow++ ) {
      int    type = ( iRow < 63 ) ? 0 : ( ( iRow > 126 ) ? 1 : 2 );
      for ( int iyz = 0; iyz < 2; iyz++ ) {
        for ( int k = 0; k < 7; k++ ) {
          //std::cout<<param.fParamS0Par[iyz][type][k]<<" "<<clparam->fParamS0Par[iyz][type][k] - param.fParamS0Par[iyz][type][k]<<std::endl;
#ifndef HAVE_NOT_ALITPCCLUSTERPARAM_r40128
          param.SetParamS0Par( iyz, type, k, clparam->ParamS0Par(iyz, type, k));
#else
          param.SetParamS0Par( iyz, type, k, clparam->fParamS0Par[iyz][type][k] );
#endif //HAVE_NOT_ALITPCCLUSTERPARAM_r40128
        }
      }
    }
    //hlt.SliceTracker( iSlice ).Initialize( param );
	hlt.InitializeSliceParam(iSlice, param);
  }
}



int AliTPCtrackerCA::LoadClusters ( TTree * fromTree )
{
  // load clusters to the local arrays
  fNClusters = 0;
  delete[] fClusters;
  delete[] fClusterSliceRow;

  AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();

  if ( fDoHLTPerformance ) {
    AliHLTTPCCAPerformance::Instance().StartEvent();
    if ( fDoHLTPerformanceClusters ) AliHLTTPCCAPerformance::Instance().SetDoClusterPulls( 1 );
  }

  if ( !fkParam ) return 1;

  // load mc tracks
  while ( fDoHLTPerformance ) {
    if ( !gAlice ) break;
#ifndef HAVE_NOT_ALIRUNLOADER30859
    AliRunLoader *rl = AliRunLoader::Instance();//gAlice->GetRunLoader();
#else
    // the old way before rev 30859
    AliRunLoader *rl = AliRunLoader::GetRunLoader();
#endif
    if ( !rl ) break;
    rl->LoadKinematics();
    AliStack *stack = rl->Stack();
    if ( !stack ) break;
    int nMCtracks = stack->GetNtrack();
    if( nMCtracks<0 || nMCtracks>10000000 ) nMCtracks = 0;

    AliHLTTPCCAPerformance::Instance().SetNMCTracks( nMCtracks );

    for ( int itr = 0; itr < nMCtracks; itr++ ) {
      TParticle *part = stack->Particle( itr );
      AliHLTTPCCAPerformance::Instance().ReadMCTrack( itr, part );
    }

    { // check for MC tracks at the TPC entrance

      rl->LoadTrackRefs();
      TTree *mcTree = rl->TreeTR();
      if ( !mcTree ) break;
      TBranch *branch = mcTree->GetBranch( "TrackReferences" );
      if ( !branch ) break;
      TClonesArray tpcdummy( "AliTrackReference", 1000 ), *tpcRefs = &tpcdummy;
      branch->SetAddress( &tpcRefs );

      bool *isTPC = new bool [nMCtracks];
      for ( int i = 0; i < nMCtracks; i++ ) isTPC[i] = 0;

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
      delete[] isTPC;
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
  fClusterSliceRow = new unsigned int [fNClusters];

  hlt.StartDataReading( fNClusters );

  if ( fDoHLTPerformance ) AliHLTTPCCAPerformance::Instance().SetNHits( fNClusters );

  int ind = 0;
  for ( int i = 0; i < nEnt; i++ ) {
    br->GetEntry( i );
    int sec, row;
    fkParam->AdjustSectorRow( clrow->GetID(), sec, row );
    int nClu = clrow->GetArray()->GetEntriesFast();
    float x = fkParam->GetPadRowRadii( sec, row );

    if ( sec >= 36 ) {
      sec = sec - 36;
      row = row + fkParam->GetNRowLow();
    }

    unsigned int sliceRow = ( sec << 8 ) + row;

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
	break;
      }

      transform->SetCurrentRecoParam((AliTPCRecoParam*)AliTPCReconstructor::GetRecoParam());
      double xx[3] = {cluster->GetRow(), cluster->GetPad(), cluster->GetTimeBin()};
      int id[1] = {cluster->GetDetector()};
      transform->Transform( xx, id, 0, 1 );

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


      int index = ind++;
      fClusters[index] = *cluster;

      fClusterSliceRow[index] = sliceRow;

      hlt.ReadCluster( index, sec, row, x, y, z, cluster->GetQ() );

      if ( fDoHLTPerformance ) AliHLTTPCCAPerformance::Instance().ReadHitLabel( index, lab0, lab1, lab2 );
    }
  }
  delete clrow;

  hlt.FinishDataReading();

  //AliHLTTPCCAPerformance::Instance().SmearClustersMC();

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

  AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();

  hlt.ProcessEvent();

  if ( event ) {  

    for ( int itr = 0; itr < hlt.Merger().NOutputTracks(); itr++ ) {

      AliTPCseed tTPC;

      const AliHLTTPCGMMergedTrack &tCA = hlt.Merger().OutputTracks()[itr];

      AliHLTTPCGMTrackParam par = tCA.GetParam();      
      par.GetExtParam( tTPC, tCA.GetAlpha() );
      tTPC.SetMass( 0.13957 );
      if ( TMath::Abs( tTPC.GetSigned1Pt() ) > 1. / 0.02 ) continue;
      int nhits = tCA.NClusters();
      int firstHit = 0;
      if ( nhits > 160 ) {
        firstHit = nhits - 160;
        nhits = 160;
      }
      tTPC.SetNumberOfClusters( nhits );
      //float alpha = tCA.GetAlpha();
      AliHLTTPCGMTrackParam t0 = par;
      for ( int ih = 0; ih < nhits; ih++ ) {
        int index = hlt.Merger().OutputClusterIds()[ tCA.FirstClusterRef() + firstHit + ih ];
        tTPC.SetClusterIndex( ih, index );
        AliTPCclusterMI *c = &( fClusters[index] );
        //int iSlice = fClusterSliceRow[index] >> 8;
        int row = fClusterSliceRow[index] & 0xff;

        tTPC.SetClusterPointer( row, c );
	/*
        AliTPCTrackerPoint &point = *( tTPC.GetTrackPoint( row ) );
        {
          //AliHLTTPCCATracker &slice = hlt.SliceTracker( iSlice );
          if ( hlt.Param(iSlice).Alpha() != alpha ) {
            if ( ! t0.Rotate(  hlt.Param(iSlice).Alpha() - alpha, .999 ) ) continue;
            alpha = hlt.Param(iSlice).Alpha();
          }
          float x = hlt.Row(iSlice, row).X();
          if ( !t0.TransportToX( x, hlt.Param(iSlice).GetBz( t0 ), .999 ) ) continue;
          float sy2, sz2;
          //slice.GetErrors2( row, t0, sy2, sz2 );
		  hlt.Param(iSlice).GetClusterErrors2( row, t0.GetZ(), t0.SinPhi(), t0.GetCosPhi(), t0.DzDs(), sy2, sz2 );
          point.SetSigmaY( c->GetSigmaY2() / sy2 );
          point.SetSigmaZ( c->GetSigmaZ2() / sz2 );
          point.SetAngleY( TMath::Abs( t0.GetSinPhi() / t0.GetCosPhi() ) );
          point.SetAngleZ( TMath::Abs( t0.GetDzDs() ) );
        }
	*/
      }
      //tTPC.CookdEdx( 0.02, 0.6 );

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

  fStatCPUTime += timer.CpuTime();
  fStatRealTime += timer.RealTime();

  //cout << "\n\nCA tracker speed: cpu = " << fStatCPUTime / ( fStatNEvents + 1 )*1.e3 << " [ms/ev], real = " << fStatRealTime / ( fStatNEvents + 1 )*1.e3 << " [ms/ev], n calls = " << ( fStatNEvents + 1 ) << endl << endl;


  //cout<<"Do performance.."<<endl;

  if ( fDoHLTPerformance ) AliHLTTPCCAPerformance::Instance().Performance();

  if ( 0 ) {// Write Event
    if ( fStatNEvents == 0 ) {
      fstream geo;
      geo.open( "CAEvents/settings.dat", ios::out );
      if ( geo.is_open() ) {
        hlt.WriteSettings( geo );
      }
      geo.close();
    }

    fstream hits;
    char name[255];
    sprintf( name, "CAEvents/%i.event.dat", fStatNEvents );
    hits.open( name, ios::out );
    if ( hits.is_open() ) {
      hlt.WriteEvent( hits );
      fstream tracks;
      sprintf( name, "CAEvents/%i.tracks.dat", fStatNEvents );
      tracks.open( name, ios::out );
      hlt.WriteTracks( tracks );
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


  //cout<<"End of AliTPCtrackerCA"<<endl;
  return 0;
}


int AliTPCtrackerCA::RefitInward ( AliESDEvent * /*event*/ )
{
  //* forward propagation of ESD tracks
#ifdef XXX
  float xTPC = fkParam->GetInnerRadiusLow();
  float dAlpha = fkParam->GetInnerAngle() / 180.*TMath::Pi();
  float yMax = xTPC * TMath::Tan( dAlpha / 2. );

  AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();

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
    AliHLTTPCCAMerger::AliHLTTPCCAClusterInfo infos[500];
    int hits[500], hits1[500];
    int nHits = esd->GetTPCclusters( hits );
    for ( int i = 0; i < nHits; i++ ) {
      hits1[i] = i;
      int index = hits[i];
      infos[i].SetISlice( fClusterSliceRow[index] >> 8 );
      int row = fClusterSliceRow[index] & 0xff;
      int type = ( row < 63 ) ? 0 : ( ( row > 126 ) ? 1 : 2 );
      infos[i].SetRowType( type );
      infos[i].SetId( index );
      infos[i].SetX( fClusters[index].GetX() );
      infos[i].SetY( fClusters[index].GetY() );
      infos[i].SetZ( fClusters[index].GetZ() );
    }

    bool ok = hlt.Merger().FitTrack( t, alpha, t0, alpha, hits1, nHits, 0, 1,infos );

    if ( ok &&  nHits > 15 ) {
      if ( t.TransportToXWithMaterial( xTPC, hlt.Merger().SliceParam().GetBz( t ) ) ) {
        if ( t.GetY() > yMax ) {
          if ( t.Rotate( dAlpha ) ) {
            alpha += dAlpha;
            t.TransportToXWithMaterial( xTPC, hlt.Merger().SliceParam().GetBz( t ) );
          }
        } else if ( t.GetY() < -yMax ) {
          if ( t.Rotate( -dAlpha ) ) {
            alpha += -dAlpha;
            t.TransportToXWithMaterial( xTPC, hlt.Merger().SliceParam().GetBz( t ) );
          }
        }
      }

      AliTPCtrack tt( *esd );
      if ( AliHLTTPCCATrackConvertor::GetExtParam( t, tt, alpha ) ) {
        if ( t.X() > 50 ) esd->UpdateTrackParams( &tt, AliESDtrack::kTPCrefit );
      }
    }
  }
#endif
  return 0;
}

int AliTPCtrackerCA::PropagateBack( AliESDEvent * /*event*/ )
{
  //* backward propagation of ESD tracks
#ifdef XXX

  AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();

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
    AliHLTTPCCAMerger::AliHLTTPCCAClusterInfo infos[500];
    int hits[500], hits1[500];
    int nHits = esd->GetTPCclusters( hits );
    for ( int i = 0; i < nHits; i++ ) {
      hits1[i] = i;
      int index = hits[i];
      infos[i].SetISlice( fClusterSliceRow[index] >> 8 );
      int row = fClusterSliceRow[index] & 0xff;
      int type = ( row < 63 ) ? 0 : ( ( row > 126 ) ? 1 : 2 );
      infos[i].SetRowType( type );
      infos[i].SetId( index );
      infos[i].SetX( fClusters[index].GetX() );
      infos[i].SetY( fClusters[index].GetY() );
      infos[i].SetZ( fClusters[index].GetZ() );
    }

    bool ok = hlt.Merger().FitTrack( t, alpha, t0, alpha, hits1, nHits, 1, 1, infos );

    if ( ok &&  nHits > 15 ) {
      AliTPCtrack tt( *esd );
      if ( AliHLTTPCCATrackConvertor::GetExtParam( t, tt, alpha ) ) {
        if ( t.X() > 50 ) esd->UpdateTrackParams( &tt, AliESDtrack::kTPCout );
      }
    }
  }
#endif
  return 0;
}

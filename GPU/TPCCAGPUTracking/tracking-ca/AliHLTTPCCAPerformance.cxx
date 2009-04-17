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


#include "AliHLTTPCCAPerformance.h"
#include "AliHLTTPCCAGBHit.h"
#include "AliHLTTPCCAMCTrack.h"
#include "AliHLTTPCCAMCPoint.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAGBTrack.h"
#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATracklet.h"


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
    fTracker( 0 ),
    fHitLabels( 0 ),
    fNHits( 0 ),
    fMCTracks( 0 ),
    fNMCTracks( 0 ),
    fMCPoints( 0 ),
    fNMCPoints( 0 ),
    fDoClusterPulls( 0 ),
    fStatNEvents( 0 ),
    fStatTime( 0 ),
    fStatSeedNRecTot( 0 ),
    fStatSeedNRecOut( 0 ),
    fStatSeedNGhost( 0 ),
    fStatSeedNMCAll( 0 ),
    fStatSeedNRecAll( 0 ),
    fStatSeedNClonesAll( 0 ),
    fStatSeedNMCRef( 0 ),
    fStatSeedNRecRef( 0 ),
    fStatSeedNClonesRef( 0 ),
    fStatCandNRecTot( 0 ),
    fStatCandNRecOut( 0 ),
    fStatCandNGhost( 0 ),
    fStatCandNMCAll( 0 ),
    fStatCandNRecAll( 0 ),
    fStatCandNClonesAll( 0 ),
    fStatCandNMCRef( 0 ),
    fStatCandNRecRef( 0 ),
    fStatCandNClonesRef( 0 ),
    fStatNRecTot( 0 ),
    fStatNRecOut( 0 ),
    fStatNGhost( 0 ),
    fStatNMCAll( 0 ),
    fStatNRecAll( 0 ),
    fStatNClonesAll( 0 ),
    fStatNMCRef( 0 ),
    fStatNRecRef( 0 ),
    fStatNClonesRef( 0 ),
    fStatGBNRecTot( 0 ),
    fStatGBNRecOut( 0 ),
    fStatGBNGhost( 0 ),
    fStatGBNMCAll( 0 ),
    fStatGBNRecAll( 0 ),
    fStatGBNClonesAll( 0 ),
    fStatGBNMCRef( 0 ),
    fStatGBNRecRef( 0 ),
    fStatGBNClonesRef( 0 ),
    fHistoDir( 0 ),
    fhResY( 0 ),
    fhResZ( 0 ),
    fhResSinPhi( 0 ),
    fhResDzDs( 0 ),
    fhResPt( 0 ),
    fhPullY( 0 ),
    fhPullZ( 0 ),
    fhPullSinPhi( 0 ),
    fhPullDzDs( 0 ),
    fhPullQPt( 0 ),
    fhPullYS( 0 ),
    fhPullZT( 0 ),
    fhHitErrY( 0 ),
    fhHitErrZ( 0 ),
    fhHitResY( 0 ),
    fhHitResZ( 0 ),
    fhHitPullY( 0 ),
    fhHitPullZ( 0 ),
    fhHitShared( 0 ),
    fhHitResY1( 0 ),
    fhHitResZ1( 0 ),
    fhHitPullY1( 0 ),
    fhHitPullZ1( 0 ),
    fhCellPurity( 0 ),
    fhCellNHits( 0 ),
    fhCellPurityVsN( 0 ),
    fhCellPurityVsPt( 0 ),
    fhEffVsP( 0 ),
    fhSeedEffVsP( 0 ),
    fhCandEffVsP( 0 ),
    fhGBEffVsP( 0 ),
    fhGBEffVsPt( 0 ),
    fhNeighQuality( 0 ),
    fhNeighEff( 0 ),
    fhNeighQualityVsPt( 0 ),
    fhNeighEffVsPt( 0 ),
    fhNeighDy( 0 ),
    fhNeighDz( 0 ),
    fhNeighChi( 0 ),
    fhNeighDyVsPt( 0 ),
    fhNeighDzVsPt( 0 ),
    fhNeighChiVsPt( 0 ),
    fhNeighNCombVsArea( 0 ),
    fhNHitsPerSeed ( 0 ),
    fhNHitsPerTrackCand( 0 ),
    fhTrackLengthRef( 0 ),
    fhRefRecoX( 0 ),
    fhRefRecoY( 0 ),
    fhRefRecoZ( 0 ),
    fhRefRecoP( 0 ),
    fhRefRecoPt( 0 ),
    fhRefRecoAngleY( 0 ),
    fhRefRecoAngleZ( 0 ),
    fhRefRecoNHits( 0 ),
    fhRefNotRecoX( 0 ),
    fhRefNotRecoY( 0 ),
    fhRefNotRecoZ( 0 ),
    fhRefNotRecoP( 0 ),
    fhRefNotRecoPt( 0 ),
    fhRefNotRecoAngleY( 0 ),
    fhRefNotRecoAngleZ( 0 ),
    fhRefNotRecoNHits( 0 )
{
  //* constructor
}


AliHLTTPCCAPerformance::AliHLTTPCCAPerformance( const AliHLTTPCCAPerformance& )
    :
    fTracker( 0 ),
    fHitLabels( 0 ),
    fNHits( 0 ),
    fMCTracks( 0 ),
    fNMCTracks( 0 ),
    fMCPoints( 0 ),
    fNMCPoints( 0 ),
    fDoClusterPulls( 0 ),
    fStatNEvents( 0 ),
    fStatTime( 0 ),
    fStatSeedNRecTot( 0 ),
    fStatSeedNRecOut( 0 ),
    fStatSeedNGhost( 0 ),
    fStatSeedNMCAll( 0 ),
    fStatSeedNRecAll( 0 ),
    fStatSeedNClonesAll( 0 ),
    fStatSeedNMCRef( 0 ),
    fStatSeedNRecRef( 0 ),
    fStatSeedNClonesRef( 0 ),
    fStatCandNRecTot( 0 ),
    fStatCandNRecOut( 0 ),
    fStatCandNGhost( 0 ),
    fStatCandNMCAll( 0 ),
    fStatCandNRecAll( 0 ),
    fStatCandNClonesAll( 0 ),
    fStatCandNMCRef( 0 ),
    fStatCandNRecRef( 0 ),
    fStatCandNClonesRef( 0 ),
    fStatNRecTot( 0 ),
    fStatNRecOut( 0 ),
    fStatNGhost( 0 ),
    fStatNMCAll( 0 ),
    fStatNRecAll( 0 ),
    fStatNClonesAll( 0 ),
    fStatNMCRef( 0 ),
    fStatNRecRef( 0 ),
    fStatNClonesRef( 0 ),
    fStatGBNRecTot( 0 ),
    fStatGBNRecOut( 0 ),
    fStatGBNGhost( 0 ),
    fStatGBNMCAll( 0 ),
    fStatGBNRecAll( 0 ),
    fStatGBNClonesAll( 0 ),
    fStatGBNMCRef( 0 ),
    fStatGBNRecRef( 0 ),
    fStatGBNClonesRef( 0 ),
    fHistoDir( 0 ),
    fhResY( 0 ),
    fhResZ( 0 ),
    fhResSinPhi( 0 ),
    fhResDzDs( 0 ),
    fhResPt( 0 ),
    fhPullY( 0 ),
    fhPullZ( 0 ),
    fhPullSinPhi( 0 ),
    fhPullDzDs( 0 ),
    fhPullQPt( 0 ),
    fhPullYS( 0 ),
    fhPullZT( 0 ),
    fhHitErrY( 0 ),
    fhHitErrZ( 0 ),
    fhHitResY( 0 ),
    fhHitResZ( 0 ),
    fhHitPullY( 0 ),
    fhHitPullZ( 0 ),
    fhHitShared( 0 ),
    fhHitResY1( 0 ),
    fhHitResZ1( 0 ),
    fhHitPullY1( 0 ),
    fhHitPullZ1( 0 ),
    fhCellPurity( 0 ),
    fhCellNHits( 0 ),
    fhCellPurityVsN( 0 ),
    fhCellPurityVsPt( 0 ),
    fhEffVsP( 0 ),
    fhSeedEffVsP( 0 ),
    fhCandEffVsP( 0 ),
    fhGBEffVsP( 0 ),
    fhGBEffVsPt( 0 ),
    fhNeighQuality( 0 ),
    fhNeighEff( 0 ),
    fhNeighQualityVsPt( 0 ),
    fhNeighEffVsPt( 0 ),
    fhNeighDy( 0 ),
    fhNeighDz( 0 ),
    fhNeighChi( 0 ),
    fhNeighDyVsPt( 0 ),
    fhNeighDzVsPt( 0 ),
    fhNeighChiVsPt( 0 ),
    fhNeighNCombVsArea( 0 ),
    fhNHitsPerSeed ( 0 ),
    fhNHitsPerTrackCand( 0 ),
    fhTrackLengthRef( 0 ),
    fhRefRecoX( 0 ),
    fhRefRecoY( 0 ),
    fhRefRecoZ( 0 ),
    fhRefRecoP( 0 ),
    fhRefRecoPt( 0 ),
    fhRefRecoAngleY( 0 ),
    fhRefRecoAngleZ( 0 ),
    fhRefRecoNHits( 0 ),
    fhRefNotRecoX( 0 ),
    fhRefNotRecoY( 0 ),
    fhRefNotRecoZ( 0 ),
    fhRefNotRecoP( 0 ),
    fhRefNotRecoPt( 0 ),
    fhRefNotRecoAngleY( 0 ),
    fhRefNotRecoAngleZ( 0 ),
    fhRefNotRecoNHits( 0 )
{
  //* dummy
}

const AliHLTTPCCAPerformance &AliHLTTPCCAPerformance::operator=( const AliHLTTPCCAPerformance& ) const
{
  //* dummy
  return *this;
}

AliHLTTPCCAPerformance::~AliHLTTPCCAPerformance()
{
  //* destructor
  StartEvent();
}

void AliHLTTPCCAPerformance::SetTracker( AliHLTTPCCAGBTracker * const Tracker )
{
  //* set pointer to HLT CA Global tracker
  fTracker = Tracker;
}

void AliHLTTPCCAPerformance::StartEvent()
{
  //* clean up arrays
  if ( !fHistoDir )  CreateHistos();
  if ( fHitLabels ) delete[] fHitLabels;
  fHitLabels = 0;
  fNHits = 0;
  if ( fMCTracks ) delete[] fMCTracks;
  fMCTracks = 0;
  fNMCTracks = 0;
  if ( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fNMCPoints = 0;
}

void AliHLTTPCCAPerformance::SetNHits( int NHits )
{
  //* set number of hits
  if ( fHitLabels ) delete[] fHitLabels;
  fHitLabels = 0;
  fHitLabels = new AliHLTTPCCAHitLabel[ NHits ];
  fNHits = NHits;
}

void AliHLTTPCCAPerformance::SetNMCTracks( int NumberOfMCTracks )
{
  //* set number of MC tracks
  if ( fMCTracks ) delete[] fMCTracks;
  fMCTracks = 0;
  fMCTracks = new AliHLTTPCCAMCTrack[ NumberOfMCTracks ];
  fNMCTracks = NumberOfMCTracks;
}

void AliHLTTPCCAPerformance::SetNMCPoints( int NMCPoints )
{
  //* set number of MC points
  if ( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fMCPoints = new AliHLTTPCCAMCPoint[ NMCPoints ];
  fNMCPoints = 0;
}

void AliHLTTPCCAPerformance::ReadHitLabel( int HitID,
    int lab0, int lab1, int lab2 )
{
  //* read the hit labels
  AliHLTTPCCAHitLabel hit;
  hit.fLab[0] = lab0;
  hit.fLab[1] = lab1;
  hit.fLab[2] = lab2;
  fHitLabels[HitID] = hit;
}

void AliHLTTPCCAPerformance::ReadMCTrack( int index, const TParticle *part )
{
  //* read mc track to the local array
  fMCTracks[index] = AliHLTTPCCAMCTrack( part );
}

void AliHLTTPCCAPerformance::ReadMCTPCTrack( int index, float X, float Y, float Z,
    float Px, float Py, float Pz )
{
  //* read mc track parameters at TPC
  fMCTracks[index].SetTPCPar( X, Y, Z, Px, Py, Pz );
}

void AliHLTTPCCAPerformance::ReadMCPoint( int TrackID, float X, float Y, float Z, float Time, int iSlice )
{
  //* read mc point to the local array
  AliHLTTPCCAMCPoint &p = fMCPoints[fNMCPoints];
  p.SetTrackID( TrackID );
  p.SetX( X );
  p.SetY( Y );
  p.SetZ( Z );
  p.SetTime( Time );
  p.SetISlice( iSlice );
  float sx, sy, sz;
  fTracker->Slice( iSlice ).Param().Global2Slice( X, Y, Z, &sx, &sy, &sz );
  p.SetSx( sx );
  p.SetSy( sy );
  p.SetSz( sz );
  if ( X*X + Y*Y > 10. ) fNMCPoints++;
}

void AliHLTTPCCAPerformance::CreateHistos()
{
  //* create performance histogramms
  TDirectory *curdir = gDirectory;
  fHistoDir = gROOT->mkdir( "HLTTPCCATrackerPerformance" );
  fHistoDir->cd();

  gDirectory->mkdir( "Links" );
  gDirectory->cd( "Links" );

  fhLinkEff[0] = new TProfile( "fhLinkEffPrimRef", "fhLinkEffPrimRef vs row", 156, 2., 158. );
  fhLinkEff[1] = new TProfile( "fhLinkEffPrimExt", "fhLinkEffPrimExt vs row", 156, 2., 158. );
  fhLinkEff[2] = new TProfile( "fhLinkEffSecRef", "fhLinkEffSecRef vs row", 156, 2., 158. );
  fhLinkEff[3] = new TProfile( "fhLinkEffSecExt", "fhLinkEffSecExt vs row", 156, 2., 158. );
  fhLinkAreaY[0] = new TH1D( "fhLinkAreaYPrimRef", "fhLinkAreaYPrimRef", 100, 0, 10 );
  fhLinkAreaZ[0] = new TH1D( "fhLinkAreaZPrimRef", "fhLinkAreaZPrimRef", 100, 0, 10 );
  fhLinkAreaY[1] = new TH1D( "fhLinkAreaYPrimExt", "fhLinkAreaYPrimExt", 100, 0, 10 );
  fhLinkAreaZ[1] = new TH1D( "fhLinkAreaZPrimExt", "fhLinkAreaZPrimExt", 100, 0, 10 );
  fhLinkAreaY[2] = new TH1D( "fhLinkAreaYSecRef", "fhLinkAreaYSecRef", 100, 0, 10 );
  fhLinkAreaZ[2] = new TH1D( "fhLinkAreaZSecRef", "fhLinkAreaZSecRef", 100, 0, 10 );
  fhLinkAreaY[3] = new TH1D( "fhLinkAreaYSecExt", "fhLinkAreaYSecExt", 100, 0, 10 );
  fhLinkAreaZ[3] = new TH1D( "fhLinkAreaZSecExt", "fhLinkAreaZSecExt", 100, 0, 10 );
  fhLinkChiRight[0] = new TH1D( "fhLinkChiRightPrimRef", "fhLinkChiRightPrimRef", 100, 0, 10 );
  fhLinkChiRight[1] = new TH1D( "fhLinkChiRightPrimExt", "fhLinkChiRightPrimExt", 100, 0, 10 );
  fhLinkChiRight[2] = new TH1D( "fhLinkChiRightSecRef", "fhLinkChiRightSecRef", 100, 0, 10 );
  fhLinkChiRight[3] = new TH1D( "fhLinkChiRightSecExt", "fhLinkChiRightSecExt", 100, 0, 10 );
  fhLinkChiWrong[0] = new TH1D( "fhLinkChiWrongPrimRef", "fhLinkChiWrongPrimRef", 100, 0, 10 );
  fhLinkChiWrong[1] = new TH1D( "fhLinkChiWrongPrimExt", "fhLinkChiWrongPrimExt", 100, 0, 10 );
  fhLinkChiWrong[2] = new TH1D( "fhLinkChiWrongSecRef", "fhLinkChiWrongSecRef", 100, 0, 10 );
  fhLinkChiWrong[3] = new TH1D( "fhLinkChiWrongSecExt", "fhLinkChiWrongSecExt", 100, 0, 10 );

  gDirectory->cd( ".." );

  gDirectory->mkdir( "Neighbours" );
  gDirectory->cd( "Neighbours" );

  fhNeighQuality = new TProfile( "NeighQuality", "Neighbours Quality vs row", 160, 0., 160. );
  fhNeighEff = new TProfile( "NeighEff", "Neighbours Efficiency vs row", 160, 0., 160. );
  fhNeighQualityVsPt = new TProfile( "NeighQualityVsPt", "Neighbours Quality vs Pt", 100, 0., 5. );
  fhNeighEffVsPt = new TProfile( "NeighEffVsPt", "Neighbours Efficiency vs Pt", 100, 0., 5. );
  fhNeighDy = new TH1D( "NeighDy", "Neighbours dy", 100, -10, 10 );
  fhNeighDz =  new TH1D( "NeighDz", "Neighbours dz", 100, -10, 10 );
  fhNeighChi = new TH1D( "NeighChi", "Neighbours chi", 100, 0, 20 );

  fhNeighDyVsPt = new TH2D( "NeighDyVsPt", "NeighDyVsPt", 100, 0, 5, 100, -20, 20 );
  fhNeighDzVsPt = new TH2D( "NeighDzVsPt", "NeighDzVsPt", 100, 0, 5, 100, -20, 20 );
  fhNeighChiVsPt = new TH2D( "NeighChiVsPt", "NeighChiVsPt", 100, 0, 5, 100, 0, 40 );
  fhNeighNCombVsArea = new TH2D( "NeighNCombVsArea", "NeighNCombVsArea", 15, 0, 3, 40, 0, 40 );

  gDirectory->cd( ".." );

  gDirectory->mkdir( "Tracklets" );
  gDirectory->cd( "Tracklets" );

  fhNHitsPerSeed = new TH1D( "NHitsPerSeed", "NHitsPerSeed", 160, 0, 160 );
  fhSeedEffVsP = new TProfile( "fhSeedEffVsP", "Track Seed Eff vs P", 100, 0., 5. );

  gDirectory->cd( ".." );

  gDirectory->mkdir( "TrackCandidates" );
  gDirectory->cd( "TrackCandidates" );

  fhNHitsPerTrackCand = new TH1D( "NHitsPerTrackCand", "NHitsPerTrackCand", 160, 0, 160 );
  fhCandEffVsP = new TProfile( "fhCandEffVsP", "Track Candidate Eff vs P", 100, 0., 5. );

  gDirectory->cd( ".." );

  gDirectory->mkdir( "Tracks" );
  gDirectory->cd( "Tracks" );

  fhTrackLengthRef = new TH1D( "TrackLengthRef", "TrackLengthRef", 100, 0, 1 );

  fhRefRecoX = new TH1D( "fhRefRecoX", "fhRefRecoX", 100, 0, 200. );
  fhRefRecoY = new TH1D( "fhRefRecoY", "fhRefRecoY", 100, -200, 200. );
  fhRefRecoZ = new TH1D( "fhRefRecoZ", "fhRefRecoZ", 100, -250, 250. );


  fhRefRecoP = new TH1D( "fhRefRecoP", "fhRefRecoP", 100, 0, 10. );
  fhRefRecoPt = new TH1D( "fhRefRecoPt", "fhRefRecoPt", 100, 0, 10. );
  fhRefRecoAngleY = new TH1D( "fhRefRecoAngleY", "fhRefRecoAngleY", 100, -180., 180. );
  fhRefRecoAngleZ = new TH1D( "fhRefRecoAngleZ", "fhRefRecoAngleZ", 100, -180., 180 );
  fhRefRecoNHits = new TH1D( "fhRefRecoNHits", "fhRefRecoNHits", 100, 0., 200 );

  fhRefNotRecoX = new TH1D( "fhRefNotRecoX", "fhRefNotRecoX", 100, 0, 200. );
  fhRefNotRecoY = new TH1D( "fhRefNotRecoY", "fhRefNotRecoY", 100, -200, 200. );
  fhRefNotRecoZ = new TH1D( "fhRefNotRecoZ", "fhRefNotRecoZ", 100, -250, 250. );


  fhRefNotRecoP = new TH1D( "fhRefNotRecoP", "fhRefNotRecoP", 100, 0, 10. );
  fhRefNotRecoPt = new TH1D( "fhRefNotRecoPt", "fhRefNotRecoPt", 100, 0, 10. );
  fhRefNotRecoAngleY = new TH1D( "fhRefNotRecoAngleY", "fhRefNotRecoAngleY", 100, -180., 180. );
  fhRefNotRecoAngleZ = new TH1D( "fhRefNotRecoAngleZ", "fhRefNotRecoAngleZ", 100, -180., 180 );
  fhRefNotRecoNHits = new TH1D( "fhRefNotRecoNHits", "fhRefNotRecoNHits", 100, 0., 200 );

  gDirectory->cd( ".." );

  gDirectory->mkdir( "TrackFit" );
  gDirectory->cd( "TrackFit" );

  fhResY = new TH1D( "resY", "track Y resoltion [cm]", 30, -.5, .5 );
  fhResZ = new TH1D( "resZ", "track Z resoltion [cm]", 30, -.5, .5 );
  fhResSinPhi = new TH1D( "resSinPhi", "track SinPhi resoltion ", 30, -.03, .03 );
  fhResDzDs = new TH1D( "resDzDs", "track DzDs resoltion ", 30, -.01, .01 );
  fhResPt = new TH1D( "resPt", "track telative Pt resoltion", 30, -.2, .2 );
  fhPullY = new TH1D( "pullY", "track Y pull", 30, -10., 10. );
  fhPullZ = new TH1D( "pullZ", "track Z pull", 30, -10., 10. );
  fhPullSinPhi = new TH1D( "pullSinPhi", "track SinPhi pull", 30, -10., 10. );
  fhPullDzDs = new TH1D( "pullDzDs", "track DzDs pull", 30, -10., 10. );
  fhPullQPt = new TH1D( "pullQPt", "track Q/Pt pull", 30, -10., 10. );
  fhPullYS = new TH1D( "pullYS", "track Y+SinPhi chi deviation", 100, 0., 30. );
  fhPullZT = new TH1D( "pullZT", "track Z+DzDs chi deviation ", 100, 0., 30. );

  gDirectory->cd( ".." );

  fhEffVsP = new TProfile( "EffVsP", "Eff vs P", 100, 0., 5. );
  fhGBEffVsP = new TProfile( "GBEffVsP", "Global tracker: Eff vs P", 100, 0., 5. );
  fhGBEffVsPt = new TProfile( "GBEffVsPt", "Global tracker: Eff vs Pt", 100, 0.2, 5. );

  gDirectory->mkdir( "Clusters" );
  gDirectory->cd( "Clusters" );

  fhHitShared = new TProfile( "fhHitSharedf", "fhHitShared vs row", 160, 0., 160. );

  fhHitResY = new TH1D( "resHitY", "Y cluster resoltion [cm]", 100, -2., 2. );
  fhHitResZ = new TH1D( "resHitZ", "Z cluster resoltion [cm]", 100, -2., 2. );
  fhHitPullY = new TH1D( "pullHitY", "Y cluster pull", 50, -10., 10. );
  fhHitPullZ = new TH1D( "pullHitZ", "Z cluster pull", 50, -10., 10. );

  fhHitResY1 = new TH1D( "resHitY1", "Y cluster resoltion [cm]", 100, -2., 2. );
  fhHitResZ1 = new TH1D( "resHitZ1", "Z cluster resoltion [cm]", 100, -2., 2. );
  fhHitPullY1 = new TH1D( "pullHitY1", "Y cluster pull", 50, -10., 10. );
  fhHitPullZ1 = new TH1D( "pullHitZ1", "Z cluster pull", 50, -10., 10. );

  fhHitErrY = new TH1D( "HitErrY", "Y cluster error [cm]", 100, 0., 1. );
  fhHitErrZ = new TH1D( "HitErrZ", "Z cluster error [cm]", 100, 0., 1. );

  gDirectory->cd( ".." );

  gDirectory->mkdir( "Cells" );
  gDirectory->cd( "Cells" );
  fhCellPurity = new TH1D( "CellPurity", "Cell Purity", 100, -0.1, 1.1 );
  fhCellNHits = new TH1D( "CellNHits", "Cell NHits", 40, 0., 40. );
  fhCellPurityVsN = new TProfile( "CellPurityVsN", "Cell purity Vs N hits", 40, 2., 42. );
  fhCellPurityVsPt = new TProfile( "CellPurityVsPt", "Cell purity Vs Pt", 100, 0., 5. );
  gDirectory->cd( ".." );

  curdir->cd();
}

void AliHLTTPCCAPerformance::WriteDir2Current( TObject *obj )
{
  //* recursive function to copy the directory 'obj' to the current one
  if ( !obj->IsFolder() ) obj->Write();
  else {
    TDirectory *cur = gDirectory;
    TDirectory *sub = cur->mkdir( obj->GetName() );
    sub->cd();
    TList *listSub = ( ( TDirectory* )obj )->GetList();
    TIter it( listSub );
    while ( TObject *obj1 = it() ) WriteDir2Current( obj1 );
    cur->cd();
  }
}

void AliHLTTPCCAPerformance::WriteHistos()
{
  //* write histograms to the file
  TDirectory *curr = gDirectory;
  // Open output file and write histograms
  TFile* outfile = new TFile( "HLTTPCCATrackerPerformance.root", "RECREATE" );
  outfile->cd();
  WriteDir2Current( fHistoDir );
  outfile->Close();
  curr->cd();
}


void AliHLTTPCCAPerformance::LinkPerformance( int iSlice )
{
  // Efficiency and quality of the found neighbours
  std::cout << "Link performance..." << std::endl;
  if ( !fTracker ) return;
  const AliHLTTPCCATracker &slice = fTracker->Slice( iSlice );

  AliHLTResizableArray<int> mcType( fNMCTracks );

  for ( int imc = 0; imc < fNMCTracks; imc++ ) {
    if ( fMCTracks[imc].P() < .2 ) {  mcType[imc] = -1; continue; }
    float x = fMCTracks[imc].Par()[0];
    float y = fMCTracks[imc].Par()[1];
    //float z = fMCTracks[imc].Par()[2];
    if ( x*x + y*y < 100. ) {
      if ( fMCTracks[imc].P() >= 1 ) mcType[imc] = 0;
      else mcType[imc] = 1;
    } else {
      if ( fMCTracks[imc].P() >= 1 ) mcType[imc] = 2;
      else mcType[imc] = 3;
    }
  }

  struct AliHLTTPCCAMCHits {
    int fNHits;
    int fID[30];
  };
  AliHLTTPCCAMCHits *mcGbHitsUp = new AliHLTTPCCAMCHits[fNMCTracks];
  AliHLTTPCCAMCHits *mcGbHitsDn = new AliHLTTPCCAMCHits[fNMCTracks];

  for ( int iRow = 2; iRow < slice.Param().NRows() - 2; iRow++ ) {

    const AliHLTTPCCARow &row = slice.Row( iRow );
    const AliHLTTPCCARow &rowUp = slice.Row( iRow + 2 );
    const AliHLTTPCCARow &rowDn = slice.Row( iRow - 2 );

    AliHLTResizableArray<int> gbHits  ( row.NHits() );
    AliHLTResizableArray<int> gbHitsUp( rowUp.NHits() );
    AliHLTResizableArray<int> gbHitsDn( rowDn.NHits() );

    for ( int ih = 0; ih < row.NHits()  ; ih++ ) gbHits  [ih] = fTracker->FirstSliceHit()[iSlice] + slice.HitInputID( row  , ih );
    for ( int ih = 0; ih < rowUp.NHits(); ih++ ) gbHitsUp[ih] = fTracker->FirstSliceHit()[iSlice] + slice.HitInputID( rowUp, ih );
    for ( int ih = 0; ih < rowDn.NHits(); ih++ ) gbHitsDn[ih] = fTracker->FirstSliceHit()[iSlice] + slice.HitInputID( rowDn, ih );

    for ( int imc = 0; imc < fNMCTracks; imc++ ) {
      mcGbHitsUp[imc].fNHits = 0;
      mcGbHitsDn[imc].fNHits = 0;
    }

    for ( int ih = 0; ih < rowUp.NHits(); ih++ ) {
      AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[gbHitsUp[ih]].ID()];
      for ( int il = 0; il < 3; il++ ) {
        int imc = l.fLab[il];
        if ( imc < 0 ) break;
        int &nmc = mcGbHitsUp[imc].fNHits;
        if ( nmc >= 30 ) continue;
        mcGbHitsUp[imc].fID[nmc] = gbHitsUp[ih];
        nmc++;
      }
    }

    for ( int ih = 0; ih < rowDn.NHits(); ih++ ) {
      AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[gbHitsDn[ih]].ID()];
      for ( int il = 0; il < 3; il++ ) {
        int imc = l.fLab[il];
        if ( imc < 0 ) break;
        int &nmc = mcGbHitsDn[imc].fNHits;
        if ( nmc >= 30 ) continue;
        mcGbHitsDn[imc].fID[nmc] = gbHitsDn[ih];
        nmc++;
      }
    }

    //float dxUp = rowUp.X() - row.X();
    //float dxDn = row.X() - rowDn.X();
    float tUp = rowUp.X() / row.X();
    float tDn = rowDn.X() / row.X();

    for ( int ih = 0; ih < row.NHits(); ih++ ) {

      int up = slice.HitLinkUpData( row, ih );
      int dn = slice.HitLinkDownData( row, ih );

      const AliHLTTPCCAGBHit &h = fTracker->Hits()[gbHits[ih]];
      AliHLTTPCCAHitLabel &l = fHitLabels[h.ID()];

      int isMC = -1;
      int mcFound = -1;

      float yUp = h.Y() * tUp, zUp = h.Z() * tUp;
      float yDn = h.Y() * tDn, zDn = h.Z() * tDn;

      for ( int il = 0; il < 3; il++ ) {
        int imc = l.fLab[il];
        if ( imc < 0 ) break;

        bool isMcUp = 0, isMcDn = 0;

        float dyMin = 1.e8, dzMin = 1.e8;
        for ( int i = 0; i < mcGbHitsUp[imc].fNHits; i++ ) {
          const AliHLTTPCCAGBHit &h1 = fTracker->Hits()[mcGbHitsUp[imc].fID[i]];
          float dy = TMath::Abs( h1.Y() - yUp );
          float dz = TMath::Abs( h1.Z() - zUp );
          if ( dy*dy + dz*dz < dyMin*dyMin + dzMin*dzMin ) {
            dyMin = dy;
            dzMin = dz;
          }
        }

        if ( mcType[imc] >= 0 && mcGbHitsUp[imc].fNHits >= 0 ) {
          fhLinkAreaY[mcType[imc]]->Fill( dyMin );
          fhLinkAreaZ[mcType[imc]]->Fill( dzMin );
        }
        if ( dyMin*dyMin + dzMin*dzMin < 100. ) isMcUp = 1;

        dyMin = 1.e8;
        dzMin = 1.e8;
        for ( int i = 0; i < mcGbHitsDn[imc].fNHits; i++ ) {
          const AliHLTTPCCAGBHit &h1 = fTracker->Hits()[mcGbHitsDn[imc].fID[i]];
          float dy = TMath::Abs( h1.Y() - yDn );
          float dz = TMath::Abs( h1.Z() - zDn );
          if ( dy*dy + dz*dz < dyMin*dyMin + dzMin*dzMin ) {
            dyMin = dy;
            dzMin = dz;
          }
        }

        if ( mcType[imc] >= 0 && mcGbHitsDn[imc].fNHits >= 0 ) {
          fhLinkAreaY[mcType[imc]]->Fill( dyMin );
          fhLinkAreaZ[mcType[imc]]->Fill( dzMin );
        }
        if ( dyMin*dyMin + dzMin*dzMin < 100. ) isMcDn = 1;

        if ( !isMcUp || !isMcDn ) continue;
        isMC = imc;

        bool found = 0;
        if ( up >= 0 && dn >= 0 ) {
          //std::cout<<"row, ih, mc, up, dn = "<<iRow<<" "<<ih<<" "<<imc<<" "<<up<<" "<<dn<<std::endl;
          const AliHLTTPCCAGBHit &hUp = fTracker->Hits()[gbHitsUp[up]];
          const AliHLTTPCCAGBHit &hDn = fTracker->Hits()[gbHitsDn[dn]];
          AliHLTTPCCAHitLabel &lUp = fHitLabels[hUp.ID()];
          AliHLTTPCCAHitLabel &lDn = fHitLabels[hDn.ID()];
          bool foundUp = 0, foundDn = 0;
          for ( int jl = 0; jl < 3; jl++ ) {
            if ( lUp.fLab[jl] == imc ) foundUp = 1;
            if ( lDn.fLab[jl] == imc ) foundDn = 1;
            //std::cout<<"mc up, dn = "<<lUp.fLab[jl]<<" "<<lDn.fLab[jl]<<std::endl;
          }
          if ( foundUp && foundDn ) found = 1;
        }
        if ( found ) { mcFound = imc; break;}
      }

      if ( mcFound >= 0 ) {
        //std::cout<<" mc "<<mcFound<<" found"<<std::endl;
        if ( mcType[mcFound] >= 0 ) fhLinkEff[mcType[mcFound]]->Fill( iRow, 1 );
      } else if ( isMC >= 0 ) {
        //std::cout<<" mc "<<isMC<<" not found"<<std::endl;
        if ( mcType[isMC] >= 0 ) fhLinkEff[mcType[isMC]]->Fill( iRow, 0 );
      }

    } // ih
  } // iRow
  delete[] mcGbHitsUp;
  delete[] mcGbHitsDn;
}


void AliHLTTPCCAPerformance::SliceTrackletPerformance( int iSlice, bool PrintFlag )
{
  //* calculate slice tracker performance
  if ( !fTracker ) return;

  int nRecTot = 0, nGhost = 0, nRecOut = 0;
  int nMCAll = 0, nRecAll = 0, nClonesAll = 0;
  int nMCRef = 0, nRecRef = 0, nClonesRef = 0;
  const AliHLTTPCCATracker &slice = fTracker->Slice( iSlice );

  int firstSliceHit = fTracker->FirstSliceHit()[iSlice];
  int endSliceHit = fTracker->NHits();
  if ( iSlice < fTracker->NSlices() - 1 ) endSliceHit = fTracker->FirstSliceHit()[iSlice+1];

  // Select reconstructable MC tracks

  {
    for ( int imc = 0; imc < fNMCTracks; imc++ ) fMCTracks[imc].SetNHits( 0 );

    for ( int ih = firstSliceHit; ih < endSliceHit; ih++ ) {
      int id = fTracker->Hits()[ih].ID();
      if ( id < 0 || id >= fNHits ) break;
      AliHLTTPCCAHitLabel &l = fHitLabels[id];
      if ( l.fLab[0] >= 0 ) fMCTracks[l.fLab[0]].SetNHits( fMCTracks[l.fLab[0]].NHits() + 1 );
      if ( l.fLab[1] >= 0 ) fMCTracks[l.fLab[1]].SetNHits( fMCTracks[l.fLab[1]].NHits() + 1 );
      if ( l.fLab[2] >= 0 ) fMCTracks[l.fLab[2]].SetNHits( fMCTracks[l.fLab[2]].NHits() + 1 );
    }

    for ( int imc = 0; imc < fNMCTracks; imc++ ) {
      AliHLTTPCCAMCTrack &mc = fMCTracks[imc];
      mc.SetSet( 0 );
      mc.SetNReconstructed( 0 );
      mc.SetNTurns( 1 );
      if ( mc.NHits() >=  30 && mc.P() >= .05 ) {
        mc.SetSet( 1 );
        nMCAll++;
        if ( mc.NHits() >=  30 && mc.P() >= 1. ) {
          mc.SetSet( 2 );
          nMCRef++;
        }
      }
    }
  }


  int traN = slice.NTracklets();
  int *traLabels = 0;
  double *traPurity = 0;

  traLabels = new int[traN];
  traPurity = new double[traN];
  {
    for ( int itr = 0; itr < traN; itr++ ) {
      traLabels[itr] = -1;
      traPurity[itr] = 0;

      int hits[1600];
      int nHits = 0;

      {
        const AliHLTTPCCAHitId &id = slice.TrackletStartHit( itr );
        int iRow = id.RowIndex();
        int ih =  id.HitIndex();

        while ( ih >= 0 ) {
          const AliHLTTPCCARow &row = slice.Row( iRow );
          hits[nHits] = firstSliceHit + slice.HitInputID( row, ih );
          nHits++;
          ih = slice.HitLinkUpData( row, ih );
          iRow++;
        }
      }

      if ( nHits < 5 ) continue;

      int lb[1600*3];
      int nla = 0;

      for ( int ih = 0; ih < nHits; ih++ ) {
        AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[hits[ih]].ID()];
        if ( l.fLab[0] >= 0 ) lb[nla++] = l.fLab[0];
        if ( l.fLab[1] >= 0 ) lb[nla++] = l.fLab[1];
        if ( l.fLab[2] >= 0 ) lb[nla++] = l.fLab[2];
      }

      sort( lb, lb + nla );
      int labmax = -1, labcur = -1, lmax = 0, lcurr = 0;
      for ( int i = 0; i < nla; i++ ) {
        if ( lb[i] != labcur ) {
          if ( labcur >= 0 && lmax < lcurr ) {
            lmax = lcurr;
            labmax = labcur;
          }
          labcur = lb[i];
          lcurr = 0;
        }
        lcurr++;
      }
      if ( labcur >= 0 && lmax < lcurr ) {
        lmax = lcurr;
        labmax = labcur;
      }
      lmax = 0;
      for ( int ih = 0; ih < nHits; ih++ ) {
        AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[hits[ih]].ID()];
        if ( l.fLab[0] == labmax || l.fLab[1] == labmax || l.fLab[2] == labmax
           ) lmax++;
      }
      traLabels[itr] = labmax;
      traPurity[itr] = ( ( nHits > 0 ) ? double( lmax ) / double( nHits ) : 0 );
    }
  }

  nRecTot += traN;

  for ( int itr = 0; itr < traN; itr++ ) {
    if ( traPurity[itr] < .9 || traLabels[itr] < 0 || traLabels[itr] >= fNMCTracks ) {
      nGhost++;
      continue;
    }

    AliHLTTPCCAMCTrack &mc = fMCTracks[traLabels[itr]];
    mc.SetNReconstructed( mc.NReconstructed() + 1 );
    if ( mc.Set() == 0 ) nRecOut++;
    else {
      if ( mc.NReconstructed() == 1 ) nRecAll++;
      else if ( mc.NReconstructed() > mc.NTurns() ) nClonesAll++;
      if ( mc.Set() == 2 ) {
        if ( mc.NReconstructed() == 1 ) nRecRef++;
        else if ( mc.NReconstructed() > mc.NTurns() ) nClonesRef++;
      }
    }
  }

  for ( int ipart = 0; ipart < fNMCTracks; ipart++ ) {
    AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
    if ( mc.Set() > 0 ) fhSeedEffVsP->Fill( mc.P(), ( mc.NReconstructed() > 0 ? 1 : 0 ) );
  }

  if ( traLabels ) delete[] traLabels;
  if ( traPurity ) delete[] traPurity;

  fStatSeedNRecTot += nRecTot;
  fStatSeedNRecOut += nRecOut;
  fStatSeedNGhost  += nGhost;
  fStatSeedNMCAll  += nMCAll;
  fStatSeedNRecAll  += nRecAll;
  fStatSeedNClonesAll  += nClonesAll;
  fStatSeedNMCRef  += nMCRef;
  fStatSeedNRecRef  += nRecRef;
  fStatSeedNClonesRef  += nClonesRef;

  if ( nMCAll == 0 ) return;

  if ( PrintFlag ) {
    cout << "Track seed performance for slice " << iSlice << " : " << endl;
    cout << " N tracks : "
         << nMCAll << " mc all, "
         << nMCRef << " mc ref, "
         << nRecTot << " rec total, "
         << nRecAll << " rec all, "
         << nClonesAll << " clones all, "
         << nRecRef << " rec ref, "
         << nClonesRef << " clones ref, "
         << nRecOut << " out, "
         << nGhost << " ghost" << endl;

    int nRecExtr = nRecAll - nRecRef;
    int nMCExtr = nMCAll - nMCRef;
    int nClonesExtr = nClonesAll - nClonesRef;

    double dRecTot = ( nRecTot > 0 ) ? nRecTot : 1;
    double dMCAll = ( nMCAll > 0 ) ? nMCAll : 1;
    double dMCRef = ( nMCRef > 0 ) ? nMCRef : 1;
    double dMCExtr = ( nMCExtr > 0 ) ? nMCExtr : 1;
    double dRecAll = ( nRecAll + nClonesAll > 0 ) ? nRecAll + nClonesAll : 1;
    double dRecRef = ( nRecRef + nClonesRef > 0 ) ? nRecRef + nClonesRef : 1;
    double dRecExtr = ( nRecExtr + nClonesExtr > 0 ) ? nRecExtr + nClonesExtr : 1;

    cout << " EffRef = ";
    if ( nMCRef > 0 ) cout << nRecRef / dMCRef; else cout << "_";
    cout << ", CloneRef = ";
    if ( nRecRef > 0 ) cout << nClonesRef / dRecRef; else cout << "_";
    cout << endl;
    cout << " EffExtra = ";
    if ( nMCExtr > 0 ) cout << nRecExtr / dMCExtr; else cout << "_";
    cout << ", CloneExtra = ";
    if ( nRecExtr > 0 ) cout << nClonesExtr / dRecExtr; else cout << "_";
    cout << endl;
    cout << " EffAll = ";
    if ( nMCAll > 0 ) cout << nRecAll / dMCAll; else cout << "_";
    cout << ", CloneAll = ";
    if ( nRecAll > 0 ) cout << nClonesAll / dRecAll; else cout << "_";
    cout << endl;
    cout << " Out = ";
    if ( nRecTot > 0 ) cout << nRecOut / dRecTot; else cout << "_";
    cout << ", Ghost = ";
    if ( nRecTot > 0 ) cout << nGhost / dRecTot; else cout << "_";
    cout << endl;
  }
}




void AliHLTTPCCAPerformance::SliceTrackCandPerformance( int iSlice, bool PrintFlag )
{
  //* calculate slice tracker performance
  if ( !fTracker ) return;

  int nRecTot = 0, nGhost = 0, nRecOut = 0;
  int nMCAll = 0, nRecAll = 0, nClonesAll = 0;
  int nMCRef = 0, nRecRef = 0, nClonesRef = 0;
  const AliHLTTPCCATracker &slice = fTracker->Slice( iSlice );

  int firstSliceHit = fTracker->FirstSliceHit()[iSlice];
  int endSliceHit = fTracker->NHits();
  if ( iSlice < fTracker->NSlices() - 1 ) endSliceHit = fTracker->FirstSliceHit()[iSlice+1];

  // Select reconstructable MC tracks

  {
    for ( int imc = 0; imc < fNMCTracks; imc++ ) fMCTracks[imc].SetNHits( 0 );

    for ( int ih = firstSliceHit; ih < endSliceHit; ih++ ) {
      int id = fTracker->Hits()[ih].ID();
      if ( id < 0 || id >= fNHits ) break;
      AliHLTTPCCAHitLabel &l = fHitLabels[id];
      if ( l.fLab[0] >= 0 ) fMCTracks[l.fLab[0]].SetNHits( fMCTracks[l.fLab[0]].NHits() + 1 );
      if ( l.fLab[1] >= 0 ) fMCTracks[l.fLab[1]].SetNHits( fMCTracks[l.fLab[1]].NHits() + 1 );
      if ( l.fLab[2] >= 0 ) fMCTracks[l.fLab[2]].SetNHits( fMCTracks[l.fLab[2]].NHits() + 1 );
    }

    for ( int imc = 0; imc < fNMCTracks; imc++ ) {
      AliHLTTPCCAMCTrack &mc = fMCTracks[imc];
      mc.SetSet( 0 );
      mc.SetNReconstructed( 0 );
      mc.SetNTurns( 1 );
      if ( mc.NHits() >=  30 && mc.P() >= .05 ) {
        mc.SetSet( 1 );
        nMCAll++;
        if ( mc.NHits() >=  30 && mc.P() >= 1. ) {
          mc.SetSet( 2 );
          nMCRef++;
        }
      }
    }
  }

  int traN = slice.NTracklets();
  int *traLabels = 0;
  double *traPurity = 0;
  traLabels = new int[traN];
  traPurity = new double[traN];
  {
    for ( int itr = 0; itr < traN; itr++ ) {
      traLabels[itr] = -1;
      traPurity[itr] = 0;

      const AliHLTTPCCATracklet &t = slice.Tracklet( itr );

      int nHits = t.NHits();
      if ( nHits < 10 ) continue;
      int firstRow = t.FirstRow();
      int lastRow = t.LastRow();
      nHits = 0;

      int lb[1600*3];
      int nla = 0;

      for ( int irow = firstRow; irow <= lastRow; irow++ ) {
        int ih = t.RowHit( irow );
        if ( ih < 0 ) continue;
        int index = firstSliceHit + slice.HitInputID( slice.Row( irow ), ih );
        AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[index].ID()];
        if ( l.fLab[0] >= 0 ) lb[nla++] = l.fLab[0];
        if ( l.fLab[1] >= 0 ) lb[nla++] = l.fLab[1];
        if ( l.fLab[2] >= 0 ) lb[nla++] = l.fLab[2];
        nHits++;
      }
      if ( nHits < 10 ) continue;

      sort( lb, lb + nla );
      int labmax = -1, labcur = -1, lmax = 0, lcurr = 0;
      for ( int i = 0; i < nla; i++ ) {
        if ( lb[i] != labcur ) {
          if ( labcur >= 0 && lmax < lcurr ) {
            lmax = lcurr;
            labmax = labcur;
          }
          labcur = lb[i];
          lcurr = 0;
        }
        lcurr++;
      }
      if ( labcur >= 0 && lmax < lcurr ) {
        lmax = lcurr;
        labmax = labcur;
      }
      lmax = 0;
      for ( int irow = firstRow; irow <= lastRow; irow++ ) {
        int ih = t.RowHit( irow );
        if ( ih < 0 ) continue;
        int index = firstSliceHit + slice.HitInputID( slice.Row( irow ), ih );
        AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hits()[index].ID()];
        if ( l.fLab[0] == labmax || l.fLab[1] == labmax || l.fLab[2] == labmax
           ) lmax++;
      }
      traLabels[itr] = labmax;
      traPurity[itr] = ( ( nHits > 0 ) ? double( lmax ) / double( nHits ) : 0 );
    }
  }

  nRecTot += traN;

  for ( int itr = 0; itr < traN; itr++ ) {
    if ( traPurity[itr] < .9 || traLabels[itr] < 0 || traLabels[itr] >= fNMCTracks ) {
      nGhost++;
      continue;
    }

    AliHLTTPCCAMCTrack &mc = fMCTracks[traLabels[itr]];
    mc.SetNReconstructed( mc.NReconstructed() + 1 );
    if ( mc.Set() == 0 ) nRecOut++;
    else {
      if ( mc.NReconstructed() == 1 ) nRecAll++;
      else if ( mc.NReconstructed() > mc.NTurns() ) nClonesAll++;
      if ( mc.Set() == 2 ) {
        if ( mc.NReconstructed() == 1 ) nRecRef++;
        else if ( mc.NReconstructed() > mc.NTurns() ) nClonesRef++;
      }
    }
  }

  for ( int ipart = 0; ipart < fNMCTracks; ipart++ ) {
    AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
    if ( mc.Set() > 0 ) fhCandEffVsP->Fill( mc.P(), ( mc.NReconstructed() > 0 ? 1 : 0 ) );
  }

  if ( traLabels ) delete[] traLabels;
  if ( traPurity ) delete[] traPurity;

  fStatCandNRecTot += nRecTot;
  fStatCandNRecOut += nRecOut;
  fStatCandNGhost  += nGhost;
  fStatCandNMCAll  += nMCAll;
  fStatCandNRecAll  += nRecAll;
  fStatCandNClonesAll  += nClonesAll;
  fStatCandNMCRef  += nMCRef;
  fStatCandNRecRef  += nRecRef;
  fStatCandNClonesRef  += nClonesRef;

  if ( nMCAll == 0 ) return;

  if ( PrintFlag ) {
    cout << "Track candidate performance for slice " << iSlice << " : " << endl;
    cout << " N tracks : "
         << nMCAll << " mc all, "
         << nMCRef << " mc ref, "
         << nRecTot << " rec total, "
         << nRecAll << " rec all, "
         << nClonesAll << " clones all, "
         << nRecRef << " rec ref, "
         << nClonesRef << " clones ref, "
         << nRecOut << " out, "
         << nGhost << " ghost" << endl;

    int nRecExtr = nRecAll - nRecRef;
    int nMCExtr = nMCAll - nMCRef;
    int nClonesExtr = nClonesAll - nClonesRef;

    double dRecTot = ( nRecTot > 0 ) ? nRecTot : 1;
    double dMCAll = ( nMCAll > 0 ) ? nMCAll : 1;
    double dMCRef = ( nMCRef > 0 ) ? nMCRef : 1;
    double dMCExtr = ( nMCExtr > 0 ) ? nMCExtr : 1;
    double dRecAll = ( nRecAll + nClonesAll > 0 ) ? nRecAll + nClonesAll : 1;
    double dRecRef = ( nRecRef + nClonesRef > 0 ) ? nRecRef + nClonesRef : 1;
    double dRecExtr = ( nRecExtr + nClonesExtr > 0 ) ? nRecExtr + nClonesExtr : 1;

    cout << " EffRef = ";
    if ( nMCRef > 0 ) cout << nRecRef / dMCRef; else cout << "_";
    cout << ", CloneRef = ";
    if ( nRecRef > 0 ) cout << nClonesRef / dRecRef; else cout << "_";
    cout << endl;
    cout << " EffExtra = ";
    if ( nMCExtr > 0 ) cout << nRecExtr / dMCExtr; else cout << "_";
    cout << ", CloneExtra = ";
    if ( nRecExtr > 0 ) cout << nClonesExtr / dRecExtr; else cout << "_";
    cout << endl;
    cout << " EffAll = ";
    if ( nMCAll > 0 ) cout << nRecAll / dMCAll; else cout << "_";
    cout << ", CloneAll = ";
    if ( nRecAll > 0 ) cout << nClonesAll / dRecAll; else cout << "_";
    cout << endl;
    cout << " Out = ";
    if ( nRecTot > 0 ) cout << nRecOut / dRecTot; else cout << "_";
    cout << ", Ghost = ";
    if ( nRecTot > 0 ) cout << nGhost / dRecTot; else cout << "_";
    cout << endl;
  }
}



void AliHLTTPCCAPerformance::SlicePerformance( int iSlice, bool PrintFlag )
{
  //* calculate slice tracker performance
  if ( !fTracker ) return;

  int nRecTot = 0, nGhost = 0, nRecOut = 0;
  int nMCAll = 0, nRecAll = 0, nClonesAll = 0;
  int nMCRef = 0, nRecRef = 0, nClonesRef = 0;
  const AliHLTTPCCATracker &slice = fTracker->Slice( iSlice );

  int firstSliceHit = 0;
  for ( ; firstSliceHit < fTracker->NHits(); firstSliceHit++ ) {
    if ( fTracker->Hit( firstSliceHit ).ISlice() == iSlice ) break;
  }
  int endSliceHit = firstSliceHit;

  for ( ; endSliceHit < fTracker->NHits(); endSliceHit++ ) {
    if ( fTracker->Hit( endSliceHit ).ISlice() != iSlice ) break;
  }


  // Select reconstructable MC tracks

  {
    for ( int imc = 0; imc < fNMCTracks; imc++ ) fMCTracks[imc].SetNHits( 0 );

    for ( int ih = firstSliceHit; ih < endSliceHit; ih++ ) {
      int id = fTracker->Hit( ih ).ID();
      if ( id < 0 || id > fNHits ) break;
      AliHLTTPCCAHitLabel &l = fHitLabels[id];
      if ( l.fLab[0] >= 0 ) fMCTracks[l.fLab[0]].SetNHits( fMCTracks[l.fLab[0]].NHits() + 1 );
      if ( l.fLab[1] >= 0 ) fMCTracks[l.fLab[1]].SetNHits( fMCTracks[l.fLab[1]].NHits() + 1 );
      if ( l.fLab[2] >= 0 ) fMCTracks[l.fLab[2]].SetNHits( fMCTracks[l.fLab[2]].NHits() + 1 );
    }

    for ( int imc = 0; imc < fNMCTracks; imc++ ) {
      AliHLTTPCCAMCTrack &mc = fMCTracks[imc];
      mc.SetSet( 0 );
      mc.SetNReconstructed( 0 );
      mc.SetNTurns( 1 );
      if ( mc.NHits() >=  30 && mc.P() >= .05 ) {
        mc.SetSet( 1 );
        nMCAll++;
        if ( mc.NHits() >=  30 && mc.P() >= 1. ) {
          mc.SetSet( 2 );
          nMCRef++;
        }
      }
    }
  }

  int traN = *slice.NOutTracks();
  int *traLabels = 0;
  double *traPurity = 0;
  traLabels = new int[traN];
  traPurity = new double[traN];
  {
    for ( int itr = 0; itr < traN; itr++ ) {
      traLabels[itr] = -1;
      traPurity[itr] = 0;
      const AliHLTTPCCAOutTrack &tCA = slice.OutTrack( itr );
      int nhits = tCA.NHits();
      int *lb = new int[nhits*3];
      int nla = 0;
      //cout<<"\nHit labels:"<<endl;
      for ( int ihit = 0; ihit < nhits; ihit++ ) {
        const int index = firstSliceHit + slice.OutTrackHit( tCA.FirstHitRef() + ihit );
        const AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hit( index ).ID()];
        //cout<<l.fLab[0]<<" "<<l.fLab[1]<<" "<<l.fLab[2]<<endl;
        if ( l.fLab[0] >= 0 ) lb[nla++] = l.fLab[0];
        if ( l.fLab[1] >= 0 ) lb[nla++] = l.fLab[1];
        if ( l.fLab[2] >= 0 ) lb[nla++] = l.fLab[2];
      }
      sort( lb, lb + nla );
      int labmax = -1, labcur = -1, lmax = 0, lcurr = 0;
      for ( int i = 0; i < nla; i++ ) {
        if ( lb[i] != labcur ) {
          if ( labcur >= 0 && lmax < lcurr ) {
            lmax = lcurr;
            labmax = labcur;
          }
          labcur = lb[i];
          lcurr = 0;
        }
        lcurr++;
      }
      if ( labcur >= 0 && lmax < lcurr ) {
        lmax = lcurr;
        labmax = labcur;
      }
      lmax = 0;
      for ( int ihit = 0; ihit < nhits; ihit++ ) {
        const int index = firstSliceHit + slice.OutTrackHit( tCA.FirstHitRef() + ihit );
        const AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hit( index ).ID()];
        if ( l.fLab[0] == labmax || l.fLab[1] == labmax || l.fLab[2] == labmax
           ) lmax++;
      }
      traLabels[itr] = labmax;
      traPurity[itr] = ( ( nhits > 0 ) ? double( lmax ) / double( nhits ) : 0 );
      //cout<<"perf track "<<itr<<": "<<nhits<<" "<<labmax<<" "<<traPurity[itr]<<endl;
      if ( lb ) delete[] lb;
    }
  }

  nRecTot += traN;

  for ( int itr = 0; itr < traN; itr++ ) {
    if ( traPurity[itr] < .9 || traLabels[itr] < 0 || traLabels[itr] >= fNMCTracks ) {
      nGhost++;
      continue;
    }

    AliHLTTPCCAMCTrack &mc = fMCTracks[traLabels[itr]];
    mc.SetNReconstructed( mc.NReconstructed() + 1 );
    if ( mc.Set() == 0 ) nRecOut++;
    else {
      if ( mc.NReconstructed() == 1 ) nRecAll++;
      else if ( mc.NReconstructed() > mc.NTurns() ) nClonesAll++;
      if ( mc.Set() == 2 ) {
        if ( mc.NReconstructed() == 1 ) nRecRef++;
        else if ( mc.NReconstructed() > mc.NTurns() ) nClonesRef++;
      }
    }
  }

  for ( int ipart = 0; ipart < fNMCTracks; ipart++ ) {
    AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
    if ( mc.Set() > 0 ) fhEffVsP->Fill( mc.P(), ( mc.NReconstructed() > 0 ? 1 : 0 ) );
  }

  if ( traLabels ) delete[] traLabels;
  if ( traPurity ) delete[] traPurity;

  fStatNRecTot += nRecTot;
  fStatNRecOut += nRecOut;
  fStatNGhost  += nGhost;
  fStatNMCAll  += nMCAll;
  fStatNRecAll  += nRecAll;
  fStatNClonesAll  += nClonesAll;
  fStatNMCRef  += nMCRef;
  fStatNRecRef  += nRecRef;
  fStatNClonesRef  += nClonesRef;

  if ( nMCAll == 0 ) return;

  if ( PrintFlag ) {
    cout << "Performance for slice " << iSlice << " : " << endl;
    cout << " N tracks : "
         << nMCAll << " mc all, "
         << nMCRef << " mc ref, "
         << nRecTot << " rec total, "
         << nRecAll << " rec all, "
         << nClonesAll << " clones all, "
         << nRecRef << " rec ref, "
         << nClonesRef << " clones ref, "
         << nRecOut << " out, "
         << nGhost << " ghost" << endl;

    int nRecExtr = nRecAll - nRecRef;
    int nMCExtr = nMCAll - nMCRef;
    int nClonesExtr = nClonesAll - nClonesRef;

    double dRecTot = ( nRecTot > 0 ) ? nRecTot : 1;
    double dMCAll = ( nMCAll > 0 ) ? nMCAll : 1;
    double dMCRef = ( nMCRef > 0 ) ? nMCRef : 1;
    double dMCExtr = ( nMCExtr > 0 ) ? nMCExtr : 1;
    double dRecAll = ( nRecAll + nClonesAll > 0 ) ? nRecAll + nClonesAll : 1;
    double dRecRef = ( nRecRef + nClonesRef > 0 ) ? nRecRef + nClonesRef : 1;
    double dRecExtr = ( nRecExtr + nClonesExtr > 0 ) ? nRecExtr + nClonesExtr : 1;

    cout << " EffRef = ";
    if ( nMCRef > 0 ) cout << nRecRef / dMCRef; else cout << "_";
    cout << ", CloneRef = ";
    if ( nRecRef > 0 ) cout << nClonesRef / dRecRef; else cout << "_";
    cout << endl;
    cout << " EffExtra = ";
    if ( nMCExtr > 0 ) cout << nRecExtr / dMCExtr; else cout << "_";
    cout << ", CloneExtra = ";
    if ( nRecExtr > 0 ) cout << nClonesExtr / dRecExtr; else cout << "_";
    cout << endl;
    cout << " EffAll = ";
    if ( nMCAll > 0 ) cout << nRecAll / dMCAll; else cout << "_";
    cout << ", CloneAll = ";
    if ( nRecAll > 0 ) cout << nClonesAll / dRecAll; else cout << "_";
    cout << endl;
    cout << " Out = ";
    if ( nRecTot > 0 ) cout << nRecOut / dRecTot; else cout << "_";
    cout << ", Ghost = ";
    if ( nRecTot > 0 ) cout << nGhost / dRecTot; else cout << "_";
    cout << endl;
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
  for ( int islice = 0; islice < fTracker->NSlices(); islice++ ) {
    //SliceTrackletPerformance( islice,0 );
    SliceTrackCandPerformance( islice, 0 );
    SlicePerformance( islice, 0 );
  }


  // global tracker performance
  {
    if ( !fTracker ) return;

    int nRecTot = 0, nGhost = 0, nRecOut = 0;
    int nMCAll = 0, nRecAll = 0, nClonesAll = 0;
    int nMCRef = 0, nRecRef = 0, nClonesRef = 0;

    // Select reconstructable MC tracks

    {
      for ( int imc = 0; imc < fNMCTracks; imc++ ) fMCTracks[imc].SetNHits( 0 );

      for ( int ih = 0; ih < fNHits; ih++ ) {
        AliHLTTPCCAHitLabel &l = fHitLabels[ih];
        if ( l.fLab[0] >= 0 ) fMCTracks[l.fLab[0]].SetNHits( fMCTracks[l.fLab[0]].NHits() + 1 );
        if ( l.fLab[1] >= 0 ) fMCTracks[l.fLab[1]].SetNHits( fMCTracks[l.fLab[1]].NHits() + 1 );
        if ( l.fLab[2] >= 0 ) fMCTracks[l.fLab[2]].SetNHits( fMCTracks[l.fLab[2]].NHits() + 1 );
      }

      for ( int imc = 0; imc < fNMCTracks; imc++ ) {
        AliHLTTPCCAMCTrack &mc = fMCTracks[imc];
        mc.SetSet( 0 );
        mc.SetNReconstructed( 0 );
        mc.SetNTurns( 1 );
        if ( mc.NHits() >=  50 && mc.P() >= .05 ) {
          mc.SetSet( 1 );
          nMCAll++;
          if ( mc.P() >= 1. ) {
            mc.SetSet( 2 );
            nMCRef++;
          }
        }
      }
    }

    int traN = fTracker->NTracks();
    int *traLabels = 0;
    double *traPurity = 0;
    traLabels = new int[traN];
    traPurity = new double[traN];
    {
      for ( int itr = 0; itr < traN; itr++ ) {
        traLabels[itr] = -1;
        traPurity[itr] = 0;
        const AliHLTTPCCAGBTrack &tCA = fTracker->Track( itr );
        const int nhits = tCA.NHits();
        int *lb = new int[nhits*3];
        int nla = 0;
        for ( int ihit = 0; ihit < nhits; ihit++ ) {
          const int index = fTracker->TrackHit( tCA.FirstHitRef() + ihit );
          const AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hit( index ).ID()];
          if ( l.fLab[0] >= 0 ) lb[nla++] = l.fLab[0];
          if ( l.fLab[1] >= 0 ) lb[nla++] = l.fLab[1];
          if ( l.fLab[2] >= 0 ) lb[nla++] = l.fLab[2];
        }
        sort( lb, lb + nla );
        int labmax = -1, labcur = -1, lmax = 0, lcurr = 0;
        for ( int i = 0; i < nla; i++ ) {
          if ( lb[i] != labcur ) {
            if ( labcur >= 0 && lmax < lcurr ) {
              lmax = lcurr;
              labmax = labcur;
            }
            labcur = lb[i];
            lcurr = 0;
          }
          lcurr++;
        }
        if ( labcur >= 0 && lmax < lcurr ) {
          lmax = lcurr;
          labmax = labcur;
        }
        lmax = 0;
        for ( int ihit = 0; ihit < nhits; ihit++ ) {
          const int index = fTracker->TrackHit( tCA.FirstHitRef() + ihit );
          const AliHLTTPCCAHitLabel &l = fHitLabels[fTracker->Hit( index ).ID()];
          if ( l.fLab[0] == labmax || l.fLab[1] == labmax || l.fLab[2] == labmax
             ) lmax++;
        }
        traLabels[itr] = labmax;
        traPurity[itr] = ( ( nhits > 0 ) ? double( lmax ) / double( nhits ) : 0 );
        if ( lb ) delete[] lb;
      }
    }

    nRecTot += traN;
    for ( int itr = 0; itr < traN; itr++ ) {
      if ( traPurity[itr] < .9 || traLabels[itr] < 0 || traLabels[itr] >= fNMCTracks ) {
        nGhost++;
        continue;
      }
      const AliHLTTPCCAGBTrack &tCA = fTracker->Track( itr );
      AliHLTTPCCAMCTrack &mc = fMCTracks[traLabels[itr]];

      mc.SetNReconstructed( mc.NReconstructed() + 1 );
      if ( mc.Set() == 0 ) nRecOut++;
      else {
        if ( mc.NReconstructed() == 1 ) nRecAll++;
        else if ( mc.NReconstructed() > mc.NTurns() ) nClonesAll++;
        if ( mc.Set() == 2 ) {
          if ( mc.NReconstructed() == 1 ) nRecRef++;
          else if ( mc.NReconstructed() > mc.NTurns() ) nClonesRef++;
          fhTrackLengthRef->Fill( tCA.NHits() / ( ( double ) mc.NHits() ) );
        }
      }

      // track resolutions
      while ( mc.Set() == 2 && TMath::Abs( mc.TPCPar()[0] ) + TMath::Abs( mc.TPCPar()[1] ) > 1 ) {
        if ( traPurity[itr] < .90 ) break;
        const AliHLTTPCCAGBTrack &t = fTracker->Track( itr );
        AliHLTTPCCATrackParam p = t.Param();
        double cosA = TMath::Cos( t.Alpha() );
        double sinA = TMath::Sin( t.Alpha() );
        double mcX =  mc.TPCPar()[0] * cosA + mc.TPCPar()[1] * sinA;
        double mcY = -mc.TPCPar()[0] * sinA + mc.TPCPar()[1] * cosA;
        double mcZ =  mc.TPCPar()[2];
        double mcEx =  mc.TPCPar()[3] * cosA + mc.TPCPar()[4] * sinA;
        double mcEy = -mc.TPCPar()[3] * sinA + mc.TPCPar()[4] * cosA;
        double mcEz =  mc.TPCPar()[5];
        double mcEt = TMath::Sqrt( mcEx * mcEx + mcEy * mcEy );
        if ( TMath::Abs( mcEt ) < 1.e-4 ) break;
        double mcSinPhi = mcEy / mcEt;
        double mcDzDs   = mcEz / mcEt;
        double mcQPt = mc.TPCPar()[6] / mcEt;
        if ( TMath::Abs( mcQPt ) < 1.e-4 ) break;
        double mcPt = 1. / TMath::Abs( mcQPt );
        if ( mcPt < 1. ) break;
        if ( t.NHits() <  50 ) break;
        double bz = fTracker->Slice( 0 ).Param().Bz();
        if ( !p.TransportToXWithMaterial( mcX, bz ) ) break;
        if ( p.GetCosPhi()*mcEx < 0 ) { // change direction
          mcSinPhi = -mcSinPhi;
          mcDzDs = -mcDzDs;
          mcQPt = -mcQPt;
        }
        const double kCLight = 0.000299792458;
        double k2QPt = 100;
        if ( TMath::Abs( bz ) > 1.e-4 ) k2QPt = 1. / ( bz * kCLight );
        double qPt = p.GetKappa( bz ) * k2QPt;
        double pt = 100;
        if ( TMath::Abs( qPt ) > 1.e-4 ) pt = 1. / TMath::Abs( qPt );

        fhResY->Fill( p.GetY() - mcY );
        fhResZ->Fill( p.GetZ() - mcZ );
        fhResSinPhi->Fill( p.GetSinPhi() - mcSinPhi );
        fhResDzDs->Fill( p.GetDzDs() - mcDzDs );
        fhResPt->Fill( ( pt - mcPt ) / mcPt );

        if ( p.GetErr2Y() > 0 ) fhPullY->Fill( ( p.GetY() - mcY ) / TMath::Sqrt( p.GetErr2Y() ) );
        if ( p.GetErr2Z() > 0 ) fhPullZ->Fill( ( p.GetZ() - mcZ ) / TMath::Sqrt( p.GetErr2Z() ) );

        if ( p.GetErr2SinPhi() > 0 ) fhPullSinPhi->Fill( ( p.GetSinPhi() - mcSinPhi ) / TMath::Sqrt( p.GetErr2SinPhi() ) );
        if ( p.GetErr2DzDs() > 0 ) fhPullDzDs->Fill( ( p.DzDs() - mcDzDs ) / TMath::Sqrt( p.GetErr2DzDs() ) );
        //if( p.GetErr2Kappa()>0 ) fhPullQPt->Fill( (qPt - mcQPt)/TMath::Sqrt(p.GetErr2Kappa()*k2QPt*k2QPt) );
        //fhPullYS->Fill( TMath::Sqrt(fTracker->GetChi2( p.GetY(), p.GetSinPhi(), p.GetCov()[0], p.GetCov()[3], p.GetCov()[5],
        //             mcY, mcSinPhi, 0,0,0 )));
        //fhPullZT->Fill( TMath::Sqrt(fTracker->GetChi2( p.GetZ(), p.GetDzDs(), p.GetCov()[2], p.GetCov()[7], p.GetCov()[9],
        //    mcZ, mcDzDs, 0,0,0 ) ));

        break;
      }
    }

    for ( int ipart = 0; ipart < fNMCTracks; ipart++ ) {
      AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
      if ( mc.Set() > 0 ) fhGBEffVsP->Fill( mc.P(), ( mc.NReconstructed() > 0 ? 1 : 0 ) );
      if ( mc.Set() > 0 ) fhGBEffVsPt->Fill( mc.Pt(), ( mc.NReconstructed() > 0 ? 1 : 0 ) );
      if ( mc.Set() == 2 ) {
        const double *p = mc.TPCPar();
        double r = TMath::Sqrt( p[0] * p[0] + p[1] * p[1] );
        double cosA = p[0] / r;
        double sinA = p[1] / r;


        double phipos = TMath::Pi() + TMath::ATan2( -p[1], -p[0] );
        double alpha =  TMath::Pi() * ( 20 * ( ( ( ( int )( phipos * 180 / TMath::Pi() ) ) / 20 ) ) + 10 ) / 180.;
        cosA = TMath::Cos( alpha );
        sinA = TMath::Sin( alpha );

        double mcX =  p[0] * cosA + p[1] * sinA;
        double mcY = -p[0] * sinA + p[1] * cosA;
        double mcZ =  p[2];
        double mcEx =  p[3] * cosA + p[4] * sinA;
        double mcEy = -p[3] * sinA + p[4] * cosA;
        double mcEz =  p[5];
        //double mcEt = TMath::Sqrt(mcEx*mcEx + mcEy*mcEy);
        double angleY = TMath::ATan2( mcEy, mcEx ) * 180. / TMath::Pi();
        double angleZ = TMath::ATan2( mcEz, mcEx ) * 180. / TMath::Pi();

        if ( mc.NReconstructed() > 0 ) {
          fhRefRecoX->Fill( mcX );
          fhRefRecoY->Fill( mcY );
          fhRefRecoZ->Fill( mcZ );
          fhRefRecoP->Fill( mc.P() );
          fhRefRecoPt->Fill( mc.Pt() );
          fhRefRecoAngleY->Fill( angleY );
          fhRefRecoAngleZ->Fill( angleZ );
          fhRefRecoNHits->Fill( mc.NHits() );
        } else {
          fhRefNotRecoX->Fill( mcX );
          fhRefNotRecoY->Fill( mcY );
          fhRefNotRecoZ->Fill( mcZ );
          fhRefNotRecoP->Fill( mc.P() );
          fhRefNotRecoPt->Fill( mc.Pt() );
          fhRefNotRecoAngleY->Fill( angleY );
          fhRefNotRecoAngleZ->Fill( angleZ );
          fhRefNotRecoNHits->Fill( mc.NHits() );
        }
      }
    }

    if ( traLabels ) delete[] traLabels;
    if ( traPurity ) delete[] traPurity;

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
    int nHits = fTracker->NHits();
    for ( int ih = 0; ih < nHits; ih++ ) {
      const AliHLTTPCCAGBHit &hit = fTracker->Hit( ih );
      AliHLTTPCCAHitLabel &l = fHitLabels[hit.ID()];
      fhHitErrY->Fill( hit.ErrY() );
      fhHitErrZ->Fill( hit.ErrZ() );
      int nmc = 0;
      for ( int il = 0; il < 3; il++ ) if ( l.fLab[il] >= 0 ) nmc++;
      if ( nmc == 1 ) fhHitShared->Fill( hit.IRow(), 0 );
      else if ( nmc > 1 ) fhHitShared->Fill( hit.IRow(), 1 );
    }
  }

  // cluster pulls

  if ( fDoClusterPulls && fNMCPoints > 0 ) {

    {
      for ( int ipart = 0; ipart < fNMCTracks; ipart++ ) {
        AliHLTTPCCAMCTrack &mc = fMCTracks[ipart];
        mc.SetNMCPoints( 0 );
      }
      sort( fMCPoints, fMCPoints + fNMCPoints, AliHLTTPCCAMCPoint::Compare );

      for ( int ip = 0; ip < fNMCPoints; ip++ ) {
        AliHLTTPCCAMCPoint &p = fMCPoints[ip];
        AliHLTTPCCAMCTrack &t = fMCTracks[p.TrackID()];
        if ( t.NMCPoints() == 0 ) t.SetFirstMCPointID( ip );
        t.SetNMCPoints( t.NMCPoints() + 1 );
      }
    }

    for ( int ih = 0; ih < fNHits; ih++ ) {

      const AliHLTTPCCAGBHit &hit = fTracker->Hit( ih );
      const AliHLTTPCCAHitLabel &l = fHitLabels[ih];

      if ( l.fLab[0] < 0 || l.fLab[0] >= fNMCTracks
           || l.fLab[1] >= 0 || l.fLab[2] >= 0       ) continue;

      int lab = l.fLab[0];

      AliHLTTPCCAMCTrack &track = fMCTracks[lab];
      //if( track.Pt()<1. ) continue;
      int ip1 = -1, ip2 = -1;
      double d1 = 1.e20, d2 = 1.e20;
      for ( int ip = 0; ip < track.NMCPoints(); ip++ ) {
        AliHLTTPCCAMCPoint &p = fMCPoints[track.FirstMCPointID() + ip];
        if ( p.ISlice() != hit.ISlice() ) continue;
        double dx = p.Sx() - hit.X();
        double dy = p.Sy() - hit.Y();
        double dz = p.Sz() - hit.Z();
        double d = dx * dx + dy * dy + dz * dz;
        if ( p.Sx() < hit.X() ) {
          if ( d < d1 ) {
            d1 = d;
            ip1 = ip;
          }
        } else {
          if ( d < d2 ) {
            d2 = d;
            ip2 = ip;
          }
        }
      }

      if ( ip1 < 0 || ip2 < 0 ) continue;

      AliHLTTPCCAMCPoint &p1 = fMCPoints[track.FirstMCPointID() + ip1];
      AliHLTTPCCAMCPoint &p2 = fMCPoints[track.FirstMCPointID() + ip2];
      double dx = p2.Sx() - p1.Sx();
      double dy = p2.Sy() - p1.Sy();
      double dz = p2.Sz() - p1.Sz();
      if ( TMath::Abs( dx ) > 1.e-8 && TMath::Abs( p1.Sx() - hit.X() ) < 2. && TMath::Abs( p2.Sx() - hit.X() ) < 2.  ) {
        double sx = hit.X();
        //double sy = p1.Sy() + dy/dx*(sx-p1.Sx());
        double sz = p1.Sz() + dz / dx * ( sx - p1.Sx() );

        //float errY, errZ;
        {
          AliHLTTPCCATrackParam t;
          t.SetZ( sz );
          t.SetSinPhi( dy / TMath::Sqrt( dx*dx + dy*dy ) );
          t.SetSignCosPhi( dx );
          t.SetDzDs( dz / TMath::Sqrt( dx*dx + dy*dy ) );
          //fTracker->GetErrors2(hit,t,errY, errZ );
          //errY = TMath::Sqrt(errY);
          //errZ = TMath::Sqrt(errZ);
        }
        /*
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
        */
      }
    }
  }

  {
    cout << "\nSlice Track Seed performance: \n" << endl;
    cout << " N tracks : "
         << fStatNMCAll / fStatNEvents << " mc all, "
         << fStatSeedNMCRef / fStatNEvents << " mc ref, "
         << fStatSeedNRecTot / fStatNEvents << " rec total, "
         << fStatSeedNRecAll / fStatNEvents << " rec all, "
         << fStatSeedNClonesAll / fStatNEvents << " clones all, "
         << fStatSeedNRecRef / fStatNEvents << " rec ref, "
         << fStatSeedNClonesRef / fStatNEvents << " clones ref, "
         << fStatSeedNRecOut / fStatNEvents << " out, "
         << fStatSeedNGhost / fStatNEvents << " ghost" << endl;

    int nRecExtr = fStatSeedNRecAll - fStatSeedNRecRef;
    int nMCExtr = fStatNMCAll - fStatNMCRef;
    int nClonesExtr = fStatSeedNClonesAll - fStatSeedNClonesRef;

    double dRecTot = ( fStatSeedNRecTot > 0 ) ? fStatSeedNRecTot : 1;
    double dMCAll = ( fStatNMCAll > 0 ) ? fStatNMCAll : 1;
    double dMCRef = ( fStatNMCRef > 0 ) ? fStatNMCRef : 1;
    double dMCExtr = ( nMCExtr > 0 ) ? nMCExtr : 1;
    double dRecAll = ( fStatSeedNRecAll + fStatSeedNClonesAll > 0 ) ? fStatSeedNRecAll + fStatSeedNClonesAll : 1;
    double dRecRef = ( fStatSeedNRecRef + fStatSeedNClonesRef > 0 ) ? fStatSeedNRecRef + fStatSeedNClonesRef : 1;
    double dRecExtr = ( nRecExtr + nClonesExtr > 0 ) ? nRecExtr + nClonesExtr : 1;

    cout << " EffRef = " << fStatSeedNRecRef / dMCRef
         << ", CloneRef = " << fStatSeedNClonesRef / dRecRef << endl;
    cout << " EffExtra = " << nRecExtr / dMCExtr
         << ", CloneExtra = " << nClonesExtr / dRecExtr << endl;
    cout << " EffAll = " << fStatSeedNRecAll / dMCAll
         << ", CloneAll = " << fStatSeedNClonesAll / dRecAll << endl;
    cout << " Out = " << fStatSeedNRecOut / dRecTot
         << ", Ghost = " << fStatSeedNGhost / dRecTot << endl;
  }

  {
    cout << "\nSlice Track candidate performance: \n" << endl;
    cout << " N tracks : "
         << fStatNMCAll / fStatNEvents << " mc all, "
         << fStatCandNMCRef / fStatNEvents << " mc ref, "
         << fStatCandNRecTot / fStatNEvents << " rec total, "
         << fStatCandNRecAll / fStatNEvents << " rec all, "
         << fStatCandNClonesAll / fStatNEvents << " clones all, "
         << fStatCandNRecRef / fStatNEvents << " rec ref, "
         << fStatCandNClonesRef / fStatNEvents << " clones ref, "
         << fStatCandNRecOut / fStatNEvents << " out, "
         << fStatCandNGhost / fStatNEvents << " ghost" << endl;

    int nRecExtr = fStatCandNRecAll - fStatCandNRecRef;
    int nMCExtr = fStatNMCAll - fStatNMCRef;
    int nClonesExtr = fStatCandNClonesAll - fStatCandNClonesRef;

    double dRecTot = ( fStatCandNRecTot > 0 ) ? fStatCandNRecTot : 1;
    double dMCAll = ( fStatNMCAll > 0 ) ? fStatNMCAll : 1;
    double dMCRef = ( fStatNMCRef > 0 ) ? fStatNMCRef : 1;
    double dMCExtr = ( nMCExtr > 0 ) ? nMCExtr : 1;
    double dRecAll = ( fStatCandNRecAll + fStatCandNClonesAll > 0 ) ? fStatCandNRecAll + fStatCandNClonesAll : 1;
    double dRecRef = ( fStatCandNRecRef + fStatCandNClonesRef > 0 ) ? fStatCandNRecRef + fStatCandNClonesRef : 1;
    double dRecExtr = ( nRecExtr + nClonesExtr > 0 ) ? nRecExtr + nClonesExtr : 1;

    cout << " EffRef = " << fStatCandNRecRef / dMCRef
         << ", CloneRef = " << fStatCandNClonesRef / dRecRef << endl;
    cout << " EffExtra = " << nRecExtr / dMCExtr
         << ", CloneExtra = " << nClonesExtr / dRecExtr << endl;
    cout << " EffAll = " << fStatCandNRecAll / dMCAll
         << ", CloneAll = " << fStatCandNClonesAll / dRecAll << endl;
    cout << " Out = " << fStatCandNRecOut / dRecTot
         << ", Ghost = " << fStatCandNGhost / dRecTot << endl;
  }

  {
    cout << "\nSlice tracker performance: \n" << endl;
    cout << " N tracks : "
         << fStatNMCAll / fStatNEvents << " mc all, "
         << fStatNMCRef / fStatNEvents << " mc ref, "
         << fStatNRecTot / fStatNEvents << " rec total, "
         << fStatNRecAll / fStatNEvents << " rec all, "
         << fStatNClonesAll / fStatNEvents << " clones all, "
         << fStatNRecRef / fStatNEvents << " rec ref, "
         << fStatNClonesRef / fStatNEvents << " clones ref, "
         << fStatNRecOut / fStatNEvents << " out, "
         << fStatNGhost / fStatNEvents << " ghost" << endl;

    int nRecExtr = fStatNRecAll - fStatNRecRef;
    int nMCExtr = fStatNMCAll - fStatNMCRef;
    int nClonesExtr = fStatNClonesAll - fStatNClonesRef;

    double dRecTot = ( fStatNRecTot > 0 ) ? fStatNRecTot : 1;
    double dMCAll = ( fStatNMCAll > 0 ) ? fStatNMCAll : 1;
    double dMCRef = ( fStatNMCRef > 0 ) ? fStatNMCRef : 1;
    double dMCExtr = ( nMCExtr > 0 ) ? nMCExtr : 1;
    double dRecAll = ( fStatNRecAll + fStatNClonesAll > 0 ) ? fStatNRecAll + fStatNClonesAll : 1;
    double dRecRef = ( fStatNRecRef + fStatNClonesRef > 0 ) ? fStatNRecRef + fStatNClonesRef : 1;
    double dRecExtr = ( nRecExtr + nClonesExtr > 0 ) ? nRecExtr + nClonesExtr : 1;

    cout << " EffRef = " << fStatNRecRef / dMCRef
         << ", CloneRef = " << fStatNClonesRef / dRecRef << endl;
    cout << " EffExtra = " << nRecExtr / dMCExtr
         << ", CloneExtra = " << nClonesExtr / dRecExtr << endl;
    cout << " EffAll = " << fStatNRecAll / dMCAll
         << ", CloneAll = " << fStatNClonesAll / dRecAll << endl;
    cout << " Out = " << fStatNRecOut / dRecTot
         << ", Ghost = " << fStatNGhost / dRecTot << endl;
    cout << " Time = " << fTracker->StatTime( 0 ) / fTracker->StatNEvents()*1.e3 << " msec/event " << endl;
    cout << " Local timers = "
         << fTracker->StatTime( 1 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 2 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 3 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 4 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 5 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 6 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 7 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 8 ) / fTracker->StatNEvents()*1.e3 << " "
         << " msec/event " << endl;
  }


  {
    cout << "\nGlobal tracker performance for " << fStatNEvents << " events: \n" << endl;
    cout << " N tracks : "
         << fStatGBNMCAll << " mc all, "
         << fStatGBNMCRef << " mc ref, "
         << fStatGBNRecTot << " rec total, "
         << fStatGBNRecAll << " rec all, "
         << fStatGBNClonesAll << " clones all, "
         << fStatGBNRecRef << " rec ref, "
         << fStatGBNClonesRef << " clones ref, "
         << fStatGBNRecOut << " out, "
         << fStatGBNGhost << " ghost" << endl;
    cout << " N tracks average : "
         << fStatGBNMCAll / fStatNEvents << " mc all, "
         << fStatGBNMCRef / fStatNEvents << " mc ref, "
         << fStatGBNRecTot / fStatNEvents << " rec total, "
         << fStatGBNRecAll / fStatNEvents << " rec all, "
         << fStatGBNClonesAll / fStatNEvents << " clones all, "
         << fStatGBNRecRef / fStatNEvents << " rec ref, "
         << fStatGBNClonesRef / fStatNEvents << " clones ref, "
         << fStatGBNRecOut / fStatNEvents << " out, "
         << fStatGBNGhost / fStatNEvents << " ghost" << endl;

    int nRecExtr = fStatGBNRecAll - fStatGBNRecRef;
    int nMCExtr = fStatGBNMCAll - fStatGBNMCRef;
    int nClonesExtr = fStatGBNClonesAll - fStatGBNClonesRef;

    double dRecTot = ( fStatGBNRecTot > 0 ) ? fStatGBNRecTot : 1;
    double dMCAll = ( fStatGBNMCAll > 0 ) ? fStatGBNMCAll : 1;
    double dMCRef = ( fStatGBNMCRef > 0 ) ? fStatGBNMCRef : 1;
    double dMCExtr = ( nMCExtr > 0 ) ? nMCExtr : 1;
    double dRecAll = ( fStatGBNRecAll + fStatGBNClonesAll > 0 ) ? fStatGBNRecAll + fStatGBNClonesAll : 1;
    double dRecRef = ( fStatGBNRecRef + fStatGBNClonesRef > 0 ) ? fStatGBNRecRef + fStatGBNClonesRef : 1;
    double dRecExtr = ( nRecExtr + nClonesExtr > 0 ) ? nRecExtr + nClonesExtr : 1;

    cout << " EffRef = " << fStatGBNRecRef / dMCRef
         << ", CloneRef = " << fStatGBNClonesRef / dRecRef << endl;
    cout << " EffExtra = " << nRecExtr / dMCExtr
         << ", CloneExtra = " << nClonesExtr / dRecExtr << endl;
    cout << " EffAll = " << fStatGBNRecAll / dMCAll
         << ", CloneAll = " << fStatGBNClonesAll / dRecAll << endl;
    cout << " Out = " << fStatGBNRecOut / dRecTot
         << ", Ghost = " << fStatGBNGhost / dRecTot << endl;
    cout << " Time = " << ( fTracker->StatTime( 0 ) + fTracker->StatTime( 9 ) ) / fTracker->StatNEvents()*1.e3 << " msec/event " << endl;
    cout << " Local timers: " << endl;
    cout << " slice tracker " << fTracker->StatTime( 0 ) / fTracker->StatNEvents()*1.e3 << ": "
         << fTracker->StatTime( 1 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 2 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 3 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 4 ) / fTracker->StatNEvents()*1.e3 << " "
         << fTracker->StatTime( 5 ) / fTracker->StatNEvents()*1.e3 << "["
         << fTracker->StatTime( 6 ) / fTracker->StatNEvents()*1.e3 << "/"
         << fTracker->StatTime( 7 ) / fTracker->StatNEvents()*1.e3 << "] "
         << fTracker->StatTime( 8 ) / fTracker->StatNEvents()*1.e3
         << " msec/event " << endl;
    cout << " GB merger " << fTracker->StatTime( 9 ) / fTracker->StatNEvents()*1.e3 << ": "
         << fTracker->StatTime( 10 ) / fTracker->StatNEvents()*1.e3 << ", "
         << fTracker->StatTime( 11 ) / fTracker->StatNEvents()*1.e3 << ", "
         << fTracker->StatTime( 12 ) / fTracker->StatNEvents()*1.e3 << " "
         << " msec/event " << endl;

    if ( StatFile && StatFile->is_open() ) {
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
      fStatTime += fTracker->SliceTrackerTime();
      double timeHz = 0;
      if ( fStatTime > 1.e-4 ) timeHz = 1. / fStatTime * fStatNEvents;

      out << "<table border>" << endl;
      out << "<tr>" << endl;
      out << "<td>      </td> <td align=center> RefSet </td> <td align=center> AllSet </td> <td align=center> ExtraSet </td>" << endl;
      out << "</tr>" << endl;
      out << "<tr>" << endl;
      out << "<td>Efficiency</td> <td align=center>" << fStatGBNRecRef / dMCRef
      << "</td> <td align=center>" << fStatGBNRecAll / dMCAll
      << "</td> <td align=center>" << nRecExtr / dMCExtr
      << "</td>" << endl;
      out << "</tr>" << endl;
      out << "<tr> " << endl;
      out << "<td>Clone</td>      <td align=center>" << fStatGBNClonesRef / dRecRef
      << "</td> <td align=center>" << fStatGBNClonesAll / dRecAll
      << "</td> <td align=center>" << nClonesExtr / dRecExtr
      << "</td>" << endl;
      out << "</tr>" << endl;
      out << "<tr> " << endl;
      out << "<td>Ghost</td>      <td colspan=3 align=center>" << fStatGBNGhost / dRecTot
      << "</td>" << endl;
      out << "</tr>" << endl;
      out << "<tr> " << endl;
      out << "<td>Time</td>      <td colspan=3 align=center>" << timeHz
      << " ev/s</td>" << endl;
      out << "</tr>" << endl;
      out << "<tr> " << endl;
      out << "<td>N Events</td>      <td colspan=3 align=center>" << fStatNEvents
      << "</td>" << endl;
      out << "</tr>" << endl;
      out << "</table>" << endl;
    }

  }

  WriteHistos();


}

void AliHLTTPCCAPerformance::WriteMCEvent( ostream &out ) const
{
  // write MC information to the file
  out << fNMCTracks << endl;
  for ( int it = 0; it < fNMCTracks; it++ ) {
    AliHLTTPCCAMCTrack &t = fMCTracks[it];
    out << it << " ";
    out << t.PDG() << endl;
    for ( int i = 0; i < 7; i++ ) out << t.Par()[i] << " ";
    out << endl << "    ";
    for ( int i = 0; i < 7; i++ ) out << t.TPCPar()[i] << " ";
    out << endl << "    ";
    out << t.P() << " ";
    out << t.Pt() << " ";
    out << t.NMCPoints() << " ";
    out << t.FirstMCPointID() << " ";
    out << t.NHits() << " ";
    out << t.NReconstructed() << " ";
    out << t.Set() << " ";
    out << t.NTurns() << endl;
  }

  out << fNHits << endl;
  for ( int ih = 0; ih < fNHits; ih++ ) {
    AliHLTTPCCAHitLabel &l = fHitLabels[ih];
    out << l.fLab[0] << " " << l.fLab[1] << " " << l.fLab[2] << endl;
  }
}

void AliHLTTPCCAPerformance::WriteMCPoints( ostream &out ) const
{
  // write Mc points to the file
  out << fNMCPoints << endl;
  for ( int ip = 0; ip < fNMCPoints; ip++ ) {
    AliHLTTPCCAMCPoint &p = fMCPoints[ip];
    out << p.X() << " ";
    out << p.Y() << " ";
    out << p.Z() << " ";
    out << p.Sx() << " ";
    out << p.Sy() << " ";
    out << p.Sz() << " ";
    out << p.Time() << " ";
    out << p.ISlice() << " ";
    out << p.TrackID() << endl;
  }
}

void AliHLTTPCCAPerformance::ReadMCEvent( istream &in )
{
  // read mc info from the file
  StartEvent();
  if ( fMCTracks ) delete[] fMCTracks;
  fMCTracks = 0;
  fNMCTracks = 0;
  if ( fHitLabels ) delete[] fHitLabels;
  fHitLabels = 0;
  fNHits = 0;
  if ( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fNMCPoints = 0;

  in >> fNMCTracks;
  fMCTracks = new AliHLTTPCCAMCTrack[fNMCTracks];
  for ( int it = 0; it < fNMCTracks; it++ ) {
    AliHLTTPCCAMCTrack &t = fMCTracks[it];
    int j;
    float f;
    in >> j;
    in >> j; t.SetPDG( j );
    for ( int i = 0; i < 7; i++ ) { in >> f; t.SetPar( i, f );}
    for ( int i = 0; i < 7; i++ ) { in >> f; t.SetTPCPar( i, f );}
    in >> f; t.SetP( f );
    in >> f; t.SetPt( f );
    in >> j; t.SetNHits( j );
    in >> j; t.SetNMCPoints( j );
    in >> j; t.SetFirstMCPointID( j );
    in >> j; t.SetNReconstructed( j );
    in >> j; t.SetSet( j );
    in >> j; t.SetNTurns( j );
  }

  in >> fNHits;
  fHitLabels = new AliHLTTPCCAHitLabel[fNHits];
  for ( int ih = 0; ih < fNHits; ih++ ) {
    AliHLTTPCCAHitLabel &l = fHitLabels[ih];
    in >> l.fLab[0] >> l.fLab[1] >> l.fLab[2];
  }
}

void AliHLTTPCCAPerformance::ReadMCPoints( istream &in )
{
  // read mc points from the file
  if ( fMCPoints ) delete[] fMCPoints;
  fMCPoints = 0;
  fNMCPoints = 0;

  in >> fNMCPoints;
  fMCPoints = new AliHLTTPCCAMCPoint[fNMCPoints];
  for ( int ip = 0; ip < fNMCPoints; ip++ ) {
    AliHLTTPCCAMCPoint &p = fMCPoints[ip];
    float f;
    int i;
    in >> f;
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

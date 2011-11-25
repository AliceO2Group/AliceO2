//-*- Mode: C++ -*-
// $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAPERFORMANCE_H
#define ALIHLTTPCCAPERFORMANCE_H

#include "AliHLTTPCCADef.h"
#include "Riostream.h"
#include <vector>

class TObject;
class TParticle;
class AliHLTTPCCAMCTrack;
class AliHLTTPCCAMCPoint;
class TDirectory;
class TH1D;
class TH2D;
class TProfile;

/**
 * @class AliHLTTPCCAPerformance
 *
 * Does performance evaluation of the HLT Cellular Automaton-based tracker
 * It checks performance for AliHLTTPCCATracker slice tracker
 * and for AliHLTTPCCAGBTracker global tracker
 *
 */
class AliHLTTPCCAPerformance
{

  public:

    struct AliHLTTPCCAHitLabel {
      int fLab[3]; //* array of 3 MC labels
    };

    AliHLTTPCCAPerformance();

    virtual ~AliHLTTPCCAPerformance();

    static AliHLTTPCCAPerformance &Instance();

    void StartEvent();
    void SetNHits( int NHits );
    void SetNMCTracks( int NMCTracks );
    void SetNMCPoints( int NMCPoints );

    void ReadHitLabel( int HitID,
                       int lab0, int lab1, int lab2 );
    void ReadMCTrack( int index, const TParticle *part );
    void ReadMCTPCTrack( int index, float X, float Y, float Z,
                         float Px, float Py, float Pz );

    void ReadMCPoint( int TrackID, float X, float Y, float Z, float Time, int iSlice );

    void CreateHistos();
    void WriteHistos();
    void GetMCLabel( std::vector<int> &ClusterIDs, int &Label, float &Purity );
    void SlicePerformance( int iSlice, bool PrintFlag  );
    void SliceTrackletPerformance( int iSlice, bool PrintFlag );
    void SliceTrackCandPerformance( int iSlice, bool PrintFlag );
    void ClusterPerformance();
    void MergerPerformance();

    void Performance( fstream *StatFile = 0 );

    void WriteMCEvent( ostream &out ) const;
    void ReadMCEvent( istream &in );
    void WriteMCPoints( ostream &out ) const;
    void ReadMCPoints( istream &in );
    bool DoClusterPulls() const { return fDoClusterPulls; }
    void SetDoClusterPulls( bool v ) { fDoClusterPulls = v; }
    AliHLTTPCCAHitLabel *HitLabels() const { return fHitLabels;}
    AliHLTTPCCAMCTrack *MCTracks() const { return fMCTracks; }
    int NMCTracks() const { return fNMCTracks; }

    TH1D *HNHitsPerSeed() const { return fhNHitsPerSeed;}
    TH1D *HNHitsPerTrackCand() const { return fhNHitsPerTrackCand; }

    TH1D *LinkChiRight( int i ) const { return fhLinkChiRight[i]; }
    TH1D *LinkChiWrong( int i ) const { return fhLinkChiWrong[i]; }

    void LinkPerformance( int iSlice );
    void SmearClustersMC();

  protected:


    AliHLTTPCCAHitLabel *fHitLabels; //* array of hit MC labels
    int fNHits;                    //* number of hits
    AliHLTTPCCAMCTrack *fMCTracks;   //* array of MC tracks
    int fNMCTracks;                //* number of MC tracks
    AliHLTTPCCAMCPoint *fMCPoints;   //* array of MC points
    int fNMCPoints;                //* number of MC points
    bool fDoClusterPulls;          //* do cluster pulls (very slow)
    int fStatNEvents; //* n of events proceed
    double fStatTime; //* reco time;

    int fStatSeedNRecTot; //* total n of reconstructed tracks
    int fStatSeedNRecOut; //* n of reconstructed tracks in Out set
    int fStatSeedNGhost;//* n of reconstructed tracks in Ghost set
    int fStatSeedNMCAll;//* n of MC tracks
    int fStatSeedNRecAll; //* n of reconstructed tracks in All set
    int fStatSeedNClonesAll;//* total n of reconstructed tracks in Clone set
    int fStatSeedNMCRef; //* n of MC reference tracks
    int fStatSeedNRecRef; //* n of reconstructed tracks in Ref set
    int fStatSeedNClonesRef; //* n of reconstructed clones in Ref set

    int fStatCandNRecTot; //* total n of reconstructed tracks
    int fStatCandNRecOut; //* n of reconstructed tracks in Out set
    int fStatCandNGhost;//* n of reconstructed tracks in Ghost set
    int fStatCandNMCAll;//* n of MC tracks
    int fStatCandNRecAll; //* n of reconstructed tracks in All set
    int fStatCandNClonesAll;//* total n of reconstructed tracks in Clone set
    int fStatCandNMCRef; //* n of MC reference tracks
    int fStatCandNRecRef; //* n of reconstructed tracks in Ref set
    int fStatCandNClonesRef; //* n of reconstructed clones in Ref set

    int fStatNRecTot; //* total n of reconstructed tracks
    int fStatNRecOut; //* n of reconstructed tracks in Out set
    int fStatNGhost;//* n of reconstructed tracks in Ghost set
    int fStatNMCAll;//* n of MC tracks
    int fStatNRecAll; //* n of reconstructed tracks in All set
    int fStatNClonesAll;//* total n of reconstructed tracks in Clone set
    int fStatNMCRef; //* n of MC reference tracks
    int fStatNRecRef; //* n of reconstructed tracks in Ref set
    int fStatNClonesRef; //* n of reconstructed clones in Ref set

    int fStatGBNRecTot; //* global tracker: total n of reconstructed tracks
    int fStatGBNRecOut; //* global tracker: n of reconstructed tracks in Out set
    int fStatGBNGhost;//* global tracker: n of reconstructed tracks in Ghost set
    int fStatGBNMCAll;//* global tracker: n of MC tracks
    int fStatGBNRecAll; //* global tracker: n of reconstructed tracks in All set
    int fStatGBNClonesAll;//* global tracker: total n of reconstructed tracks in Clone set
    int fStatGBNMCRef; //* global tracker: n of MC reference tracks
    int fStatGBNRecRef; //* global tracker: n of reconstructed tracks in Ref set
    int fStatGBNClonesRef; //* global tracker: n of reconstructed clones in Ref set

    TDirectory *fHistoDir; //* ROOT directory with histogramms

    TH1D
    *fhResY,       //* track Y resolution at the TPC entrance
    *fhResZ,       //* track Z resolution at the TPC entrance
    *fhResSinPhi,  //* track SinPhi resolution at the TPC entrance
    *fhResDzDs,    //* track DzDs resolution at the TPC entrance
    *fhResPt,      //* track Pt relative resolution at the TPC entrance
    *fhPullY,      //* track Y pull at the TPC entrance
    *fhPullZ,      //* track Z pull at the TPC entrance
    *fhPullSinPhi, //* track SinPhi pull at the TPC entrance
    *fhPullDzDs, //* track DzDs pull at the TPC entrance
    *fhPullQPt,    //* track Q/Pt pull at the TPC entrance
    *fhPullYS,       //* sqrt(chi2/ndf) deviation of the track parameters Y and SinPhi at the TPC entrance
    *fhPullZT;      //* sqrt(chi2/ndf) deviation of the track parameters Z and DzDs at the TPC entrance

    TH1D
    *fhHitErrY, //* hit error in Y
    *fhHitErrZ,//* hit error in Z
    *fhHitResY,//* hit resolution Y
    *fhHitResZ,//* hit resolution Z
    *fhHitPullY,//* hit  pull Y
    *fhHitPullZ;//* hit  pull Z
    TProfile *fhHitShared; //* ratio of the shared clusters

    TH1D
    *fhHitResY1,//* hit resolution Y, pt>1GeV
    *fhHitResZ1,//* hit resolution Z, pt>1GeV
    *fhHitPullY1,//* hit  pull Y, pt>1GeV
    *fhHitPullZ1;//* hit  pull Z, pt>1GeV

    TH1D
    *fhCellPurity,//* cell purity
    *fhCellNHits//* cell n hits
    ;

    TProfile
    *fhCellPurityVsN, //* cell purity vs N hits
    *fhCellPurityVsPt,//* cell purity vs MC Pt
    *fhEffVsP, //* reconstruction efficiency vs P plot
    *fhSeedEffVsP, //* reconstruction efficiency vs P plot
    *fhCandEffVsP, //* reconstruction efficiency vs P plot
    *fhGBEffVsP, //* global reconstruction efficiency vs P plot
    *fhGBEffVsPt, //* global reconstruction efficiency vs P plot
    *fhNeighQuality, // quality for neighbours finder
    *fhNeighEff,// efficiency for neighbours finder
    *fhNeighQualityVsPt,// quality for neighbours finder vs track Pt
    *fhNeighEffVsPt;// efficiency for neighbours finder vs track Pt
    TH1D
    *fhNeighDy, // dy for neighbours
    *fhNeighDz,// dz for neighbours
    *fhNeighChi;// chi2^0.5 for neighbours
    TH2D
    *fhNeighDyVsPt, // dy for neighbours vs track Pt
    *fhNeighDzVsPt,// dz for neighbours vs track Pt
    *fhNeighChiVsPt, // chi2^0.5 for neighbours vs track Pt
    *fhNeighNCombVsArea; // N neighbours in the search area

    TH1D
    *fhNHitsPerSeed, // n hits per track seed
    *fhNHitsPerTrackCand; // n hits per track candidate

    TH1D
    *fhTrackLengthRef, // reconstructed track length, %
    *fhRefRecoX,// parameters of non-reconstructed ref. mc track
    *fhRefRecoY,// parameters of non-reconstructed ref. mc track
    *fhRefRecoZ,// parameters of non-reconstructed ref. mc track
    *fhRefRecoP, // parameters of non-reconstructed ref. mc track
    *fhRefRecoPt,// parameters of non-reconstructed ref. mc track
    *fhRefRecoAngleY,// parameters of non-reconstructed ref. mc track
    *fhRefRecoAngleZ,// parameters of non-reconstructed ref. mc track
    *fhRefRecoNHits,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoX,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoY,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoZ,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoP, // parameters of non-reconstructed ref. mc track
    *fhRefNotRecoPt,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoAngleY,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoAngleZ,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoNHits;// parameters of non-reconstructed ref. mc track

    TProfile * fhLinkEff[4]; // link efficiency
    TH1D *fhLinkAreaY[4]; // area in Y for the link finder
    TH1D *fhLinkAreaZ[4]; // area in Z for the link finder
    TH1D *fhLinkChiRight[4]; // sqrt(chi^2) for right neighbours
    TH1D *fhLinkChiWrong[4]; // sqrt(chi^2) for wrong neighbours

    static void WriteDir2Current( TObject *obj );

private:
  /// copy constructor prohibited
  AliHLTTPCCAPerformance( const AliHLTTPCCAPerformance& );
  /// assignment operator prohibited
  AliHLTTPCCAPerformance &operator=( const AliHLTTPCCAPerformance& ) const;

};

#endif //ALIHLTTPCCAPERFORMANCE_H

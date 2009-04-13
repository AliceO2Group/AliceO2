//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAGBTRACKER_H
#define ALIHLTTPCCAGBTRACKER_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATrackParam.h"

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#endif

class AliHLTTPCCATracker;
class AliHLTTPCCAGBTrack;
class AliHLTTPCCAGBHit;
class TParticle;
class TProfile;
class AliHLTTPCCATrackParam;
class  AliHLTTPCCAMerger;

/**
 * @class AliHLTTPCCAGBTracker
 * 
 * Global Cellular Automaton-based HLT tracker for TPC detector
 * The class reconstructs tracks in the whole TPC
 * It calls the AliHLTTPCCATracker slice tracker and constructs 
 * the global TPC tracks by merging the slice tracks 
 *
 * The tracker is designed stand-alone. 
 * It will be integrated to the HLT framework via AliHLTTPCCAGBTrackerComponent interface,
 * and to off-line framework via TPC/AliTPCtrackerCA class
 * The class is under construction.
 *
 */
class AliHLTTPCCAGBTracker
{

public:

  AliHLTTPCCAGBTracker();
  AliHLTTPCCAGBTracker(const AliHLTTPCCAGBTracker&);
  const AliHLTTPCCAGBTracker &operator=(const AliHLTTPCCAGBTracker&) const;

  ~AliHLTTPCCAGBTracker();

  void StartEvent();
  void SetNSlices( Int_t N );
  void SetNHits( Int_t nHits );

  void ReadHit( Float_t x, Float_t y, Float_t z, 
		Float_t ErrY, Float_t ErrZ, Float_t amp,
		Int_t ID, Int_t iSlice, Int_t iRow );

  void FindTracks();

  void Merge();

  AliHLTTPCCATracker *Slices() const { return fSlices; }
  AliHLTTPCCAGBHit *Hits() const { return fHits; }
  Int_t Ext2IntHitID( Int_t i ) const { return fExt2IntHitID[i]; }

  Int_t NHits() const { return fNHits; }
  Int_t NSlices() const { return fNSlices; }
  Double_t Time() const { return fTime; }
  Double_t StatTime( Int_t iTimer ) const { return fStatTime[iTimer]; }
  Int_t StatNEvents() const { return fStatNEvents; }
  Int_t NTracks() const { return fNTracks; }
  AliHLTTPCCAGBTrack *Tracks() const { return fTracks; }
  Int_t *TrackHits() const { return fTrackHits; }

  Bool_t FitTrack( AliHLTTPCCATrackParam &T, AliHLTTPCCATrackParam t0, 
		   Float_t &Alpha, Int_t hits[], Int_t &NTrackHits, 
		   Bool_t dir );

  void WriteSettings( std::ostream &out ) const;
  void ReadSettings( std::istream &in );
  void WriteEvent( std::ostream &out ) const;
  void ReadEvent( std::istream &in );
  void WriteTracks( std::ostream &out ) const;
  void ReadTracks( std::istream &in );

  Double_t SliceTrackerTime() const { return fSliceTrackerTime; }
  void SetSliceTrackerTime( Double_t v ){ fSliceTrackerTime = v; }
  const Int_t *FirstSliceHit() const { return fFirstSliceHit; }


protected:

  AliHLTTPCCATracker *fSlices; //* array of slice trackers
  Int_t fNSlices;              //* N slices
  AliHLTTPCCAGBHit *fHits;     //* hit array
  Int_t *fExt2IntHitID;        //* array of internal hit indices
  Int_t fNHits;                //* N hits in event
  Int_t *fTrackHits;           //* track->hits reference array
  AliHLTTPCCAGBTrack *fTracks; //* array of tracks
  Int_t fNTracks;              //* N tracks
  AliHLTTPCCAMerger *fMerger;  //* global merger

  Double_t fTime; //* total time
  Double_t fStatTime[20]; //* timers 
  Int_t fStatNEvents;    //* n events proceed
  Int_t fFirstSliceHit[100]; // hit array

  Double_t fSliceTrackerTime; // reco time of the slice tracker;

};

#endif

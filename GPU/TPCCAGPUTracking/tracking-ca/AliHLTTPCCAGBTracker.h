//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAGBTRACKER_H
#define ALIHLTTPCCAGBTRACKER_H

#include "AliHLTTPCCADef.h"

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#endif

class AliHLTTPCCATracker;
class AliHLTTPCCAGBTrack;
class AliHLTTPCCAGBHit;
class TParticle;
class TProfile;
class AliHLTTPCCATrackParam;

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
  AliHLTTPCCAGBTracker &operator=(const AliHLTTPCCAGBTracker&);

  ~AliHLTTPCCAGBTracker();

  void StartEvent();
  void SetNSlices( Int_t N );
  void SetNHits( Int_t nHits );

  void ReadHit( Float_t x, Float_t y, Float_t z, 
		Float_t ErrY, Float_t ErrZ, Float_t amp,
		Int_t ID, Int_t iSlice, Int_t iRow );

  void FindTracks();

  void FindTracks0();
  void FindTracks1();
  void FindTracks2();


  void Merging();
  AliHLTTPCCATracker *Slices(){ return fSlices; }
  AliHLTTPCCAGBHit *Hits(){ return fHits; }
  Int_t NHits() const { return fNHits; }
  Int_t NSlices() const { return fNSlices; }
  Double_t Time() const { return fTime; }
  Double_t StatTime( Int_t iTimer ) const { return fStatTime[iTimer]; }
  Int_t StatNEvents() const { return fStatNEvents; }
  Int_t NTracks() const { return fNTracks; }
  AliHLTTPCCAGBTrack *Tracks(){ return fTracks; }
  Int_t *TrackHits() {return fTrackHits; }
  void GetErrors2( AliHLTTPCCAGBHit &h, AliHLTTPCCATrackParam &t, Float_t &Err2Y, Float_t &Err2Z );
  void GetErrors2( Int_t iSlice, Int_t iRow, AliHLTTPCCATrackParam &t, Float_t &Err2Y, Float_t &Err2Z );

  void WriteSettings( std::ostream &out ) const;
  void ReadSettings( std::istream &in );
  void WriteEvent( std::ostream &out ) const;
  void ReadEvent( std::istream &in );
  void WriteTracks( std::ostream &out ) const;
  void ReadTracks( std::istream &in );

  Double_t &SliceTrackerTime(){ return fSliceTrackerTime; }

protected:

  AliHLTTPCCATracker *fSlices; //* array of slice trackers
  Int_t fNSlices;              //* N slices
  AliHLTTPCCAGBHit *fHits;     //* hit array
  Int_t fNHits;                //* N hits in event
  Int_t *fTrackHits;           //* track->hits reference array
  AliHLTTPCCAGBTrack *fTracks; //* array of tracks
  Int_t fNTracks;              //* N tracks

  struct AliHLTTPCCAGBSliceTrackInfo{
    Int_t fPrevNeighbour; //* neighbour in the previous slide
    Int_t fNextNeighbour; //* neighbour in the next slide
    Bool_t fUsed;         //* is the slice track used by global tracks
  };

  AliHLTTPCCAGBSliceTrackInfo **fSliceTrackInfos; //* additional information for slice tracks
  Double_t fTime; //* total time
  Double_t fStatTime[20]; //* timers 
  Int_t fStatNEvents;    //* n events proceed
  Int_t fFirstSliceHit[100]; // hit array

  Double_t fSliceTrackerTime; // reco time of the slice tracker;

};

#endif

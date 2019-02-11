#ifndef ALIESDTOFCLUSTER_H
#define ALIESDTOFCLUSTER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: $ */

//----------------------------------------------------------------------//
//                                                                      //
// AliESDTOFCluster Class                                                //
//                                                                      //
//----------------------------------------------------------------------//
#include "TMath.h"
#include <AliVTOFcluster.h>
#include "AliESDTOFHit.h"
#include "AliESDTOFMatch.h"

class AliESDEvent;

class AliESDTOFCluster : public AliVTOFcluster {

 public:
  AliESDTOFCluster(Int_t id=-1);
  AliESDTOFCluster(Int_t ,Int_t ,Float_t ,Float_t ,Float_t ,Int_t*,Int_t ,Int_t ,  Bool_t ,Float_t ,Float_t ,Float_t ,
		   Int_t ,Float_t ,Float_t ,Float_t ,Float_t ,Double_t*);
  AliESDTOFCluster(Int_t ,Int_t ,Float_t ,Float_t ,Float_t t,Int_t*,Int_t ,Int_t ,
		   Bool_t ,Float_t ,Float_t ,Float_t );
  AliESDTOFCluster(const AliESDTOFCluster & source);
  AliESDTOFCluster & operator=(const AliESDTOFCluster & source);
  virtual ~AliESDTOFCluster() {}

  Int_t GetESDID()          const {return GetUniqueID();}
  void  SetESDID(Int_t id)        {SetUniqueID(id);}
  Int_t GetID()             const {return fID;}
  void  SetID(Int_t id)           {fID = id;}

  Int_t Update(Int_t trackIndex,Float_t dX,Float_t dY,Float_t dZ,Float_t length,Double_t expTimes[9]);
  void  SuppressMatchedTrack(Int_t id);
  void  ReplaceMatchedTrackID(int oldID, int newID);
  void  ReplaceMatchID(int oldID, int newID);
  void  ReplaceHitID(int oldID, int newID);
  void  FixSelfReferences(int oldID, int newID);
  Int_t GetClusterIndex(Int_t ihit=0) const; // cluster index
  Int_t GetTOFchannel(Int_t ihit=0) const; // TOF channel
  Float_t GetTime(Int_t ihit=0) const; // TOF time
  Float_t GetTimeRaw(Int_t ihit=0) const; // TOF raw time
  Float_t GetTOT(Int_t ihit=0) const; // TOF tot
  Float_t GetTOFsignalToT(Int_t ihit=0) const; // TOF tot
  Int_t GetLabel(Int_t i=0,Int_t ihit=0) const;
  Int_t GetDeltaBC(Int_t ihit=0) const;
  Int_t GetL0L1Latency(Int_t ihit=0) const;
  Bool_t GetStatus() const;
  Float_t GetZ() const;
  Float_t GetPhi() const;
  Float_t GetR() const;
  Int_t GetNMatchableTracks() const;
  Int_t GetNTOFhits() const;

  Int_t GetTrackIndex(Int_t i=0) const;
  Float_t GetDistanceInStripPlane(Int_t i=0)   const; // distance
  Float_t GetDx(Int_t i=0)  const; // distance, X component
  Float_t GetDy(Int_t i=0)  const; // distance, Y component
  Float_t GetDz(Int_t i=0)  const; // distance, Z component
  Float_t GetLength(Int_t i=0) const; // reconstructed track length at TOF
  Double_t GetIntegratedTime(Int_t iPart=0,Int_t i=0) const; // reconstructed track length at TOF
  void SetStatus(Int_t status) {fStatus=status;};

  void AddESDTOFHitIndex(Int_t hitID);
  void AddTOFhit(Int_t ,Int_t ,Float_t ,Float_t ,Float_t ,Int_t*,Int_t ,Int_t , Bool_t ,Float_t ,Float_t ,Float_t );
  void AddTOFhit(AliESDTOFHit *hit);

  Int_t GetHitIndex(Int_t i) const {return fHitIndex[i];}
  void  SetHitIndex(Int_t i,Int_t index) {fHitIndex[i] = index;}

  void Print(const Option_t *opt=0) const;

  AliESDTOFHit*   GetTOFHit(Int_t i) const;
  AliESDTOFMatch* GetTOFMatch(Int_t i) const;

 protected:
  Int_t  fID;               // raw cluster id
  Char_t fNTOFhits;         // number of TOF hit in the cluster
  Bool_t fStatus;           // !
  Char_t fNmatchableTracks; // number of matchable tracks with the same TOF matchable hit
  Int_t  fHitIndex[kMaxHits];          // pointing to hit info
  Int_t  fMatchIndex[kMaxMatches];     // pointing to matching info

  ClassDef(AliESDTOFCluster, 2) // TOF matchable cluster

}; 

#endif

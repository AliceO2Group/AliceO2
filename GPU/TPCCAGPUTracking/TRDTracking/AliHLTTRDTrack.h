#ifndef ALIHLTTRDTRACK_H
#define ALIHLTTRDTRACK_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


#include "AliKalmanTrack.h"

class AliESDtrack;
class AliHLTTRDTrackDataRecord;

//_____________________________________________________________________________
class AliHLTTRDTrack : public AliKalmanTrack
{
 public:

  AliHLTTRDTrack();
  AliHLTTRDTrack(AliESDtrack& t,Bool_t c=kFALSE) throw (const Char_t *);
  AliHLTTRDTrack(AliExternalTrackParam& t ) throw (const Char_t *);
  AliHLTTRDTrack(const AliHLTTRDTrack& t);
  AliHLTTRDTrack &operator=(const AliHLTTRDTrack& t);

  Int_t GetTPCtrackId() const { return fTPCtrackId; }
  Int_t GetNtracklets() const { return fNtracklets; }
  Int_t GetNlayers() const { return fNlayers; }
  Int_t GetNtrackletsOffline() const { return fNtrackletsOffline; }
  Int_t GetTracklet(Int_t iLayer) const;
  Int_t GetNmissingConsecLayers() const { return fNmissingConsecLayers; }
  Bool_t GetIsStopped() const { return fIsStopped; }

  void AddTracklet(int iLayer, int idx) { fAttachedTracklets[iLayer] = idx; }
  void SetTPCtrackId( Int_t v ){ fTPCtrackId = v;}
  void SetNtracklets( Int_t nTrklts) { fNtracklets = nTrklts; }
  void SetNlayers(Int_t nLayers) { fNlayers = nLayers; }
  void SetNtrackletsOffline(Int_t nTrklts) { fNtrackletsOffline = nTrklts; }
  void SetNmissingConsecLayers(Int_t nLayers) { fNmissingConsecLayers = nLayers; }
  void SetIsStopped() { fIsStopped = true; }

  using AliExternalTrackParam::GetPredictedChi2;
  using AliExternalTrackParam::Update;

  //methods below need to be implemented from abstract base class
  Double_t GetPredictedChi2(const AliCluster *c) { Error("AliHLTTRDTrack", "This is a dummy method and should never be used"); return -1.0; }
  Double_t GetPredictedChi2(const AliCluster *c) const { Error("AliHLTTRDTrack", "This is a dummy method and should never be used"); return -1.0; }
  Double_t GetPredictedChi2(Double_t cy, Double_t cz, Double_t cerr2Y, Double_t cerr2Z) const { Error("AliHLTTRDTrack", "This is a dummy method and should never be used"); return -1.0; }
  Bool_t PropagateTo(Double_t xr, Double_t x0, Double_t rho) { Error("AliHLTTRDTrack", "This is a dummy method and should never be used"); return false; }
  Bool_t Update(const AliCluster* c, Double_t chi2, Int_t index) { Error("AliHLTTRDTrack", "This is a dummy method and should never be used"); return false; }

  // virtual method of AliKalmanTrack, it is used when the TRD track is saved in AliESDTrack
  Int_t GetTrackletIndex(Int_t iLayer) const {
    return GetTracklet(iLayer);
  }


  // convertion to HLT track structure

  void ConvertTo( AliHLTTRDTrackDataRecord &t ) const;
  void ConvertFrom( const AliHLTTRDTrackDataRecord &t );


 protected:

  Int_t fTPCtrackId;            // corresponding TPC track
  Int_t fNtracklets;            // number of attached TRD tracklets
  Int_t fNlayers;               // number of layers where tracklet should exist
  Int_t fNmissingConsecLayers;  // number of missing consecutive layers
  Int_t fNtrackletsOffline;     // number of attached offline TRD tracklets for debugging only
  Int_t fAttachedTracklets[6];  // IDs for attached tracklets sorted by layer
  Bool_t fIsStopped;            // track ends in TRD

  ClassDef(AliHLTTRDTrack,1) //HLT TRD tracker
};


#endif

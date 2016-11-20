#ifndef ALIHLTTRDTRACK1_H
#define ALIHLTTRDTRACK1_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


#include "AliKalmanTrack.h"

class AliHLTGlobalBarrelTrack;
class AliESDtrack;
class AliHLTExternalTrackParam;

//_____________________________________________________________________________
class AliHLTTRDtrack : public AliKalmanTrack
{
 public:

  AliHLTTRDtrack();
  AliHLTTRDtrack(AliESDtrack& t,Bool_t c=kFALSE) throw (const Char_t *);
  AliHLTTRDtrack(AliExternalTrackParam& t ) throw (const Char_t *);
  AliHLTTRDtrack(const AliHLTGlobalBarrelTrack& t);
  AliHLTTRDtrack(const AliHLTTRDtrack& t);
  AliHLTTRDtrack &operator=(const AliHLTTRDtrack& t);

  Int_t GetTPCtrackId() const { return fTPCtrackId; }
  Int_t GetNtracklets() const { return fNtracklets; }
  Int_t GetTracklet(Int_t iLayer) const;

  void AddTracklet(int iLayer, int idx) { fAttachedTracklets[iLayer] = idx; }
  void SetTPCtrackId( Int_t v ){ fTPCtrackId = v;}
  void SetNtracklets( Int_t nTrklts) { fNtracklets = nTrklts; }

  using AliExternalTrackParam::GetPredictedChi2;

  //methods below need to be implemented from abstract base class
  Double_t GetPredictedChi2(const AliCluster *c) { Error("AliHLTTRDtrack", "This is a dummy method and should never be used"); return -1.0; }
  Double_t GetPredictedChi2(const AliCluster *c) const { Error("AliHLTTRDtrack", "This is a dummy method and should never be used"); return -1.0; }
  Double_t GetPredictedChi2(Double_t cy, Double_t cz, Double_t cerr2Y, Double_t cerr2Z) const { Error("AliHLTTRDtrack", "This is a dummy method and should never be used"); return -1.0; }
  Bool_t PropagateTo(Double_t xr, Double_t x0, Double_t rho) { Error("AliHLTTRDtrack", "This is a dummy method and should never be used"); return kFALSE; }
  Bool_t Update(const AliCluster* c, Double_t chi2, Int_t index) { Error("AliHLTTRDtrack", "This is a dummy method and should never be used"); return kFALSE; }

  // virtual method of AliKalmanTrack, it is used when the TRD track is saved in AliESDTrack
  Int_t GetTrackletIndex(Int_t iLayer) const { 
    return GetTracklet(iLayer);
  }


  // convertion to HLT track structure
 
  size_t ConvertTo( AliHLTExternalTrackParam* t ) const;
  void ConvertFrom( const AliHLTExternalTrackParam* t );
  

 protected:

  Int_t fTPCtrackId;            // corresponding TPC track
  Int_t fNtracklets;            // number of attached TRD tracklets
  Int_t fAttachedTracklets[6];  // IDs for attached tracklets sorted by layer

  ClassDef(AliHLTTRDtrack,1) //HLT TRD tracker
};


#endif

#ifndef ALIKALMANTRACK_H
#define ALIKALMANTRACK_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//                          Class AliKalmanTrack
//      fixed the interface for the derived reconstructed track classes 
//            Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch 
//-------------------------------------------------------------------------

#include "AliExternalTrackParam.h"
#include "AliLog.h"
#include "AliPID.h"

class AliCluster;

class AliKalmanTrack : public AliExternalTrackParam {
public:
  AliKalmanTrack();
  AliKalmanTrack(const AliKalmanTrack &t);
  virtual ~AliKalmanTrack(){};
  AliKalmanTrack& operator=(const AliKalmanTrack &o);
  void SetLabel(Int_t lab) {fLab=lab;}

  virtual Double_t GetPredictedChi2(const AliCluster *c) const = 0;
  virtual Bool_t PropagateTo(Double_t xr, Double_t x0, Double_t rho) = 0;
  virtual Bool_t Update(const AliCluster* c, Double_t chi2, Int_t index) = 0;

  Bool_t   IsSortable() const {return kTRUE;}
  Int_t    GetLabel()   const {return fLab;}
  Double_t GetChi2()    const {return fChi2;}
  Double_t GetMass()    const {return fMass;}
  Int_t    GetNumberOfClusters() const {return fN;}
  virtual Int_t GetClusterIndex(Int_t) const { //reserved for AliTracker
    AliWarning("Method must be overloaded !\n");
    return 0;
  } 
  virtual Int_t GetNumberOfTracklets() const {
    AliWarning("Method must be overloaded !");
    return 0;
  }
  virtual Int_t GetTrackletIndex(Int_t) const { //reserved for AliTracker
    AliWarning("Method must be overloaded !");
    return -1;
  } 
  virtual Double_t GetPIDsignal() const {
    AliWarning("Method must be overloaded !\n");
    return 0.;
  }

  virtual Int_t Compare(const TObject *) const {return 0;} 

  void GetExternalParameters(Double_t &xr,Double_t p[5]) const {
    xr=GetX();
    for (Int_t i=0; i<5; i++) p[i]=GetParameter()[i];
  }
  void GetExternalCovariance(Double_t cov[15]) const {
    for (Int_t i=0; i<15; i++) cov[i]=GetCovariance()[i];
  }

  // Time integration (S.Radomski@gsi.de)
  void StartTimeIntegral();
  void SetIntegratedLength(Double_t l) {fIntegratedLength=l;}
  void SetIntegratedTimes(const Double_t *times);

  virtual Bool_t IsStartedTimeIntegral() const {return fStartTimeIntegral;}
  virtual void AddTimeStep(Double_t length);
  void GetIntegratedTimes(Double_t *times, Int_t nspec=AliPID::kSPECIESC) const;
  Double_t GetIntegratedTime(Int_t pdg) const;
  Double_t GetIntegratedLength() const {return fIntegratedLength;}

  void SetNumberOfClusters(Int_t n) {fN=n;} 

  void SetFakeRatio(Float_t ratio) {fFakeRatio=ratio;}
  Float_t  GetFakeRatio()   const {return fFakeRatio;}
  void SetMass(Double_t mass) {fMass=mass;}
  void SetChi2(Double_t chi2) {fChi2=chi2;} 

protected:

  Double32_t fFakeRatio;  // fake ratio
  Double32_t fChi2;       // total chi2 value for this track
  Double32_t fMass;       // mass hypothesis
  Int_t fLab;             // track label
  Int_t fN;               // number of associated clusters

private:
  Bool_t  fStartTimeIntegral;       // indicator wether integrate time
  // variables for time integration (S.Radomski@gsi.de)
  Double32_t fIntegratedTime[AliPID::kSPECIESC];       // integrated time
  Double32_t fIntegratedLength;        // integrated length
  
  ClassDef(AliKalmanTrack,8)    // Reconstructed track
};

#endif



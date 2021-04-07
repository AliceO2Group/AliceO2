// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgTrack.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Track model for the alignment

/**
  * Track model for the alignment: trackParam_t for kinematics
  * proper with number of multiple scattering kinks.
  * Full support for derivatives and residuals calculation
  */

#ifndef ALIALGTRACK_H
#define ALIALGTRACK_H

#include "Align/AliAlgPoint.h"
#include "ReconstructionDataFormats/Track.h"
#include <TObjArray.h>
#include <TArrayD.h>
#include <TArrayI.h>

namespace o2
{
namespace align
{

using trackParam_t = o2::track::TrackParametrizationWithError<double>;

class AliAlgTrack : public trackParam_t, public TObject
{
 public:
  enum { kCosmicBit = BIT(14),
         kFieldONBit = BIT(15),
         kResidDoneBit = BIT(16),
         kDerivDoneBit = BIT(17),
         kKalmanDoneBit = BIT(18) };
  enum { kNKinParBOFF = 4 // N params for ExternalTrackParam part w/o field
         ,
         kNKinParBON = 5 // N params for ExternalTrackParam part with field
         ,
         kParY = 0 // track parameters
         ,
         kParZ,
         kParSnp,
         kParTgl,
         kParQ2Pt
  };
  AliAlgTrack();
  virtual ~AliAlgTrack();
  void DefineDOFs();
  Double_t GetMass() const { return fMass; }
  Double_t GetMinX2X0Pt2Account() const { return fMinX2X0Pt2Account; }
  Int_t GetNPoints() const { return fPoints.GetEntriesFast(); }
  AliAlgPoint* GetPoint(int i) const { return (AliAlgPoint*)fPoints[i]; }
  void AddPoint(AliAlgPoint* p) { fPoints.AddLast(p); }
  void SetMass(double m) { fMass = m; }
  void SetMinX2X0Pt2Account(double v) { fMinX2X0Pt2Account = v; }
  Int_t GetNLocPar() const { return fNLocPar; }
  Int_t GetNLocExtPar() const { return fNLocExtPar; }
  Int_t GetInnerPointID() const { return fInnerPointID; }
  AliAlgPoint* GetInnerPoint() const { return GetPoint(fInnerPointID); }
  //
  virtual void Clear(Option_t* opt = "");
  virtual void Print(Option_t* opt = "") const;
  virtual void DumpCoordinates() const;
  //
  Bool_t PropagateToPoint(trackParam_t& tr, const AliAlgPoint* pnt,
                          int minNSteps, double maxStep, Bool_t matCor, double* matPar = 0);
  Bool_t PropagateParamToPoint(trackParam_t& tr, const AliAlgPoint* pnt, double maxStep = 3);             // param only
  Bool_t PropagateParamToPoint(trackParam_t* trSet, int nTr, const AliAlgPoint* pnt, double maxStep = 3); // params only
  //
  Bool_t CalcResiduals(const double* params = 0);
  Bool_t CalcResidDeriv(double* params = 0);
  Bool_t CalcResidDerivGlo(AliAlgPoint* pnt);
  //
  Bool_t IsCosmic() const { return TestBit(kCosmicBit); }
  void SetCosmic(Bool_t v = kTRUE) { SetBit(kCosmicBit, v); }
  Bool_t GetFieldON() const { return TestBit(kFieldONBit); }
  void SetFieldON(Bool_t v = kTRUE) { SetBit(kFieldONBit, v); }
  Bool_t GetResidDone() const { return TestBit(kResidDoneBit); }
  void SetResidDone(Bool_t v = kTRUE) { SetBit(kResidDoneBit, v); }
  Bool_t GetDerivDone() const { return TestBit(kDerivDoneBit); }
  void SetDerivDone(Bool_t v = kTRUE) { SetBit(kDerivDoneBit, v); }
  Bool_t GetKalmanDone() const { return TestBit(kKalmanDoneBit); }
  void SetKalmanDone(Bool_t v = kTRUE) { SetBit(kKalmanDoneBit, v); }
  //
  void SortPoints();
  Bool_t IniFit();
  Bool_t ResidKalman();
  Bool_t ProcessMaterials();
  Bool_t CombineTracks(trackParam_t& trcL, const trackParam_t& trcU);
  //
  void SetChi2(double c) { fChi2 = c; };
  Double_t GetChi2() const { return fChi2; }
  void SetChi2Ini(double c) { fChi2Ini = c; };
  Double_t GetChi2Ini() const { return fChi2Ini; }
  Double_t GetChi2CosmUp() const { return fChi2CosmUp; }
  Double_t GetChi2CosmDn() const { return fChi2CosmDn; }
  //
  void ImposePtBOff(double pt) { setQ2Pt(1. / pt); }
  // propagation methods
  void CopyFrom(const trackParam_t* etp);
  Bool_t ApplyMatCorr(trackParam_t& trPar, const Double_t* corrDiag, const AliAlgPoint* pnt);
  Bool_t ApplyMatCorr(trackParam_t* trSet, int ntr, const Double_t* corrDiaf, const AliAlgPoint* pnt);
  Bool_t ApplyMatCorr(trackParam_t& trPar, const Double_t* corrpar);
  //
  Double_t GetResidual(int dim, int pntID) const { return fResidA[dim][pntID]; }
  Double_t* GetDResDLoc(int dim, int pntID) const { return &fDResDLocA[dim][pntID * fNLocPar]; }
  Double_t* GetDResDGlo(int dim, int id) const { return &fDResDGloA[dim][id]; }
  Int_t* GetGloParID() const { return fGloParIDA; }
  //
  void SetParams(trackParam_t& tr, double x, double alp, const double* par, Bool_t add);
  void SetParams(trackParam_t* trSet, int ntr, double x, double alp, const double* par, Bool_t add);
  void SetParam(trackParam_t& tr, int par, double val);
  void SetParam(trackParam_t* trSet, int ntr, int par, double val);
  void ModParam(trackParam_t& tr, int par, double delta);
  void ModParam(trackParam_t* trSet, int ntr, int par, double delta);
  //
  void RichardsonDeriv(const trackParam_t* trSet, const double* delta,
                       const AliAlgPoint* pnt, double& derY, double& derZ);
  //
  const Double_t* GetLocPars() const { return fLocParA; }
  void SetLocPars(const double* pars);
  //
 protected:
  //
  Bool_t CalcResidDeriv(double* params, Bool_t invert, int pFrom, int pTo);
  Bool_t CalcResiduals(const double* params, Bool_t invert, int pFrom, int pTo);
  Bool_t FitLeg(trackParam_t& trc, int pFrom, int pTo, Bool_t& inv);
  Bool_t ProcessMaterials(trackParam_t& trc, int pFrom, int pTo);
  //
  void CheckExpandDerGloBuffer(int minSize);
  //
  static Double_t RichardsonExtrap(double* val, int ord = 1);
  static Double_t RichardsonExtrap(const double* val, int ord = 1);
  //
  // ---------- dummies ----------
  AliAlgTrack(const AliAlgTrack&);
  AliAlgTrack& operator=(const AliAlgTrack&);
  //
 protected:
  //trackParam_t
  Int_t fNLocPar;              // number of local params
  Int_t fNLocExtPar;           // number of local params for the external track param
  Int_t fNGloPar;              // number of free global parameters the track depends on
  Int_t fNDF;                  // number of degrees of freedom
  Int_t fInnerPointID;         // ID of inner point in sorted track. For 2-leg cosmics - innermost point of lower leg
  Bool_t fNeedInv[2];          // set if one of cosmic legs need inversion
  Double_t fMinX2X0Pt2Account; // minimum X2X0/pT accumulated between 2 points worth to account
  Double_t fMass;              // assumed mass
  Double_t fChi2;              // chi2 with current residuals
  Double_t fChi2CosmUp;        // chi2 for cosmic upper leg
  Double_t fChi2CosmDn;        // chi2 for cosmic down leg
  Double_t fChi2Ini;           // chi2 with current residuals
  TObjArray fPoints;           // alignment points
  TArrayD fResid[2];           // residuals array
  TArrayD fDResDLoc[2];        // array for derivatives over local params
  TArrayD fDResDGlo[2];        // array for derivatives over global params
  TArrayD fLocPar;             // local parameters array
  TArrayI fGloParID;           // IDs of relevant global params
  Double_t* fResidA[2];        //! fast access to residuals
  Double_t* fDResDLocA[2];     //! fast access to local derivatives
  Double_t* fDResDGloA[2];     //! fast access to global derivatives
  Int_t* fGloParIDA;           //! fast access to relevant global param IDs
  Double_t* fLocParA;          //! fast access to local params
  //
  ClassDef(AliAlgTrack, 2)
};

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParams(trackParam_t& tr, double x, double alp, const double* par, Bool_t add)
{
  // set track params
  const double kDefQ2PtCosm = 1;
  const double kDefG2PtColl = 1. / 0.6;
  params_t tmp;
  std::copy(par, par + kNKinParBON, std::begin(tmp));
  tr.set(x, alp, tmp);
  if (add) { // par is correction to reference params
    for (size_t i = 0; i < kNKinParBON; ++i) {
      const double val = tr.getParam(i) + this->getParam(i);
      tr.setParam(val, i);
    }
  }
  if (!GetFieldON()) {
    const double val = [&]() {
      if (this->IsCosmic()) {
        return kDefQ2PtCosm;
      } else {
        return kDefG2PtColl;
      }
    }();
    tr.setQ2Pt(val); // only 4 params are valid
  }
}

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParams(trackParam_t* trSet, int ntr, double x, double alp, const double* par, Bool_t add)
{
  // set parames for multiple tracks (VECTORIZE THIS)
  if (!add) { // full parameter supplied
    for (int itr = ntr; itr--;)
      SetParams(trSet[itr], x, alp, par, kFALSE);
    return;
  }
  params_t partr{0}; // par is a correction to reference parameter
  for (int i = fNLocExtPar; i--;)
    partr[i] = getParam(i) + par[i];
  for (int itr = ntr; itr--;)
    SetParams(trSet[itr], x, alp, partr.data(), kFALSE);
}

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParam(trackParam_t& tr, int par, double val)
{
  // set track parameter
  tr.setParam(val, par);
}

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParam(trackParam_t* trSet, int ntr, int par, double val)
{
  // set parames for multiple tracks (VECTORIZE THIS)
  for (int i = 0; i < ntr; ++i) {
    SetParam(trSet[i], par, val);
  }
}

//____________________________________________________________________________________________
inline void AliAlgTrack::ModParam(trackParam_t& tr, int par, double delta)
{
  // modify track parameter
  const auto val = tr.getParam(par) + delta;
  SetParam(tr, par, val);
}

//____________________________________________________________________________________________
inline void AliAlgTrack::ModParam(trackParam_t* trSet, int ntr, int par, double delta)
{
  // modify track parameter (VECTORIZE THOS)
  for (size_t i = 0; i < ntr; ++i) {
    ModParam(trSet[i], par, delta);
  }
}

//______________________________________________
inline void AliAlgTrack::CopyFrom(const trackParam_t* etp)
{
  // assign kinematics
  set(etp->getX(), etp->getAlpha(), etp->getParams(), etp->getCov().data());
}
} // namespace align
} // namespace o2
#endif
